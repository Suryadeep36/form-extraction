"""
Pretrained table detection + structure recognition (Paddle stack).

The model load is lazy and cached: the models are created once (on first
request that actually needs them) and reused for every subsequent request.

Device selection:
    TABLE_MODEL_DEVICE=auto -> CUDA > GPU > CPU (paddle resolver)
    TABLE_MODEL_DEVICE=cpu  -> force CPU
    TABLE_MODEL_DEVICE=gpu  -> force gpu:0

The entire module degrades gracefully: when the models cannot be loaded or a
request fails, the caller receives [] and `available` is set to False so the
CV path takes over.
"""

import threading

import numpy as np
import cv2

import util.config as config


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device():
    requested = config.TABLE_MODEL_DEVICE
    if requested and requested != "auto":
        if requested == "cpu":
            return "cpu"
        if requested == "gpu":
            return "gpu:0"
        return requested

    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            count = paddle.device.cuda.device_count()
            if count > 0:
                return "gpu:0"
    except Exception:
        pass

    return "cpu"


# ---------------------------------------------------------------------------
# Structure-token parsing
# ---------------------------------------------------------------------------

_ROW_OPEN_TOKENS = {"<tr>", "<tr"}
_COLSPAN_ATTR = "colspan"
_ROWSPAN_ATTR = "rowspan"


def _parse_span_attr(token, attr):
    """
    Parse `colspan="2"` (also single-quoted / unquoted) out of a structure
    token like `<td colspan="2">`. Returns 1 when absent.
    """
    import re

    match = re.search(attr + r'\s*=\s*["\']?(\d+)[\'"]?', token)
    if not match:
        return 1
    try:
        return max(1, int(match.group(1)))
    except ValueError:
        return 1


def parse_table_structure(cell_bboxes, structure_tokens):
    """
    Convert SLANet-style `structure` tokens + cell bboxes into a row/column
    grid with spans.

    Args:
        cell_bboxes: list of [x1, y1, x2, y2] (or 4-point quads) aligned with
                     the `<td...>` tokens in `structure_tokens`.
        structure_tokens: e.g. ['<tr>', '<td>', '</td>', '<td colspan="2">',
                          '</td>', '</tr>', ...]

    Returns:
        (cells, max_row, max_col)
        cells: list of {
            "id", "bbox", "row", "column",
            "row_span", "column_span", "cell_type"
        }
    """
    cells = []
    if isinstance(cell_bboxes, np.ndarray):
        cell_bboxes = cell_bboxes.tolist()
    if not isinstance(cell_bboxes, (list, tuple)):
        cell_bboxes = list(cell_bboxes)

    # Normalize: paddlex returns either a flat list of (quad|xyxy) bboxes or a
    # list-of-lists. Flatten one level only when entries are themselves lists.
    flat_bboxes = []
    for item in cell_bboxes:
        if isinstance(item, (list, tuple, np.ndarray)) and len(item) and isinstance(
            item[0], (list, tuple, np.ndarray)
        ):
            flat_bboxes.extend(item)
        else:
            flat_bboxes.append(item)
    cell_bboxes = flat_bboxes

    bbox_iter = iter(cell_bboxes)
    current_row = 0
    current_col = 0

    # SLANet emits each physical cell once. To build a dense grid we also
    # remember max column count per logical row start.
    row_max_col = {}

    row_open = False
    in_header = False
    cell_index = 0

    def _emit_next_bbox():
        nonlocal cell_index
        try:
            raw = next(bbox_iter)
        except StopIteration:
            return None
        raw = np.asarray(raw)
        if raw.size == 8:
            xs = raw[0::2]
            ys = raw[1::2]
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        else:
            bbox = [float(v) for v in raw]
        return bbox

    for token in structure_tokens:
        lowered = str(token).strip().lower().replace(" ", "")

        if lowered.startswith("<thead"):
            in_header = True
            continue
        if lowered.startswith("</thead"):
            in_header = False
            continue

        if lowered.startswith("</tr"):
            if row_open:
                row_max_col[current_row] = max(row_max_col.get(current_row, 0),
                                              current_col)
            current_row += 1
            current_col = 0
            row_open = False
            continue

        if lowered.startswith("<tr"):
            row_open = True
            continue

        if lowered.startswith("<td") or lowered.startswith("<th"):
            bbox = _emit_next_bbox()

            if bbox is None:
                break

            col_span = _parse_span_attr(lowered, _COLSPAN_ATTR)
            row_span = _parse_span_attr(lowered, _ROWSPAN_ATTR)

            cell_type = "header" if (in_header or lowered.startswith("<th")) else "body"

            cells.append(
                {
                    "id": f"cell_{cell_index:03d}",
                    "bbox": bbox,
                    "row": current_row,
                    "column": current_col,
                    "row_span": row_span,
                    "column_span": col_span,
                    "cell_type": cell_type,
                }
            )
            cell_index += 1

            current_col += col_span
            if current_row not in row_max_col or current_col > row_max_col[current_row]:
                row_max_col[current_row] = current_col

    max_col = max(row_max_col.values(), default=0)
    return cells, current_row, max_col


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TableModelService:
    """
    Lazy, cached wrapper around the pretrained Paddle table stack.

      * table detection   : PicoDet_layout_1x_table (full-page layout)
      * structure         : SLANeXt_wired / SLANeXt_wireless per table crop
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self._available = False
        self._device = "cpu"
        self._detection_model = None
        self._wired_model = None
        self._wireless_model = None
        self._load_error = None

    @property
    def available(self):
        return self._available

    @property
    def device(self):
        return self._device

    @property
    def load_error(self):
        return self._load_error

    # -- lazy loading ---------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return self._available
        with self._lock:
            if self._loaded:
                return self._available
            self._loaded = True
            try:
                self._load()
            except Exception as e:
                self._load_error = str(e)
                self._available = False
                print(f"[TABLE-MODEL] Load failed ({e}). Falling back to CV.")
            return self._available

    def _load(self):
        if not config.TABLE_MODEL_ENABLED:
            print("[TABLE-MODEL] Disabled by configuration.")
            self._available = False
            return

        from paddlex import create_model

        self._device = _resolve_device()
        print(f"[TABLE-MODEL] Device: {self._device}")

        self._detection_model = create_model(
            config.TABLE_DETECTION_MODEL, device=self._device
        )

        # Wired structure model covers bordered tables; wireless covers
        # borderless layouts. Both are loaded lazily when first needed.
        self._wired_model = create_model(
            config.TABLE_STRUCTURE_MODEL_WIRED, device=self._device
        )
        self._wireless_model = create_model(
            config.TABLE_STRUCTURE_MODEL_WIRELESS, device=self._device
        )

        self._available = True
        print("[TABLE-MODEL] Pretrained table models ready.")
        self._ensure_prediction_works()

    def _ensure_prediction_works(self):
        """
        Fire a tiny inference on a blank image so download/compile issues
        surface at load time rather than mid-request.
        """
        try:
            probe = np.full((64, 64, 3), 255, dtype=np.uint8)
            self._detection_model.predict(probe)
        except Exception as e:
            self._available = False
            self._load_error = f"probe inference failed: {e}"
            print(f"[TABLE-MODEL] Probe failure: {e}")

    # -- detection -------------------------------------------------------

    def detect_tables(self, image):
        """
        Detect tables on a full page using the pretrained model.

        Returns list of:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "source": "model",
                "sources": ["model"],
                "label": "table",
            }
        """
        if not self._ensure_loaded():
            return []

        try:
            import paddle

            results = self._detection_model.predict(image)
            candidates = []
            for res in results:
                dt = res if isinstance(res, dict) else res.get("res", {})
                records = dt.get("boxes")
                if records is None:
                    records = dt.get("records")
                if records is None:
                    records = dt.get("res", {}).get("boxes")
                if records is None and isinstance(dt, (list, tuple)):
                    records = dt

                for rec in records or []:
                    if isinstance(rec, dict):
                        label = str(rec.get("label", "")).lower()
                        score = float(rec.get("score", 0.0) or 0.0)
                        coord = rec.get("coordinate")
                    else:
                        continue

                    if label != "table":
                        continue
                    if score < config.TABLE_DETECTION_THRESHOLD:
                        continue

                    pts = np.asarray(coord, dtype=np.float64)
                    if pts.ndim >= 2 and pts.size >= 8:
                        # Quad (2N points) -> bounding rectangle.
                        flat = pts.reshape(-1)
                        x1 = float(flat[0::2].min())
                        y1 = float(flat[1::2].min())
                        x2 = float(flat[0::2].max())
                        y2 = float(flat[1::2].max())
                    elif pts.size == 4:
                        # Flat xyxy: [x1, y1, x2, y2].
                        x1, y1, x2, y2 = (float(v) for v in pts.tolist())
                    else:
                        continue

                    candidates.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(score, 4),
                            "source": "model",
                            "sources": ["model"],
                            "label": label,
                        }
                    )
            print(f"[TABLE-MODEL] Candidates: {len(candidates)}")
            return candidates
        except Exception as e:
            print(f"[TABLE-MODEL] Detection inference failed: {e}")
            return []

    # -- structure ---------------------------------------------------------

    def recognize_structure(self, table_image, wired=True):
        """
        Recognize the cell grid of one table crop.

        Returns:
            (cells, rows, columns) where cells include bbox in the crop's
            coordinate system.
        """
        if not self._ensure_loaded():
            return [], [], []

        model = self._wired_model if wired else self._wireless_model

        try:
            results = list(model.predict(table_image))
            if not results:
                return [], [], []

            res = results[0]
            data = res.get("res", res)
            bboxes = data.get("bbox") or []
            structures = data.get("structure") or []

            if not bboxes or not structures:
                return [], [], []

            cells, rows, cols = parse_table_structure(
                bboxes,
                structures,
            )
            return cells, rows, cols
        except Exception as e:
            print(f"[TABLE-MODEL] Structure inference failed: {e}")
            return [], [], []


_SERVICE_LOCK = threading.Lock()
_SERVICE = None


def get_table_model_service():
    """Process-wide singleton so the models load exactly once."""
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = TableModelService()
    return _SERVICE