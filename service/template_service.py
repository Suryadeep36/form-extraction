"""
Two-pass template registration + filled-form extraction.

Pass 1 - register_template(empty_form_image):
    Preprocess + analyze the EMPTY form with the existing pipeline, turn its
    blank value regions into a reusable template, and persist both the template
    JSON and the warped reference image.

Pass 2 - extract_filled(template, filled_form_image):
    Preprocess + analyze the FILLED form, warp it into the template coordinate
    space via Homography alignment, then recover each field's value.  Checkbox
    regions are decided by OpenCV pixel density, everything else by OCR text
    that lands inside the pre-registered value region.

Both passes reuse the existing document pipeline (`analyze_document`), so
templates are truly dynamic: any form the pipeline can parse works, with no
hardcoded labels or positions.  The only coupling to the empty form is the
layout itself, which is exactly what we want for a template.
"""

import os
import json
import re
import uuid
import base64
import shutil

import cv2
import numpy as np

from service.document_service import (
    analyze_document,
    preprocess_document,
    build_document_representation,
)
from util.alignment_utils import estimate_transform
from util.form_utils import _repair_date_range
from util.template_utils import (
    build_template_fields,
    build_template_tables,
    find_label_anchor,
)
from util.config import UPLOAD_DIR

TEMPLATES_DIR = os.path.join(UPLOAD_DIR, "templates")


# ---------------------------------------------------------------------------
# Persistence (file-based, no DB dependency)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Convert numpy scalars/arrays to plain JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.ndarray,)):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def _templates_dir():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    return TEMPLATES_DIR


def _template_path(template_id):
    return os.path.join(_templates_dir(), f"{template_id}.json")


def _reference_path(template_id):
    return os.path.join(_templates_dir(), f"{template_id}_ref.jpg")


def save_template(template):
    """Persist a template dict + its reference image. Returns template_id."""
    template_id = template.get("id") or str(uuid.uuid4())
    template["id"] = template_id
    ref_image = template.pop("reference_image", None)
    with open(_template_path(template_id), "w") as f:
        meta = {k: v for k, v in template.items() if k != "reference_image"}
        json.dump(meta, f, indent=2)
    if ref_image is not None:
        ok, buf = cv2.imencode(".jpg", ref_image)
        if ok:
            with open(_reference_path(template_id), "wb") as f:
                f.write(buf.tobytes())
    return template_id


def load_template(template_id, with_reference=False):
    path = _template_path(template_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        template = json.load(f)
    if with_reference:
        ref_path = _reference_path(template_id)
        if os.path.exists(ref_path):
            template["reference_image"] = cv2.imread(ref_path)
    return template


def get_template_image(template_id):
    """
    Raw bytes of a template's reference (preprocessed blank-form) image, or
    None when the template does not exist / has no stored image.
    """
    if not os.path.exists(_template_path(template_id)):
        return None
    ref_path = _reference_path(template_id)
    if not os.path.exists(ref_path):
        return None
    with open(ref_path, "rb") as f:
        return f.read()


def list_templates():
    if not os.path.isdir(_templates_dir()):
        return []
    out = []
    for fname in sorted(os.listdir(_templates_dir())):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_templates_dir(), fname)) as f:
                t = json.load(f)
            out.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "source_filename": t.get("source_filename"),
                "field_count": len(t.get("fields", [])),
                "table_count": len(t.get("tables", [])),
                "created_at": t.get("created_at"),
            })
        except Exception as e:
            print(f"[TEMPLATE] skip {fname}: {e}")
    return out


# ---------------------------------------------------------------------------
# Pass 1 - registration
# ---------------------------------------------------------------------------

def register_template(image_path, name=None):
    """
    Take an EMPTY form image and build a template from its blank structure.

    Returns:
        {
            "template_id", "name", "source_filename", "fields": [...],
            "image_width", "image_height", "alignment_reference": data-url,
            "created_at",
        }
    """
    info = preprocess_document(image_path)
    doc_rep = build_document_representation(info["image"], image_path=image_path)
    w, h = doc_rep["image_width"], doc_rep["image_height"]
    fields = build_template_fields(doc_rep, w, h)
    tables = build_template_tables(doc_rep, w, h)

    template = {
        "id": str(uuid.uuid4()),
        "name": name or os.path.splitext(os.path.basename(image_path))[0],
        "source_filename": os.path.basename(image_path),
        "image_width": w,
        "image_height": h,
        "fields": fields,
        "tables": tables,
        "static_noise": doc_rep.get("static_noise") or [],
        "created_at": None,
        "reference_image": info["image"],
    }
    import datetime
    template["created_at"] = datetime.datetime.utcnow().isoformat()

    template_id = save_template(template)
    # Encode reference image as data URL for the API response (not stored in JSON).
    ok, buf = cv2.imencode(".jpg", info["image"])
    ref_url = None
    if ok:
        ref_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    template.pop("reference_image", None)
    template["template_id"] = template_id
    template["reference_image_data_url"] = ref_url
    print(f"[TEMPLATE] registered {template_id} with {len(fields)} fields")
    return template


# ---------------------------------------------------------------------------
# Pass 2 - extraction
# ---------------------------------------------------------------------------

def extract_filled(template, image_path):
    """
    Extract key/value pairs from a FILLED form using a registered template.

    Returns:
        {
            "template_id", "fields": [ {label, value, ...} ... ],
            "alignment": {...transform...},
            "warped_image_data_url": optional,
        }
    """
    info = preprocess_document(image_path)
    doc_rep = build_document_representation(info["image"], image_path=image_path)

    # --- Align filled -> template coordinate space (Homography) ----------
    elements = doc_rep["elements"]
    ref_image = template.get("reference_image")
    if ref_image is None:
        # No stored reference; fall back to identity alignment.
        warped = info["image"]
        transform = {"method": "identity", "matches": 0, "inlier_ratio": 0.0}
        aligned_elements = elements
    else:
        warped, transform = estimate_transform(ref_image, info["image"])
        if len(warped.shape) == 2:
            warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        else:
            warped_bgr = warped
        # Transform each element bbox through the homography into template space.
        H = transform.get("homography")
        aligned_elements = _warp_element_bbox(elements, H, doc_rep["image_width"])
        # Fall back to raw elements when alignment was identity.
        if transform["method"] == "identity":
            aligned_elements = elements

    # --- Extract each field's value ---------------------------------------
    w = template.get("image_width") or doc_rep["image_width"]
    h = template.get("image_height") or doc_rep["image_height"]
    # The template value bboxes are normalized to the TEMPLATE page; we work
    # in the warped image space, which equals the template space, so convert
    # normalized -> pixels using the template dims.
    norm_to_px = lambda vb: [vb[0]*w, vb[1]*h, vb[2]*w, vb[3]*h]

    # The elements are in the aligned space; build a workspace object with
    # raw pixel bboxes and the warped image for checkbox density.
    global_labels_px = []
    for f in template.get("fields", []):
        if f.get("label_bbox"):
            global_labels_px.append(norm_to_px(f["label_bbox"]))

    # Static background text recorded at registration (watermarks, printed
    # decorations that sit inside a value area on the blank form).  Stored
    # normalized; convert to the warped/template pixel space.
    static_noise_px = []
    for nb in (template.get("static_noise") or []):
        if nb and len(nb) == 4:
            static_noise_px.append(norm_to_px(nb))

    workspace = {
        "image": warped if transform.get("homography") is not None else info["image"],
        "element_px": aligned_elements,
        "checkboxes": doc_rep.get("checkboxes") or [],
        "global_labels_px": global_labels_px, # Injected here
        "static_noise_px": static_noise_px,
    }

    extracted = []
    for field in template["fields"]:
        vb_px = norm_to_px(field["value_bbox"])
        # Make a copy of the field with pixel value_bbox so the helper works
        # with real coordinates. value_bboxes stores one box per physical
        # line for multi-line fields (e.g. "Address :"); fall back to the
        # union box when a field has no line breakdown.
        fcopy = dict(field)
        fcopy["value_bbox_px"] = vb_px
        fcopy["label_bbox_px"] = norm_to_px(field["label_bbox"]) if field.get("label_bbox") else None
        fcopy["value_boxes_px"] = [
            norm_to_px(b) for b in (field.get("value_bboxes") or [field["value_bbox"]])
        ]
        result = _extract_one(fcopy, workspace, w, h)
        # Surface the COMMITTED template value_bbox (the tight field region),
        # not the OCR extraction window `_extract_one` expands vertically.
        result["bbox"] = field["value_bbox"]
        result["bboxes"] = field.get("value_bboxes") or [field["value_bbox"]]
        extracted.append(result)

    # --- Extract each table's cells --------------------------------------
    tables_out = _extract_template_tables(template, aligned_elements, w, h)

    # --- Response ----------------------------------------------------------
    out = {
        "template_id": template.get("id"),
        "template_name": template.get("name"),
        "fields": extracted,
        "tables": tables_out,
        "alignment": _json_safe(transform),
        "element_count": len(doc_rep["elements"]),
    }
    # The template value bboxes are normalized to the TEMPLATE page; the warped
    # image shown in the UI must live in that same coordinate space or the
    # frontend overlay cannot line up. Homography/similarity warps already use
    # (w, h); the identity fallback leaves the raw scan at its own size, so
    # resize it to the template canvas before encoding.
    display = workspace["image"]
    if tuple(display.shape[:2]) != (h, w):
        display = cv2.resize(display, (w, h))

    # Optionally include the warped image for debugging.
    try:
        ok, buf = cv2.imencode(".jpg", display)
        if ok:
            out["warped_image_data_url"] = (
                "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            )
    except Exception as e:
        print(f"[TEMPLATE] warped encode failed: {e}")
    return out


def _warp_element_bbox(elements, H, width):
    """Transform each element's pixel bbox through homography H."""
    if H is None:
        return elements
    out = []
    for e in elements:
        bbox = e["bbox"]
        corners = np.float32([
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]],
        ]).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, H).reshape(4, 2)
            xs, ys = warped[:, 0], warped[:, 1]
            new_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        except cv2.error:
            new_bbox = list(bbox)
        ne = dict(e)
        ne["bbox"] = new_bbox
        # Word-level boxes are stored in the same original OCR space as the
        # element bbox; they must be transformed through the same homography
        # or the compound-header/word assignment in `_assign_ocr_to_cells`
        # would resolve words against misaligned (un-warped) coordinates and
        # drop them into the wrong table row.
        if e.get("words"):
            warped_words = []
            for w in e["words"]:
                wb = w.get("bbox")
                if not wb or len(wb) != 4:
                    continue
                wc = np.float32([
                    [wb[0], wb[1]], [wb[2], wb[1]],
                    [wb[2], wb[3]], [wb[0], wb[3]],
                ]).reshape(-1, 1, 2)
                try:
                    ww = cv2.perspectiveTransform(wc, H).reshape(4, 2)
                    wxs, wys = ww[:, 0], ww[:, 1]
                    wbbox = [float(wxs.min()), float(wys.min()), float(wxs.max()), float(wys.max())]
                except cv2.error:
                    wbbox = list(wb)
                nw = dict(w)
                nw["bbox"] = wbbox
                warped_words.append(nw)
            if warped_words:
                ne["words"] = warped_words
        out.append(ne)
    return out


# def _extract_one(field_copy, workspace, w, h):
#     """Extract a single field respecting checkbox regions."""
#     vb = field_copy["value_bbox_px"]
#     cb_match = None
#     best_iou = 0.0
#     for cb in workspace["checkboxes"]:
#         cb_bbox = cb["bbox"]
#         ix1 = max(vb[0], cb_bbox[0]); iy1 = max(vb[1], cb_bbox[1])
#         ix2 = min(vb[2], cb_bbox[2]); iy2 = min(vb[3], cb_bbox[3])
#         if ix2 <= ix1 or iy2 <= iy1:
#             continue
#         inter = (ix2 - ix1) * (iy2 - iy1)
#         union = (vb[2]-vb[0])*(vb[3]-vb[1]) + (cb_bbox[2]-cb_bbox[0])*(cb_bbox[3]-cb_bbox[1]) - inter
#         r = inter / union if union else 0.0
#         if r > best_iou:
#             best_iou, cb_match = r, cb
#     if cb_match is not None and best_iou >= 0.15:
#         from util.checkbox_utils import _mark_kind, _classify_window
#         from util.image_utils import _threshold_gray
#         bin_img = _threshold_gray(workspace["image"])
#         mark = _mark_kind(bin_img, cb_match["bbox"])
#         checked = mark in ("X", "filled", "tick")
#         return {
#             "label": field_copy["label"],
#             "value": "True" if checked else "False",
#             "value_type": "boolean",
#             "confidence": cb_match.get("confidence"),
#             "bbox": [round(x / w, 5) for x in vb],
#             "source": "checkbox",
#             "checked": checked,
#             "mark_type": mark,
#         }

#     # OCR text extraction inside the value bbox.
#     #
#     # Sampling the raw detection region bleeds text from neighboring rows
#     # (the region is full-width/tall while a value is short).  Instead anchor
#     # the window to the matched LABEL on the filled form:
#     #   x from label-right -> region-right
#     #   y clipped tightly around the label's row band.
#     # This keeps the value crisp and avoids catching the next row's label.
#     anchor = find_label_anchor(field_copy["label"], workspace["element_px"])

#     value_window = None
#     if anchor is not None:
#         lb = anchor["bbox"]
#         label_h = max(lb[3] - lb[1], 12)
#         row_cy = (lb[1] + lb[3]) / 2.0
#         # Values sit on the label's own line; keep the vertical band tight to
#         # the anchor row so an adjacent row (e.g. a title above a field) never
#         # bleeds in.
#         row_half = max(label_h * 0.65, 10)
#         gap = max(label_h * 0.25, 4)
#         x_left = lb[2] + gap
#         # Cap the window width so a full-page underline cannot swallow text
#         # belonging to independent right-hand content.
#         max_width = max((vb[2] - vb[0]) * 0.55, 160)
#         x_right = min(vb[2], x_left + max_width)
#         x_left = min(max(x_left, vb[0]), x_right)
#         value_window = [x_left, row_cy - row_half, x_right, row_cy + row_half]
#     else:
#         # No reliable label anchor: use the template region (with tight pad).
#         value_window = [
#             vb[0],
#             vb[1] + (vb[3] - vb[1]) * 0.08,
#             vb[2],
#             vb[3] - (vb[3] - vb[1]) * 0.08,
#         ]

#     def _center_in_region(bbox, region):
#         cx = (bbox[0] + bbox[2]) / 2.0
#         cy = (bbox[1] + bbox[3]) / 2.0
#         return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]

#     matched = []
#     for e in workspace["element_px"]:
#         bbox = e["bbox"]
#         if not _center_in_region(bbox, value_window):
#             continue
#         # Exclude the anchor label itself and any text starting before the
#         # anchor's right edge (a value cannot live left of its own label).
#         if anchor is not None and (bbox[1] + bbox[3]) / 2.0 is not None:
#             if e["id"] == anchor["id"]:
#                 continue
#             emid_x = (bbox[0] + bbox[2]) / 2.0
#             lb = anchor["bbox"]
#             if emid_x < lb[0]:
#                 continue
#         matched.append(e)
#     if not matched:
#         return {
#             "label": field_copy["label"],
#             "value": None,
#             "value_type": None,
#             "confidence": None,
#             "bbox": [round(x / w, 5) for x in vb],
#             "source": "blank",
#         }
#     matched.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
#     value = " ".join(e["text"].strip() for e in matched).strip()
#     # The printed label often ends with ":" but OCR can glue it onto the start
#     # of a value (or leave a lone colon breadcrumb); the colon belongs to the
#     # label, never to the value.
#     value = re.sub(r"^\s*:+\s*|\s*:+\s*$", "", value)
#     value = _repair_date_range(value)
#     # Drop a lone anchor breadcrumb such as ':' if it's the only thing.
#     if value in (":", ":", ""):
#         value = None
#     conf = float(np.mean([e.get("confidence", 0.0) for e in matched]))
#     return {
#         "label": field_copy["label"],
#         "value": value or None,
#         "value_type": None,
#         "confidence": round(conf, 4),
#         "bbox": [round(x / w, 5) for x in vb],
#         "source": "ocr",
#     }

def _extract_checkbox_group(field_copy, workspace, w, h):
    """Extract a checkbox_group field: evaluate each option's checkbox ring.

    Option bboxes are registered-normalized and live in the template page
    space, which equals the warped image space here, so map them through
    norm_to_px (template dims).  The CHEAKBOX bbox of each option carries the
    mark, evaluated with the shared density/threshold classifier.  Returns the
    list of checked option texts (multi-select for e.g. Hobbies, single for
    e.g. Gender).
    """
    from util.checkbox_utils import _mark_kind
    from util.image_utils import _load_gray, _threshold_gray
    bin_img = _threshold_gray(_load_gray(workspace["image"]))
    options_out = []
    checked = []
    for opt in field_copy.get("options") or []:
        ob = opt.get("bbox")
        if not ob or len(ob) != 4:
            continue
        px = [ob[0] * w, ob[1] * h, ob[2] * w, ob[3] * h]
        # The registered bbox is the checkbox RING, not its interior, so the
        # ring's own frame would count as "ink" and make every box look filled.
        # Shrink toward the centre to sample just the inside of the box where
        # a real X / tick / fill mark lives.
        ix1, iy1 = int(px[0]), int(px[1])
        ix2, iy2 = int(px[2]), int(px[3])
        shrink = max(1, int(round(min(iy2 - iy1, ix2 - ix1) * 0.22)))
        mark = _mark_kind(bin_img, [ix1 + shrink, iy1 + shrink, ix2 - shrink, iy2 - shrink])
        is_checked = mark in ("X", "filled", "tick")
        options_out.append({
            "text": opt.get("text") or "",
            "bbox": ob,
            "checked": is_checked,
            "mark_type": mark,
        })
        if is_checked:
            checked.append((opt.get("text") or "").strip())
    return {
        "label": field_copy.get("label") or "",
        "value": [c for c in checked if c],
        "value_type": "checkbox_group",
        "confidence": None,
        "bbox": field_copy["value_bbox"],
        "checked": bool(checked),
        "options": options_out,
    }


def _extract_one(field_copy, workspace, w, h):
    """Extract a single field respecting checkbox regions and homography alignment."""
    # Checkbox option groups bypass the single-checkbox IoU / OCR paths: each
    # option's registered checkbox bbox carries the mark to evaluate.
    if field_copy.get("kind") == "checkbox_group":
        return _extract_checkbox_group(field_copy, workspace, w, h)
    vb = field_copy["value_bbox_px"]
    cb_match = None
    best_iou = 0.0
    
    # --- 1. Checkbox Processing ---
    for cb in workspace["checkboxes"]:
        cb_bbox = cb["bbox"]
        ix1 = max(vb[0], cb_bbox[0]); iy1 = max(vb[1], cb_bbox[1])
        ix2 = min(vb[2], cb_bbox[2]); iy2 = min(vb[3], cb_bbox[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = (vb[2]-vb[0])*(vb[3]-vb[1]) + (cb_bbox[2]-cb_bbox[0])*(cb_bbox[3]-cb_bbox[1]) - inter
        r = inter / union if union else 0.0
        if r > best_iou:
            best_iou, cb_match = r, cb
            
    if cb_match is not None and best_iou >= 0.15:
        from util.checkbox_utils import _mark_kind, _classify_window
        from util.image_utils import _threshold_gray
        bin_img = _threshold_gray(workspace["image"])
        mark = _mark_kind(bin_img, cb_match["bbox"])
        checked = mark in ("X", "filled", "tick")
        return {
            "label": field_copy["label"],
            "value": "True" if checked else "False",
            "value_type": "boolean",
            "confidence": cb_match.get("confidence"),
            "bbox": [round(x / w, 5) for x in vb],
            "source": "checkbox",
            "checked": checked,
            "mark_type": mark,
        }

    # --- 2. Homography-Trusted Text Extraction ---
    boxes = field_copy.get("value_boxes_px") or [vb]

    def _center_in_region(bbox, region):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]

    def _overlaps_any_box(bbox, boxes):
        for b in boxes:
            ix1 = max(bbox[0], b[0]); iy1 = max(bbox[1], b[1])
            ix2 = min(bbox[2], b[2]); iy2 = min(bbox[3], b[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > 0 and inter / area > 0.3:
                return True
        return False

    def _overlaps_any_label(bbox):
        return _overlaps_any_box(bbox, workspace.get("global_labels_px", []))

    def _excluded(token_bbox):
        # This field's own printed label can sit just inside the value window
        # ("Address :" starts a line that is also the value line); never read
        # our own label text back as the value.
        own_label = field_copy.get("label_bbox_px")
        if own_label and _overlaps_any_box(token_bbox, [own_label]):
            return True
        # Static background text (watermarks, decorations) that was present on
        # the BLANK form and overlaps a value area must not leak into values.
        if _overlaps_any_box(token_bbox, workspace.get("static_noise_px", [])):
            return True
        # Never exclude content squarely inside this field's value area even
        # when it also overlaps a (full-width) registered label.
        if any(_center_in_region(token_bbox, b) for b in boxes):
            return False
        return _overlaps_any_label(token_bbox)

    # Flatten each OCR element into its constituent tokens. Word-level boxes
    # split multi-field rows ("23 Out of 50") so that "23" lands in one blank
    # and "50" in the other; elements without word data stay as a single token.
    def _tokens(elem):
        words = [w for w in (elem.get("words") or []) if (w.get("text") or "").strip()]
        return words if words else [elem]

    collected = []
    seen_keys = set()

    for box in boxes:
        x1, y1, x2, y2 = box
        box_height = y2 - y1

        if box_height < 15:
            value_window = [x1, y1 - 25, x2, y2 + 4]
        else:
            value_window = [x1, y1 - 8, x2, y2 + 8]

        # The printed underline may begin to the RIGHT of where the user
        # actually starts writing (e.g. "Date of Birth: __/__/____" where the
        # pre-printed "/ /" separators sit before the underline). Extend the
        # capture window leftward to just past the field's label so the first
        # digits of such values are not missed.
        label_px = field_copy.get("label_bbox_px")
        if label_px and field_copy.get("kind") == "underline":
            label_x2 = label_px[2]
            label_h = max(label_px[3] - label_px[1], 12)
            min_x = label_x2 + max(label_h * 0.3, 4)
            if min_x < x1:
                value_window[0] = min_x

        raw_matched = []
        for e in workspace["element_px"]:
            for tok in _tokens(e):
                tb = tok.get("bbox")
                if not tb or len(tb) != 4:
                    continue
                if not _center_in_region(tb, value_window):
                    continue
                if _excluded(tb):
                    continue
                raw_matched.append((e, tok, tb))

        if not raw_matched:
            continue

        raw_matched.sort(key=lambda t: (t[2][1] + t[2][3]) / 2.0)

        lines = []
        current_line = []
        last_cy = None

        for (e, tok, tb) in raw_matched:
            cy = (tb[1] + tb[3]) / 2.0
            if last_cy is None or abs(cy - last_cy) < 15:
                current_line.append((e, tok, tb))
                if last_cy is None:
                    last_cy = cy
            else:
                lines.append(current_line)
                current_line = [(e, tok, tb)]
                last_cy = cy
        if current_line:
            lines.append(current_line)

        per_line_contract = (
            field_copy.get("value_bboxes") is not None
            and field_copy.get("kind") == "underline"
        )
        if len(lines) > 1 and (per_line_contract or box_height < 45):
            target_y = (y1 + y2) / 2.0
            best_line = None
            min_dist = float('inf')

            for line in lines:
                line_cy = np.mean([(t[2][1] + t[2][3]) / 2.0 for t in line])
                dist = abs(line_cy - target_y)
                if dist < min_dist:
                    min_dist = dist
                    best_line = line
            picked = best_line
        else:
            picked = [t for line in lines for t in line]

        for (e, tok, tb) in picked:
            key = (id(e), id(tok))
            if key not in seen_keys:
                seen_keys.add(key)
                collected.append((e, tok, tb))

    if not collected:
        return {
            "label": field_copy["label"],
            "value": None,
            "value_type": None,
            "confidence": None,
            "bbox": [round(x / w, 5) for x in vb],
            "source": "blank",
        }

    # --- 4. Rebuild tokens in reading order --------------------------------
    # OCR can return a line's per-word boxes in reverse order (emails read
    # "com . gmail @ user" instead of "user@gmail.com").  When EVERY word of a
    # source element landed in this field, trust the element's own recognized
    # text - it preserves the true order - instead of resurrecting it from the
    # shuffled word boxes.  Partial captures keep the per-word boxes.
    groups = {}
    for e, tok, tb in collected:
        key = id(e)
        if key not in groups:
            words = [w for w in (e.get("words") or []) if (w.get("text") or "").strip()]
            groups[key] = {"element": e, "word_count": len(words), "tokens": []}
        groups[key]["tokens"].append((tok, tb))

    merged = []
    for g in groups.values():
        if g["word_count"] > 1 and len(g["tokens"]) == g["word_count"]:
            e = g["element"]
            merged.append(({"bbox": list(e["bbox"]), "text": (e.get("text") or "").strip()}, e["bbox"]))
        else:
            merged.extend(g["tokens"])

    # --- 5. Value Cleanup ---
    matched = sorted(
        merged,
        key=lambda t: ((t[1][1] + t[1][3]) / 2.0, t[1][0]),
    )
    value = " ".join(t[0]["text"].strip() for t in matched).strip()
    
    value = re.sub(r"^\s*:+\s*|\s*:+\s*$", "", value)
    value = _repair_date_range(value)
    
    if value in (":", ""):
        value = None
        
    conf = float(np.mean([t[0].get("confidence", 0.0) for t in matched]))

    return {
        "label": field_copy["label"],
        "value": value,
        "value_type": None,
        "confidence": round(conf, 4),
        "bbox": [round(x / w, 5) for x in vb],
        "source": "ocr",
    }

def _extract_template_tables(template, aligned_elements, w, h):
    """
    Refill each template table's cells using the aligned filled form.

    The template stored the empty form's cell grid (normalized bboxes + static
    header text). We project that grid onto the warped filled form and reuse
    the pipeline's deterministic OCR->cell mapping so every cell gets the text
    that landed inside it (printed headers first, then the filled-in values).

    Returns a list of:
        {
            "id", "bbox", "n_rows", "n_cols", "n_cells",
            "structure_source",
            "cells": [
                {"row", "column", "row_span", "column_span",
                 "bbox": normalized, "template_text": static text or None,
                 "text": value found on the filled form or None}
            ],
        }
    """
    from service.table_service import _assign_ocr_to_cells

    norm_to_px = lambda vb: [vb[0]*w, vb[1]*h, vb[2]*w, vb[3]*h]

    tables_out = []
    for table in template.get("tables", []):
        t = dict(table)
        tb_px = norm_to_px(table["bbox"]) if table.get("bbox") else None
        cells = []
        for i, cell in enumerate(table.get("cells", [])):
            nc = dict(cell)
            nc["id"] = f"cell_{i:03d}"
            nc["template_text"] = nc.get("text")
            nc["bbox"] = norm_to_px(cell["bbox"])
            cells.append(nc)

        if tb_px is not None and cells:
            try:
                _assign_ocr_to_cells(aligned_elements, cells, tb_px)
            except Exception as e:
                print(f"[TEMPLATE] table cell assignment failed: {e}")

        for cell in cells:
            c = cell["bbox"]
            cell["bbox"] = [
                round(c[0] / w, 5),
                round(c[1] / h, 5),
                round(c[2] / w, 5),
                round(c[3] / h, 5),
            ]
        t["cells"] = cells
        tables_out.append(t)

    return tables_out