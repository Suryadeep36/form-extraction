"""Batch: table candidate + band-occupancy stats for every form image, one process."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.image_utils import _load_image  # noqa: E402
from util.line_utils import (  # noqa: E402
    detect_line_segments,
    merge_collinear_horizontal,
)
from util.table_utils import detect_table_candidates  # noqa: E402
from service.table_model_service import get_table_model_service  # noqa: E402
from service.ocr_service import get_ocr_data  # noqa: E402
from service.detection_service import detect_vertical_lines  # noqa: E402
from service.table_service import _is_single_column_stack  # noqa: E402
from util.geometry_utils import _iou  # noqa: E402

ms = get_table_model_service()
ms.detect_tables(_load_image(glob.glob("extract-form-forms/*.png")[0]))  # warm models

files = sorted(glob.glob("extract-form-forms/*.png")) + ["this.jpeg"]
for path in files:
    image = _load_image(path)
    H, W = image.shape[:2]
    elements = get_ocr_data(image)
    hs = merge_collinear_horizontal(
        detect_line_segments(image, orientation="horizontal", min_length_ratio=0.20)
    )
    cv_cands = detect_table_candidates(image, elements=elements)
    try:
        model_cands = ms.detect_tables(image)
    except Exception as e:
        print(f"{os.path.basename(path)}: detect_tables failed {e}")
        continue
    kept_now = 0
    print(f"\n########## {os.path.basename(path)}  ({W}x{H}) model={len(model_cands)} cv={len(cv_cands)}")
    for mod in model_cands:
        bb = mod["bbox"]
        ys = sorted({round(h["y"]) for h in hs if bb[1] - 4 <= h["y"] <= bb[3] + 4})
        if len(ys) < 2:
            tb, total = 0, 0
        else:
            occ = []
            for y1, y2 in zip(ys[:-1], ys[1:]):
                n = sum(
                    1 for e in elements
                    if e.get("type") == "text" and e["bbox"]
                    and y1 - 6 <= (e["bbox"][1] + e["bbox"][3]) / 2 <= y2 + 6
                )
                occ.append(n)
            tb = sum(1 for c in occ if c > 0)
            total = len(occ)
        m = next((c for c in cv_cands if _iou(bb, c["bbox"]) > 0.5), None)
        drop = (
            (bb[3] - bb[1]) < max(90, H * 0.05)
            or (m is not None and m.get("n_intersections", 0) <= 4)
            or (m is not None and _is_single_column_stack(m))
        )
        if not drop:
            kept_now += 1
        print(
            f"  bbox={[round(v) for v in bb]} h={round(bb[3]-bb[1])} "
            f"text_bands={tb}/{total} empty={total-tb} "
            f"cv={'y' if m else 'n'} -> {'KEEP' if not drop else 'drop'}"
        )
    print(f"  ==> current kept: {kept_now}")
    sys.stdout.flush()

os._exit(0)
