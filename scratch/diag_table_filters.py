"""Per-candidate filter instrumentation + band occupancy, for table validation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import config  # noqa: E402
from util.image_utils import _load_image  # noqa: E402
from util.line_utils import (  # noqa: E402
    detect_line_segments,
    merge_collinear_horizontal,
    merge_collinear_vertical,
)
from util.table_utils import detect_table_candidates  # noqa: E402
from service.table_service import _is_single_column_stack  # noqa: E402
from service.table_model_service import get_table_model_service  # noqa: E402
from service.detection_service import detect_vertical_lines  # noqa: E402
from util.geometry_utils import _iou as iou  # noqa: E402


def bands_inside(bbox, horizontals, pad=4):
    """Horizontal rules strictly inside bbox -> band edges."""
    ys = sorted({
        round(h["y"]) for h in horizontals
        if bbox[1] - pad <= h["y"] <= bbox[3] + pad
    })
    return ys


def main():
    path = sys.argv[1]
    image = _load_image(path)
    from service.ocr_service import get_ocr_data
    elements = get_ocr_data(image)
    H, W = image.shape[:2]

    hs = merge_collinear_horizontal(
        detect_line_segments(image, orientation="horizontal", min_length_ratio=0.20)
    )
    vs = merge_collinear_vertical(
        detect_line_segments(image, orientation="vertical", min_length_ratio=0.15)
    )
    cv_cands = detect_table_candidates(image, elements=elements)
    ms = get_table_model_service()
    model_cands = ms.detect_tables(image)
    print(f"page {W}x{H}  cv_candidates={len(cv_cands)}  model_candidates={len(model_cands)}")

    print("\n=== per model candidate ===")
    for mod in model_cands:
        bb = mod["bbox"]
        bh, bw = bb[3] - bb[1], bb[2] - bb[0]
        m = next((c for c in cv_cands if iou(bb, c["bbox"]) > 0.5), None)
        ys = bands_inside(bb, hs)
        edges = [bb[1]] + ys + [bb[3]]
        bands = list(zip(edges[:-1], edges[1:]))
        occ = []
        for y1, y2 in bands:
            n = sum(
                1 for e in elements
                if e.get("type") == "text" and e["bbox"]
                and y1 - 6 <= (e["bbox"][1] + e["bbox"][3]) / 2 <= y2 + 6
            )
            occ.append(n)
        lv = [v for v in detect_vertical_lines(image) if bb[0] - 2 <= v["x"] <= bb[2] + 2]
        print(f"\n  bbox={[round(v) for v in bb]} w={round(bw)} h={round(bh)} conf={mod.get('confidence')}")
        print(f"    FILTER1 height>=max(90,{H*0.05:.0f})={max(90, H*0.05):.0f} -> "
              f"{'PASS' if bh >= max(90, H * 0.05) else 'DROP'}")
        print(f"    FILTER2 cv_match={'yes n_int=' + str(m.get('n_intersections')) if m else 'NONE'}"
              f" -> {'DROP' if m and m.get('n_intersections', 0) <= 4 else 'pass'}")
        print(f"    FILTER2b single_column_stack="
              f"{'DROP' if m and _is_single_column_stack(m) else 'pass'}")
        ncol = sum(1 for e in elements if e.get("type") == "text" and e["bbox"]
                   and (e["text"] or "").strip().endswith(":")
                   and bb[1] - 60 <= (e["bbox"][1] + e["bbox"][3]) / 2 <= bb[3])
        print(f"    FILTER3 colon_labels={ncol} long_verticals={len(lv)}"
              f" -> {'DROP' if ncol >= 3 and len(lv) <= 3 else 'pass'}")
        print(f"    band_edges={ys}")
        print(f"    band_text_counts={occ}  empty_bands={sum(1 for c in occ if c == 0)}/{len(occ)}")


if __name__ == "__main__":
    main()

import os
sys.stdout.flush()
os._exit(0)
