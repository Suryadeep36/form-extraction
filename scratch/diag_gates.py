"""Per-gate instrumentation of _checkbox_candidates for a single image."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from service.ocr_service import get_ocr_data  # noqa: E402
from util import checkbox_group_utils as g  # noqa: E402
from util.image_utils import _load_image, _load_gray  # noqa: E402

path = sys.argv[1]
rgb = _load_image(path)
gray = _load_gray(path)
elements = get_ocr_data(rgb)
hs = [e["bbox"][3] - e["bbox"][1] for e in elements if e.get("bbox")]
mh = float(np.median(hs))
print(f"median_text_h={mh}  elements={len(elements)}")

bin_img = g._threshold_gray(gray)
low, high = g._checkbox_scale(mh)
print(f"accepted side range = {low}..{high}")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
h, w = closed.shape[:2]
m = max(2, int(round(min(h, w) * 0.006)))
closed = closed.copy()
closed[:m, :] = 0
closed[-m:, :] = 0
closed[:, :m] = 0
closed[:, -m:] = 0
hmask, vmask = g._grid_line_masks(bin_img, min_len=3.0 * high)

y_lo, y_hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, h)
print(f"\n--- contour pass, candidates whose TOP is in y=[{y_lo},{y_hi}) ---")
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
reasons = {}
shown = 0
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    if not (y_lo <= y < y_hi):
        continue
    side, big = min(cw, ch), max(cw, ch)
    if side < low or big > high:
        continue
    aspect = cw / float(ch) if ch else 0
    if not (0.75 <= aspect <= 1.4):
        continue
    bbox = [float(x), float(y), float(x + cw), float(y + ch)]
    if not g._is_selection_box(bin_img, int(x), int(y), int(cw), int(ch)):
        r = g._ring_density(bin_img, x, y, x + cw, y + ch)
        c2 = g._classify_window(bin_img, x, y, x + cw, y + ch)
        bands = g._ring_side_darkness(bin_img, x, y, x + cw, y + ch)
        why = f"not_selection_box ring={r} classify={c2} bands={bands}"
    elif g._has_inner_word(bbox, elements):
        why = "inner_word"
    elif g._is_grid_cell(bbox, hmask, vmask):
        why = "grid_cell"
    else:
        why = "ACCEPTED"
    reasons[why] = reasons.get(why, 0) + 1
    if shown < 25:
        print(f"  bbox={bbox} side={side} aspect={aspect:.2f}  {why}")
        shown += 1
print("\n--- rejection tally ---")
for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  {v:4d}  {k}")

print(f"\nfinal _checkbox_candidates -> {len(g._checkbox_candidates(gray, elements, mh))}")
