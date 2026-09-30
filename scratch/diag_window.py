"""Why does the hole-pass window get rejected? Compare exact ring vs expanded window."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from util.image_utils import _load_gray, _threshold_gray  # noqa: E402
from util import checkbox_group_utils as g  # noqa: E402

path = sys.argv[1]
gray = _load_gray(path)
bin_img = _threshold_gray(gray)

hmask, vmask = g._grid_line_masks(bin_img, min_len=3.0 * 68)

windows = [
    ("exact ring        ", 191, 402, 50, 50),
    ("hole-pass expanded", 185, 396, 62, 62),
    ("tight 1px outset  ", 190, 401, 52, 52),
]
for name, x, y, w, h in windows:
    rd = g._ring_density(bin_img, x, y, x + w, y + h)
    cw = g._classify_window(bin_img, x, y, x + w, y + h)
    bands = g._ring_side_darkness(bin_img, x, y, x + w, y + h)
    sel = g._is_selection_box(bin_img, x, y, w, h)
    grid = g._is_grid_cell([x, y, x + w, y + h], hmask, vmask)
    print(f"{name} ({x},{y},{w},{h})")
    print(f"    _ring_density  outer={rd[0] if rd else None} center={rd[1] if rd else None}")
    print(f"    edge_dark      {cw[0] if cw else None}")
    print(f"    side_bands     {bands}")
    print(f"    _is_selection_box -> {sel}    _is_grid_cell -> {grid}")
