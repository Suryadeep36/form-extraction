"""Dump raw contour boundingRects near a target point, with no gates."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from util.image_utils import _load_gray, _threshold_gray  # noqa: E402

path = sys.argv[1]
tx, ty = int(sys.argv[2]), int(sys.argv[3])
radius = int(sys.argv[4]) if len(sys.argv) > 4 else 200

gray = _load_gray(path)
bin_img = _threshold_gray(gray)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
h, w = closed.shape[:2]
m = max(2, int(round(min(h, w) * 0.006)))
closed = closed.copy()
closed[:m, :] = 0
closed[-m:, :] = 0
closed[:, :m] = 0
closed[:, -m:] = 0

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"total external contours: {len(contours)}")
print(f"contours overlapping a {radius}px box around ({tx},{ty}):")
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    if x <= tx <= x + cw and y <= ty <= y + ch:
        print(f"  bbox=({x},{y},{cw},{ch})  side={min(cw,ch)} big={max(cw,ch)}"
              f"  aspect={cw/float(ch) if ch else 0:.2f}  area={cv2.contourArea(c):.0f}")

print("\nCCOMP holes overlapping that box:")
cc, hier = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    hier = hier[0]
    for i, hole in enumerate(cc):
        if hier[i][3] < 0:
            continue
        hx, hy, hw, hh = cv2.boundingRect(hole)
        if hx - 10 <= tx <= hx + hw + 10 and hy - 10 <= ty <= hy + hh + 10:
            ring = max(3, min(8, int(round(min(hw, hh) * 0.12))))
            print(f"  hole=({hx},{hy},{hw},{hh}) -> expanded=({hx-ring},{hy-ring},"
                  f"{hw+2*ring},{hh+2*ring})")
