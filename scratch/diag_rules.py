"""Structural map: rules + OCR text, to identify the real tables."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.image_utils import _load_gray, _threshold_gray  # noqa: E402
from util.table_utils import detect_horizontal_lines, detect_vertical_lines  # noqa: E402

path = sys.argv[1]
gray = _load_gray(path)
bin_img = _threshold_gray(gray)
H, W = gray.shape[:2]
print(f"page: {W} x {H}")

hlines = detect_horizontal_lines(gray)
vlines = detect_vertical_lines(gray)
print(f"\n=== HORIZONTAL RULES ({len(hlines)}) ===")
for ln in sorted(hlines, key=lambda d: d["bbox"][1]):
    x1, y1, x2, y2 = [int(v) for v in ln["bbox"]]
    print(f"  y={y1:>5} x=[{x1:>5},{x2:>5}] w={x2-x1:>5}  thick={y2-y1}")

print(f"\n=== VERTICAL RULES ({len(vlines)}) ===")
for ln in sorted(vlines, key=lambda d: d["bbox"][0]):
    x1, y1, x2, y2 = [int(v) for v in ln["bbox"]]
    print(f"  x={x1:>5} y=[{y1:>5},{y2:>5}] h={y2-y1:>5}  thick={x2-x1}")

print("\n=== ASCII ink map (coarse) ===")
sy, sx = max(1, H // 70), max(1, W // 100)
for y in range(0, H, sy):
    row = ""
    for x in range(0, W, sx):
        blk = bin_img[y:y + sy, x:x + sx]
        row += "#" if np.mean(blk > 0) > 0.18 else ("+" if np.any(blk > 0) else ".")
    print(f"{y:>5} {row}")
