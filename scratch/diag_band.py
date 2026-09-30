"""Coarse ink map of a band, to see whether the checkbox ink exists at all."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from util.image_utils import _load_gray, _threshold_gray  # noqa: E402

path = sys.argv[1]
x0, x1, y0, y1 = (int(v) for v in sys.argv[2:6])
step = int(sys.argv[6]) if len(sys.argv) > 6 else 6

gray = _load_gray(path)
bin_img = _threshold_gray(gray)
print(f"gray range in band: min={gray[y0:y1, x0:x1].min()} max={gray[y0:y1, x0:x1].max()}")
print(f"dark pixel fraction after threshold: {float(np.mean(bin_img[y0:y1, x0:x1] > 0)):.4f}")
print(f"\nASCII map of x[{x0},{x1}) y[{y0},{y1}) step={step}  ('#'=ink, '.'=blank)")
for y in range(y0, y1, step):
    row = ""
    for x in range(x0, x1, step):
        blk = bin_img[y:y + step, x:x + step]
        row += "#" if np.any(blk > 0) else "."
    print(f"{y:5d} {row}")
