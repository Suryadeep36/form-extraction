import cv2
import numpy as np

def _load_image(image_or_path):
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    img = cv2.imread(image_or_path)
    if img is None:
        raise FileNotFoundError(image_or_path)
    return img


def _load_gray(image_or_path):
    img = _load_image(image_or_path)
    return _as_gray(img)


def _as_gray(img):
    """Coerce any image to a single-channel 8-bit array.

    `adaptiveThreshold` and friends assert `CV_8UC1`, so a colour (BGR) or
    float input raises `(-215) Assertion failed (src.type() == CV_8UC1)`.
    Normalising here keeps that failure from ever reaching a caller.
    """
    if img is None:
        return img
    # Cast before cvtColor: OpenCV's colour conversion only accepts uint8 /
    # uint16, so a float64 frame would fail the (different) CV_8UC1 assert.
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        # 4-channel BGRA: drop alpha rather than letting cvtColor fail.
        code = (
            cv2.COLOR_BGRA2GRAY
            if img.shape[2] == 4
            else cv2.COLOR_BGR2GRAY
        )
        img = cv2.cvtColor(img, code)
    elif img.ndim > 3:
        img = img.reshape(img.shape[-3], img.shape[-1])
    return img

def _rotate_image(image, angle):
    if abs(angle) < 1e-6:
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        image,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderValue=(255, 255, 255),
    )


def _estimate_skew(image_or_path):
    gray = _load_gray(image_or_path)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return angle

def _threshold_gray(gray):
    return cv2.adaptiveThreshold(
        _as_gray(gray),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )