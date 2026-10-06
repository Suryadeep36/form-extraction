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
    """Estimate page skew in degrees from TEXT BASELINE projection.

    The previous implementation ran `minAreaRect` over every ink pixel, which
    measures the orientation of the page's largest ink mass.  That is the text
    baseline only when text dominates the page; when large selection squares,
    logos or graphic bars dominate, their arrangement determines the returned
    angle instead.  On a level form whose squares happen to form a diagonal the
    method reported +2.6 deg and the pipeline rotated an already-straight page.

    Projection-profile variance is the standard alternative: text rows are long
    horizontal bands, so rotating by the true skew maximises the sharpness of
    the row-occupancy profile.  Non-text blobs (hollow squares) contribute a
    diffuse background that barely responds to rotation, so they cannot tilt
    the estimate.  Returns 0.0 when the page has too little ink or when no
    angle in the search range clearly beats level.
    """
    gray = _load_gray(image_or_path)
    if gray.size == 0:
        return 0.0
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    ink = (binary > 0)
    # Text is the only thing that yields many separate rows; without a decent
    # row population any angle is indistinguishable from noise.
    row_profile = ink.sum(axis=1).astype(np.float64)
    peaks = int(((row_profile[1:-1] > row_profile[:-2]) &
                 (row_profile[1:-1] > row_profile[2:])).sum())
    if ink.sum() < 200 or peaks < 3:
        return 0.0

    h, w = ink.shape
    # Centre-crop so a full-width graphic border cannot dominate the profile.
    ch = max(1, int(h * 0.9))
    cw = max(1, int(w * 0.9))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    sub = ink[y0:y0 + ch, x0:x0 + cw]

    def _score(angle_deg):
        # Rotate about the centre and score the row-occupancy variance: the
        # sharper the text bands, the higher the variance of the profile.
        mat = cv2.getRotationMatrix2D(
            (cw / 2.0, ch / 2.0), float(angle_deg), 1.0
        )
        rot = cv2.warpAffine(
            sub.astype(np.uint8), mat, (cw, ch),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        prof = rot.sum(axis=1).astype(np.float64)
        return float(np.var(np.diff(prof)))

    # Coarse sweep then refine; the true skew of a scanned page is small.
    coarse = np.arange(-5.0, 5.01, 0.5)
    best = max(coarse, key=_score)
    fine = np.arange(best - 0.5, best + 0.51, 0.05)
    best = float(max(fine, key=_score))

    # Reject angles that do not actually beat the level page by a clear
    # margin: a level scan must stay level rather than be nudged by noise.
    if _score(0.0) > 0.0 and _score(best) <= _score(0.0) * 1.02:
        return 0.0
    # `best` is the skew OF THE PAGE, but callers pass this straight to
    # `cv2.getRotationMatrix2D` as the correction to apply, so negate it.
    return -best


def _threshold_gray(gray):
    return cv2.adaptiveThreshold(
        _as_gray(gray),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )