"""
Robust document-to-template alignment.

Pass 2 of the template pipeline warps a filled form into the exact coordinate
space of a pre-registered empty template *before* any value region is read.
This absorbs scan-to-scan variation: differing page size, margins, translation
and even a small in-plane rotation / skew.

Strategy (best-effort, degrading gracefully):
    1. Homography via ORB feature detection + descriptor matching + RANSAC.
       This is the primary path and the only one that can correct perspective
       distortion (skew / rotation / scale) of a photographed or rescanned
       page.
    2. Similarity transform fallback: when not enough reliable point matches
       exist, estimate scale + rotation + translation from the matched points.
    3. Identity fallback: if matching fails entirely, return the image unchanged
       (the caller's value-region lookup still works for identical scans).

The returned ``transform`` dict describes the warp so callers can also map the
empty-template coordinate system back onto the filled source.
"""

import cv2
import numpy as np

from util.image_utils import _load_gray

# Tunable knobs for the matcher.
_ORB_MAX_FEATURES = 4000
_GOOD_RATIO = 0.75          # Lowe's ratio for keeping correspondences
_MIN_MATCHES = 8            # minimum point matches to attempt an estimate


def _preprocess(gray):
    """Contrast-normalize the gray page for feature detection."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_describe(gray):
    orb = cv2.ORB_create(nfeatures=_ORB_MAX_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def _match_descriptors(desc_src, desc_dst):
    if desc_src is None or desc_dst is None or len(desc_src) < 4:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        raw = matcher.knnMatch(desc_src, desc_dst, k=2)
    except cv2.error:
        return []
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < _GOOD_RATIO * n.distance:
            good.append(m)
    return good


def estimate_transform(ref_image, filled_image):
    """
    Estimate the geometric transform mapping ``filled_image`` into ``ref_image``
    coordinate space.

    Returns (warped_filled, transform) where transform is a dict:
        {
            "method": "homography" | "similarity" | "identity",
            "homography": 3x3 ndarray | None,
            "matrix": 2x3 affine | None,      # for similarity
            "inlier_ratio": float,
            "matches": int,
        }
    If estimation fails, a white image is NOT returned; instead the filled
    image is returned unchanged with method "identity".
    """
    ref_gray = _preprocess(_load_gray(ref_image))
    fill_gray = _preprocess(_load_gray(filled_image))
    ref_h, ref_w = ref_gray.shape

    # --- 1. Homography (primary) -----------------------------------------
    kp_ref, des_ref = _detect_describe(ref_gray)
    kp_fill, des_fill = _detect_describe(fill_gray)
    good = _match_descriptors(des_fill, des_ref)

    if len(good) >= _MIN_MATCHES:
        src_pts = np.float32([kp_fill[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            inliers = int(mask.sum())
            ratio = inliers / max(len(good), 1)
            if inliers >= _MIN_MATCHES and ratio >= 0.25:
                try:
                    warped = cv2.warpPerspective(
                        _load_gray(filled_image), H, (ref_w, ref_h),
                        flags=cv2.INTER_LINEAR,
                        borderValue=255,
                    )
                    return warped, {
                        "method": "homography",
                        "homography": H,
                        "inlier_ratio": round(ratio, 3),
                        "matches": inliers,
                    }
                except cv2.error:
                    pass

    # --- 2. Similarity fallback (scale + rotation + translation) ----------
    if len(good) >= 3:
        src_pts = np.float32([kp_fill[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good])
        try:
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, cv2.RANSAC, 5.0
            )
            if M is not None:
                n_in = int(inliers.sum()) if inliers is not None else 0
                warped = cv2.warpAffine(
                    _load_gray(filled_image), M, (ref_w, ref_h),
                    flags=cv2.INTER_LINEAR, borderValue=255,
                )
                return warped, {
                    "method": "similarity",
                    "matrix": M,
                    "inlier_ratio": round(n_in / max(len(good), 1), 3),
                    "matches": n_in,
                }
        except cv2.error:
            pass

    # --- 3. Identity fallback ---------------------------------------------
    return _load_gray(filled_image), {
        "method": "identity",
        "homography": None,
        "matrix": None,
        "inlier_ratio": 0.0,
        "matches": 0,
    }


def align_to_template(ref_image, filled_image):
    """
    Public helper: warp ``filled_image`` onto ``ref_image``.
    Returns (warped_gray_image, transform_dict).
    """
    return estimate_transform(ref_image, filled_image)
