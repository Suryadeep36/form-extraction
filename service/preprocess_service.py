import numpy as np
import cv2

from util.image_utils import (
    _load_image,
    _estimate_skew,
    _rotate_image,
)
from util.ocr_model_utils import (
    _get_orientation_model
)

def preprocess_document(image_or_path, output_path=None):
    """
    Correct document orientation and deskew.

    Returns:
        {
            "image": ndarray,
            "path": path of the corrected image (or None),
            "orientation_corrected": int (0, 90, 180, 270),
            "deskew_angle": float
        }
    """
    image = _load_image(image_or_path)

    orientation = 0
    try:
        result = list(_get_orientation_model().predict(image))
        label = str(result[0]["label_names"][0])
        orientation = int(label)
        k = orientation // 90
        if k:
            image = np.rot90(image, k=k)
    except Exception as e:
        print(f"Orientation classification failed: {e}")

    skew = _estimate_skew(image)
    if abs(skew) >= 0.75:
        image = _rotate_image(image, skew)

    info = {
        "image": image,
        "path": output_path,
        "orientation_corrected": orientation,
        "deskew_angle": round(skew, 3),
    }

    if output_path:
        cv2.imwrite(output_path, image)

    return info
