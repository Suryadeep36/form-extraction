import numpy as np
import cv2
import json
from fastapi import HTTPException
from service.candidate_field_service import (
    generate_field_candidates
)
from service.llm_service import (
    build_structure_prompt
)
from service.gemini_service import (
    gemini_client
)
from util.image_utils import (
    _load_image,
    _estimate_skew,
    _rotate_image
)
from util.ocr_model_utils import (
    _get_orientation_model
)
from service.ocr_service import (
    get_ocr_data,
    normalize_ocr_elements
)
from service.detection_service import (
    detect_horizontal_lines,
    detect_vertical_lines,
    detect_checkboxes,
    detect_rectangular_regions
)

def build_document_representation(image, image_path=None):
    """
    Run OCR + layout analysis and assemble the internal document graph.

    Returns:
        {
            "image_path": str | None,
            "image_width": int,
            "image_height": int,
            "elements": [...],   # text elements (t000, ...)
            "regions": [...],    # r000, ...
            "checkboxes": [...], # c000, ...
            "lines": {
                "horizontal": [...],
                "vertical": [...]
            }
        }
    """
    if isinstance(image, np.ndarray):
        image_height, image_width = image.shape[:2]
    else:
        loaded = _load_image(image)
        image_height, image_width = loaded.shape[:2]
        image = loaded

    print("Running PaddleOCR...")
    raw_elements = get_ocr_data(image)
    print(f"OCR items: {len(raw_elements)}")

    print("Normalizing compound OCR elements (Pass 1)...")
    elements = normalize_ocr_elements(raw_elements)
    print(f"Normalized OCR items: {len(elements)}")

    print("Detecting horizontal lines...")
    horizontal_lines = detect_horizontal_lines(image)
    print(f"Horizontal lines: {len(horizontal_lines)}")

    print("Detecting vertical lines...")
    vertical_lines = detect_vertical_lines(image)
    print(f"Vertical lines: {len(vertical_lines)}")

    print("Detecting rectangular regions...")
    raw_regions = detect_rectangular_regions(image)
    regions = [
        {
            "id": f"r{i:03d}",
            "bbox": r["bbox"],
            "x": r["x"],
            "y": r["y"],
            "width": r["width"],
            "height": r["height"],
            "area": r["area"],
        }
        for i, r in enumerate(raw_regions)
    ]
    print(f"Regions: {len(regions)}")

    print("Detecting checkboxes...")
    checkboxes = detect_checkboxes(image)
    print(f"Checkboxes: {len(checkboxes)}")

    return {
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "elements": elements,
        "regions": regions,
        "checkboxes": checkboxes,
        "lines": {
            "horizontal": horizontal_lines,
            "vertical": vertical_lines,
        },
    }

def analyze_document(image_or_path, image_path=None):
    """
    End-to-end: preprocess -> OCR -> layout -> candidates.
    Returns (doc_rep, candidates). The LLM stage lives in main.py.
    """
    info = preprocess_document(image_or_path)

    doc_rep = build_document_representation(
        info["image"],
        image_path=image_path,
    )
    doc_rep["preprocessing"] = {
        "orientation_corrected": info["orientation_corrected"],
        "deskew_angle": info["deskew_angle"],
    }

    candidates = generate_field_candidates(doc_rep)

    return doc_rep, candidates

def interpret_document(doc_rep, candidates):
    prompt = build_structure_prompt(doc_rep, candidates)

    print(prompt)

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are a document understanding system. "
                            "You interpret relationships and return JSON "
                            "referencing only the IDs you are given. "
                            "Never invent coordinates or IDs.\n\n"
                            + prompt
                        )
                    }
                ]
            }
        ],
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        }
    )

    llm_output = response.text

    print("Gemini response received.")
    print(llm_output)
    if not llm_output:
        raise HTTPException(
            status_code=500,
            detail="LLM returned an empty response."
        )

    try:
        return json.loads(llm_output)

    except json.JSONDecodeError as e:
        print("Invalid JSON returned by LLM:")
        print(llm_output)

        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON: " + str(e)
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

