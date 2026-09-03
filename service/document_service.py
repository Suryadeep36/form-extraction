import numpy as np
import cv2
import json
import base64
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
    detect_checkboxes_ocr_anchored,
    detect_rectangular_regions
)
from util.checkbox_utils import (
    strip_leading_option_mark,
)
from service.field_service import (
    process_form_fields
)
from service.table_service import (
    detect_and_fuse_tables,
    finalize_tables,
)
from util.perspective_utils import (
    correct_perspective,
)
from util.geometry_utils import (
    _iou
)

def build_document_representation(image, image_path=None):
    """
    Run OCR + layout analysis and assemble the internal document graph.

    The image is assumed to be already preprocessed (perspective-corrected,
    oriented, deskewed). All coordinates produced here live in this processed
    image's coordinate system.

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
            },
            "fields": [...],     # first-class form fields
            "tables": [...],     # first-class tables with cells
            "input_regions": [...],
            "preprocessing": {...},
        }
    """
    if isinstance(image, np.ndarray):
        image_height, image_width = image.shape[:2]
    else:
        loaded = _load_image(image)
        image_height, image_width = loaded.shape[:2]
        image = loaded

    print("[OCR] Running PaddleOCR...")
    raw_elements = get_ocr_data(image)
    print(f"[OCR] Elements: {len(raw_elements)}")

    print("Detecting checkboxes...")
    # NOTE: detect on pre-strip texts so the OCR leading-mark signal can see
    # the original selection marks ("XPartially Boatable", "_NO IfNO...").
    checkboxes = detect_checkboxes_ocr_anchored(image, raw_elements)
    print(f"Checkboxes: {len(checkboxes)}")

    # Strip selection-mark glyphs that OCR glued onto the front of option
    # labels (e.g. "XPartially Boatable" -> "Partially Boatable").
    # Guards keep legitimate leading tokens ("X Latitude North") intact.
    for element in raw_elements:
        text = element.get("text") or ""
        cleaned, _changed = strip_leading_option_mark(text)
        if _changed:
            element["text"] = cleaned
        for w in element.get("words") or []:
            wtext = w.get("text") or ""
            cw, cchanged = strip_leading_option_mark(wtext)
            if cchanged:
                w["text"] = cw

    # ---- Tables stage 1: detect + fuse (bbox-only) -----------------------
    fused, cv_candidates = detect_and_fuse_tables(image, raw_elements)
    valid_tables = []
    for t in fused:
        bbox = t["bbox"]
        table_height = bbox[3] - bbox[1]
        
        cv_match = next((c for c in cv_candidates if _iou(bbox, c["bbox"]) > 0.5), None)
        
        if cv_match and cv_match.get("n_intersections", 0) == 0:
            print(f"[TABLE FILTER] Dropped fake table (0 intersections): {bbox}")
            continue
            
        if table_height < 60: 
            print(f"[TABLE FILTER] Dropped fake table (too short): {bbox}")
            continue
            
        valid_tables.append(t)
        
    fused = valid_tables
    # ==========================================

    # 2. Build the exclusion boxes using ONLY the valid tables
    table_bboxes = [t["bbox"] for t in fused]

    # ---- Form fields (input regions + compound splitting) ----------------
    fields, input_regions, elements = process_form_fields(
        image,
        raw_elements,
        table_bboxes=table_bboxes,
        checkboxes=checkboxes,
    )
    print(f"[OCR] Final elements: {len(elements)}")

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

    doc_rep = {
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
        "fields": fields,
        "input_regions": input_regions,
        "_cv_tables": cv_candidates,
    }

    # ---- Tables stage 2: structure recognition + OCR-to-cell -------------
    print("[TABLE] Recognizing structure...")
    tables = finalize_tables(fused, doc_rep, image)
    doc_rep["tables"] = tables

    # Drop internal-only keys before propagating further.
    doc_rep.pop("_cv_tables", None)

    return doc_rep

def analyze_document(image_or_path, image_path=None):
    """
    End-to-end: preprocess -> OCR -> layout -> candidates.
    Returns (doc_rep, candidates). The LLM stage lives in main.py.

    `doc_rep` additionally carries `processed_image_data_url`: a base64
    data URL of the perspective-corrected / oriented / deskewed image that
    all reported coordinates are normalized against. The frontend must
    render this image (not the raw upload) so overlay bboxes line up.
    """
    info = preprocess_document(image_or_path)

    doc_rep = build_document_representation(
        info["image"],
        image_path=image_path,
    )
    doc_rep["preprocessing"] = info["preprocessing"]

    try:
        ok, buf = cv2.imencode(".jpg", info["image"])
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            doc_rep["processed_image_data_url"] = (
                "data:image/jpeg;base64," + b64
            )
    except Exception as e:
        print(f"[PROCESS] corrected-image encode failed: {e}")

    candidates = generate_field_candidates(doc_rep)

    from util.debug_utils import write_debug_document
    write_debug_document(info["image"], doc_rep, image_path or "analyze")

    return doc_rep, candidates

def interpret_document(doc_rep, candidates):
    prompt = build_structure_prompt(doc_rep, candidates)

    print(prompt)

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
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
    Prepare the image for OCR/CV.

    Pipeline:
        1. perspective correction (page-boundary detection + warp)
        2. orientation correction (content rotation classifier)
        3. deskew (small-angle rotation)

    Returns:
        {
            "image": ndarray,       # in the final corrected coordinate system
            "path": path of the corrected image (or None),
            "preprocessing": {
                "orientation_corrected": int (0, 90, 180, 270),
                "deskew_angle": float,
                "perspective_correction": {...},
            }
        }
    """
    image = _load_image(image_or_path)

    # ---- 1. perspective correction -------------------------------------
    image, perspective_info = correct_perspective(image)

    # ---- 2. orientation --------------------------------------------------
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

    # ---- 3. deskew -------------------------------------------------------
    skew = _estimate_skew(image)
    if abs(skew) >= 0.75:
        image = _rotate_image(image, skew)

    info = {
        "image": image,
        "path": output_path,
        "preprocessing": {
            "orientation_corrected": orientation,
            "deskew_angle": round(skew, 3),
            "perspective_correction": perspective_info["perspective_correction"],
        },
    }

    if output_path:
        cv2.imwrite(output_path, image)

    return info

