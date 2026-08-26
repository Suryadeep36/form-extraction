import numpy as np

from util.ocr_model_utils import (
    _get_ocr_engine,
    _get_orientation_model,
)

from service.llm_service import (
    run_pass_1_llm_splitter
)

from util.geometry_utils import (
    _interpolate_sub_bbox
)

def get_ocr_data(image_or_path):
    ocr = _get_ocr_engine()
    result = ocr.predict(image_or_path)

    if not result:
        return []

    page = result[0]

    if hasattr(page, "keys"):
        data = page.get("res", page)
    else:
        return []

    dt_polys = data.get("dt_polys", [])
    rec_texts = data.get("rec_texts", [])
    rec_scores = data.get("rec_scores", [])

    output = []

    for i, poly in enumerate(dt_polys):

        if i >= len(rec_texts):
            continue

        text = str(rec_texts[i]).strip()

        if not text:
            continue

        points = [[float(p[0]), float(p[1])] for p in poly]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        score = None

        if i < len(rec_scores):
            score = float(rec_scores[i])

        output.append(
            {
                "id": f"t{i:03d}",
                "type": "text",
                "text": text,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
                "confidence": score,
            }
        )

    return output

def normalize_ocr_elements(elements: list) -> list:
    """
    Orchestrates the splitting of compound elements and reconstructs
    the geometry so the rest of the pipeline functions normally.
    """
    compound_candidates = []

    # 1. Filter elements that might be compound to save LLM tokens
    # (e.g., look for multiple colons or long strings)
    for el in elements:
        text = el.get("text", "")
        if text.count(":") > 1 or (":" in text and len(text.split()) > 4):
            compound_candidates.append(text)

    # If nothing looks compound, skip the LLM call entirely
    if not compound_candidates:
        return elements

    # 2. Get the semantic splits from the fast LLM
    try:
        split_map = run_pass_1_llm_splitter(compound_candidates)
    except Exception as e:
        print(f"Pass 1 Splitter failed: {e}. Falling back to raw OCR.")
        return elements

    normalized_elements = []

    # 3. Rebuild the elements array with new visual coordinates
    for el in elements:
        text = el.get("text", "")

        # If the LLM successfully split this specific text into multiple parts
        if (
            text in split_map
            and isinstance(split_map[text], list)
            and len(split_map[text]) > 1
        ):

            sub_texts = split_map[text]
            for i, sub_text in enumerate(sub_texts):
                new_bbox = _interpolate_sub_bbox(text, sub_text, el["bbox"])
                new_x1, y1, new_x2, y2 = new_bbox

                # Create synthetic elements that look identical to PaddleOCR output
                normalized_elements.append(
                    {
                        "id": f"{el['id']}_s{i}",  # e.g., t005_s0, t005_s1
                        "type": "text",
                        "text": sub_text.strip(),
                        "bbox": new_bbox,
                        "center": [
                            round((new_x1 + new_x2) / 2, 2),
                            round((y1 + y2) / 2, 2),
                        ],
                        "width": round(new_x2 - new_x1, 2),
                        "height": round(y2 - y1, 2),
                        "confidence": el.get("confidence"),
                    }
                )
        else:
            # Leave non-compound elements exactly as they are
            normalized_elements.append(el)

    return normalized_elements

def group_into_rows(items, y_tolerance=12):
    items = sorted(items, key=lambda x: x["center"][1])

    rows = []

    for item in items:

        cy = item["center"][1]

        matched_row = None

        for row in rows:
            if abs(cy - row["center_y"]) <= y_tolerance:
                matched_row = row
                break

        if matched_row:
            matched_row["items"].append(item)
            matched_row["center_y"] = np.mean(
                [x["center"][1] for x in matched_row["items"]]
            )
        else:
            rows.append({"center_y": cy, "items": [item]})

    for row in rows:
        row["items"].sort(key=lambda x: x["center"][0])

    rows.sort(key=lambda x: x["center_y"])

    return rows