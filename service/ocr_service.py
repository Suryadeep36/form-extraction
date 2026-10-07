import numpy as np

from util.ocr_model_utils import (
    _get_ocr_engine,
    _get_orientation_model,
)



from util.geometry_utils import (
    _interpolate_sub_bbox
)

import util.config as config


def _quad_to_bbox(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return x1, y1, x2, y2


def get_ocr_data(image_or_path):
    ocr = _get_ocr_engine()
    result = ocr.predict(
        image_or_path,
        return_word_box=config.OCR_USE_WORD_BOXES,
    )

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
    word_texts = data.get("text_word") or []
    word_regions = data.get("text_word_region") or []

    output = []

    for i, poly in enumerate(dt_polys):

        if i >= len(rec_texts):
            continue

        text = str(rec_texts[i]).strip()

        if not text:
            continue

        x1, y1, x2, y2 = _quad_to_bbox(poly)

        score = None

        if i < len(rec_scores):
            score = float(rec_scores[i])

        element = {
            "id": f"t{i:03d}",
            "type": "text",
            "text": text,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),
            "confidence": score,
        }

        # Attach word-level boxes when the engine can provide them.
        if i < len(word_texts) and i < len(word_regions):
            words = []
            for wt, wr in zip(word_texts[i], word_regions[i]):
                max_len = len(wr) if hasattr(wr, "__len__") else 0
                if max_len >= 4:
                    wx1, wy1, wx2, wy2 = _quad_to_bbox(wr)
                else:
                    continue
                words.append(
                    {
                        "text": str(wt),
                        "bbox": [wx1, wy1, wx2, wy2],
                    }
                )
            if words:
                element["words"] = words

        output.append(element)

    return output

def normalize_ocr_elements(elements: list) -> list:
    """
    Returns elements as is since LLM usage has been removed.
    """
    return elements

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