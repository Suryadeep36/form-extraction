from util.geometry_utils import (
    _norm
)

def _format_elements_for_prompt(doc_rep):
    lines = []

    for element in doc_rep["elements"]:
        bbox = _norm(element["bbox"], doc_rep["image_width"], doc_rep["image_height"])
        confidence = (
            "unknown"
            if element["confidence"] is None
            else f"{element['confidence']:.3f}"
        )
        lines.append(
            f'{element["id"]} "{element["text"]}" '
            f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) "
            f"conf={confidence}"
        )

    return "\n".join(lines)


def _format_regions_for_prompt(doc_rep):
    lines = []

    for region in doc_rep["regions"]:
        bbox = _norm(region["bbox"], doc_rep["image_width"], doc_rep["image_height"])
        lines.append(
            f'{region["id"]} bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) '
            f'size={region["width"]}x{region["height"]}'
        )

    return "\n".join(lines)


def _format_checkboxes_for_prompt(doc_rep):
    lines = []

    for checkbox in doc_rep["checkboxes"]:
        bbox = _norm(checkbox["bbox"], doc_rep["image_width"], doc_rep["image_height"])
        lines.append(
            f'{checkbox["id"]} state="{checkbox["state"]}" '
            f'conf={checkbox["confidence"]} '
            f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
        )

    return "\n".join(lines)


def _format_candidates_for_prompt(candidates):
    lines = []

    for candidate in candidates:
        pairs = ", ".join(
            f"{c['value_id']}({c['score']})" for c in candidate["candidates"]
        )
        lines.append(f'{candidate["label_id"]} -> {pairs}')

    return "\n".join(lines)