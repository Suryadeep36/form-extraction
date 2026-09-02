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


def _format_fields_for_prompt(doc_rep):
    """
    Geometry-derived fields (input regions + geometric label/value binding).
    These are strong starting points; the LLM confirms or corrects them via
    label_value relationships referencing the same element ids.
    """
    lines = []

    for field in doc_rep["fields"]:
        region = field.get("input_region") or {}
        bbox = field.get("value_bbox") or field.get("label_bbox")
        bbox_text = ""
        if bbox:
            b = _norm(bbox, doc_rep["image_width"], doc_rep["image_height"])
            bbox_text = f" bbox=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f})"
        lines.append(
            f'{field["id"]} kind={region.get("kind") or "?"} '
            f'label="{field["label"] or ""}" '
            f'value="{field["value"] or ""}" '
            f"label_ids={field.get('label_element_ids')} "
            f"value_ids={field.get('value_element_ids')}"
            f"{bbox_text}"
        )

    return "\n".join(lines)


def _format_tables_for_prompt(doc_rep):
    """
    Geometry/model-derived tables with OCR already assigned to cells.
    The LLM confirms them with a `table` relationship on the covering region.
    """
    lines = []

    for table in doc_rep["tables"]:
        bbox = _norm(table["bbox"], doc_rep["image_width"], doc_rep["image_height"])
        lines.append(
            f'[{table["id"]}] bbox=({bbox[0]:.3f},{bbox[1]:.3f},'
            f"{bbox[2]:.3f},{bbox[3]:.3f}) "
            f"cells={table['n_cells']} structure={table['structure_source']}"
        )
        for cell in table.get("cells", []):
            text = (cell.get("text") or "").strip()
            if not text:
                continue
            cb = cell["bbox"]
            lines.append(
                f"    row={cell.get('row')} col={cell.get('column')} "
                f"(span {cell.get('row_span', 1)}x{cell.get('column_span', 1)}) "
                f'"{text}"'
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
        assoc = checkbox.get("associated_text") or ""
        eid = checkbox.get("element_id") or ""
        lines.append(
            f'{checkbox["id"]} state="{checkbox["state"]}" '
            f'mark="{checkbox.get("mark_type") or "?"}" '
            f'conf={checkbox["confidence"]} '
            f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) "
            f'element={eid} label="{assoc}"'
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