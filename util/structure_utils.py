import re

from util.geometry_utils import (
    _point_in_bbox,
    _union_bbox,
    _norm
)
from service.ocr_service import (
    group_into_rows
)

def _infer_value_type(text):
    stripped = text.strip()

    if re.fullmatch(r"\d{1,4}[-/]\d{1,2}[-/]\d{2,4}", stripped):
        return "date"

    if re.fullmatch(r"\d{1,2}:\d{2}(\s?(AM|PM))?", stripped, re.IGNORECASE):
        return "time"

    if re.fullmatch(r"[+-]?\d[\d,]*([.,]\d+)?%?", stripped):
        return "number"

    if re.fullmatch(
        r"[+-]?\d[\d,]*([.,]\d+)?\s*(kg|cm|m|km|mi|mph|ml|l|hr|min|g|mg|USD|\$|°C|F)",
        stripped,
        re.IGNORECASE,
    ):
        return "measurement"

    return "text"


def _avg_confidence(confidences):
    values = [c for c in confidences if c is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _element_map(doc_rep):
    return {e["id"]: e for e in doc_rep["elements"]}


def _region_map(doc_rep):
    return {r["id"]: r for r in doc_rep["regions"]}


def _checkbox_map(doc_rep):
    return {c["id"]: c for c in doc_rep["checkboxes"]}


def _elements_in_bbox(doc_rep, bbox):
    return [e for e in doc_rep["elements"] if _point_in_bbox(e["center"], bbox)]


def _resolve_sections(llm_data, doc_rep):
    elements = _element_map(doc_rep)
    regions = _region_map(doc_rep)

    sections = []

    raw_sections = llm_data.get("sections") or []

    for section in raw_sections:

        name = str(section.get("name") or "").strip() or "Section"

        region_bboxes = []
        for rid in section.get("region_ids") or []:
            region = regions.get(rid)
            if region:
                region_bboxes.append(region["bbox"])

        section_element_ids = []
        for eid in section.get("element_ids") or []:
            if eid in elements and eid not in section_element_ids:
                section_element_ids.append(eid)

        # Elements whose center lies inside one of the section's regions
        # automatically belong to the section.
        for region in regions.values():
            if region["bbox"] in region_bboxes:
                for e in _elements_in_bbox(doc_rep, region["bbox"]):
                    if e["id"] not in section_element_ids:
                        section_element_ids.append(e["id"])

        sections.append(
            {
                "name": name,
                "region_ids": list(section.get("region_ids") or []),
                "element_ids": section_element_ids,
                "region_bboxes": region_bboxes,
            }
        )

    return sections


def _find_section_for_element(sections, element_id, elements):
    element = elements.get(element_id)
    if not element:
        return 0

    for index, section in enumerate(sections):
        for bbox in section["region_bboxes"]:
            if _point_in_bbox(element["center"], bbox):
                return index

    # Fall back to a section that explicitly lists the element.
    for index, section in enumerate(sections):
        if element_id in section["element_ids"]:
            return index

    return -1


def _build_table(doc_rep, region_id, has_header):
    region = _region_map(doc_rep).get(region_id)
    if not region:
        return None

    items = _elements_in_bbox(doc_rep, region["bbox"])
    if not items:
        return None

    rows = group_into_rows(items, y_tolerance=12)

    # Cluster columns by x-center gaps.
    row_cells = []

    for row in rows:
        cells = []
        current_cell = [row["items"][0]]

        for item in row["items"][1:]:
            prev = current_cell[-1]
            gap = item["center"][0] - (prev["bbox"][2] + prev["bbox"][0]) / 2
            if gap > 0 and gap > prev["width"] * 0.5:
                cells.append(current_cell)
                current_cell = [item]
            else:
                current_cell.append(item)

        cells.append(current_cell)
        row_cells.append(cells)

    width = doc_rep["image_width"]
    height = doc_rep["image_height"]

    normalized_cells = []

    for cells in row_cells:
        normalized_row = []
        for cell in cells:
            bbox = _union_bbox([e["bbox"] for e in cell])
            normalized_row.append(
                {
                    "value": " ".join(e["text"] for e in cell).strip(),
                    "element_ids": [e["id"] for e in cell],
                    "bbox": _norm(bbox, width, height),
                    "bbox_pixels": bbox,
                    "confidence": _avg_confidence([e["confidence"] for e in cell]),
                }
            )
        normalized_cells.append(normalized_row)

    if not normalized_cells:
        return None

    if has_header and len(normalized_cells) > 1:
        headers = [cell["value"] for cell in normalized_cells[0]]
        rows_data = normalized_cells[1:]
    else:
        headers = []
        rows_data = normalized_cells

    return {
        "title": None,
        "headers": headers,
        "rows": rows_data,
        "region_id": region_id,
        "bbox": _norm(region["bbox"], width, height),
        "bbox_pixels": region["bbox"],
        "confidence": None,
        "uncertain": False,
        "reason": None,
    }


def _split_inline_label_value(label_element):
    text = label_element["text"]
    match = re.split(r"\s*[:\uFF1A]\s*", text, maxsplit=1)
    if len(match) == 2 and match[1].strip():
        return match[0].strip(), match[1].strip()
    return text, None