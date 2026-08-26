import numpy as np

from util.structure_utils import (
    _element_map,
    _checkbox_map,
    _region_map,
    _resolve_sections,
    _find_section_for_element,
    _split_inline_label_value,
    _avg_confidence,
    _infer_value_type,
    _build_table
)

from util.geometry_utils import (
    _union_bbox,
    _norm,
    _point_in_bbox
)

def resolve_structure(llm_data, doc_rep, candidates):
    elements = _element_map(doc_rep)
    checkboxes = _checkbox_map(doc_rep)
    regions = _region_map(doc_rep)
    width = doc_rep["image_width"]
    height = doc_rep["image_height"]

    sections = _resolve_sections(llm_data, doc_rep)

    # Guard: if the LLM produced no sections, create a single catch-all.
    if not sections:
        sections = [
            {
                "name": "Document",
                "region_ids": [],
                "element_ids": [e["id"] for e in doc_rep["elements"]],
                "region_bboxes": [],
            }
        ]

    # Section metadata (bbox, consumed elements).
    section_outputs = []
    section_consumed = [[] for _ in sections]

    for index, section in enumerate(sections):
        bboxes = list(section["region_bboxes"])
        for eid in section["element_ids"]:
            element = elements.get(eid)
            if element:
                bboxes.append(element["bbox"])

        if bboxes:
            bbox_pixels = _union_bbox(bboxes)
        else:
            bbox_pixels = None

        section_outputs.append(
            {
                "name": section["name"],
                "region_ids": section["region_ids"],
                "element_ids": list(section["element_ids"]),
                "bbox_pixels": bbox_pixels,
                "bbox": _norm(bbox_pixels, width, height) if bbox_pixels else None,
                "fields": [],
                "questions": [],
                "checkboxes": [],
                "tables": [],
                "other_elements": [],
                "uncertain": False,
                "reason": None,
            }
        )

    def section_for_element(eid, create=True):
        index = _find_section_for_element(sections, eid, elements)
        if index >= 0:
            return index
        return None

    relationships = llm_data.get("relationships") or []
    review_reasons = []

    # ----- label_value relationships -> fields -----
    for relation in relationships:

        rtype = relation.get("type")

        if rtype == "label_value":
            label_id = relation.get("label_id")
            value_ids = relation.get("value_ids") or []

            if label_id not in elements:
                review_reasons.append(f"LLM referenced unknown label id {label_id!r}")
                continue

            label_element = elements[label_id]
            valid_value_ids = [vid for vid in value_ids if vid in elements]
            invalid_value_ids = [vid for vid in value_ids if vid not in elements]
            if invalid_value_ids:
                review_reasons.append(
                    f"LLM referenced unknown value ids {invalid_value_ids}"
                )

            value_elements = [elements[vid] for vid in valid_value_ids]
            value_elements.sort(key=lambda e: (e["center"][1], e["center"][0]))

            label_text = label_element["text"]

            if value_elements:
                value_text = " ".join(e["text"] for e in value_elements).strip()
                value_bbox = _union_bbox([e["bbox"] for e in value_elements])
            else:
                value_text = None
                value_bbox = label_element["bbox"]

            inline_value = None
            if not value_elements:
                label_text, inline_value = _split_inline_label_value(label_element)
                if inline_value is not None:
                    value_text = inline_value
                    value_bbox = label_element["bbox"]
                    valid_value_ids = [label_element["id"]]

            confidence = _avg_confidence(
                [label_element["confidence"]]
                + [e["confidence"] for e in value_elements]
            )

            index = section_for_element(label_id)
            field = {
                "label": label_text,
                "value": value_text,
                "value_type": _infer_value_type(value_text) if value_text else None,
                "confidence": confidence,
                "uncertain": value_text is None,
                "reason": "value not found" if value_text is None else None,
                "label_element_ids": [label_element["id"]],
                "value_element_ids": valid_value_ids,
                "label_bbox": _norm(label_element["bbox"], width, height),
                "label_bbox_pixels": label_element["bbox"],
                "value_bbox": _norm(value_bbox, width, height),
                "value_bbox_pixels": value_bbox,
            }

            if index is None:
                section_outputs.append(
                    {
                        "name": "Unassigned",
                        "region_ids": [],
                        "element_ids": [],
                        "bbox_pixels": (
                            _union_bbox(
                                [label_element["bbox"]]
                                + [e["bbox"] for e in value_elements]
                            )
                            if value_elements
                            else label_element["bbox"]
                        ),
                        "bbox": None,
                        "fields": [],
                        "questions": [],
                        "checkboxes": [],
                        "tables": [],
                        "other_elements": [],
                        "uncertain": True,
                        "reason": "no section assigned",
                    }
                )
                index = len(section_outputs) - 1
                section_consumed.append([])

            section_outputs[index]["fields"].append(field)
            section_consumed[index].extend([label_element["id"]] + valid_value_ids)

        elif rtype == "question":
            question_id = relation.get("question_id")
            answer_id = relation.get("answer_id")
            option_ids = relation.get("option_ids") or []

            if question_id not in elements:
                review_reasons.append(
                    f"LLM referenced unknown question id {question_id!r}"
                )
                continue

            question_element = elements[question_id]
            answer_element = elements.get(answer_id)
            option_elements = [elements[oid] for oid in option_ids if oid in elements]

            options = []
            for option in option_elements:
                options.append(
                    {
                        "label": option["text"],
                        "selected": option["id"] == answer_id,
                        "confidence": option["confidence"],
                        "element_id": option["id"],
                        "bbox": _norm(option["bbox"], width, height),
                        "bbox_pixels": option["bbox"],
                    }
                )

            index = section_for_element(question_id)

            entry = {
                "question": question_element["text"],
                "answer": (answer_element["text"] if answer_element else None),
                "options": options,
                "confidence": _avg_confidence(
                    [question_element["confidence"]]
                    + [e["confidence"] for e in option_elements]
                ),
                "uncertain": answer_element is None,
                "reason": (None if answer_element else "no visible selected option"),
                "question_element_ids": [question_element["id"]],
                "answer_element_ids": (
                    [answer_element["id"]] if answer_element else []
                ),
            }

            if index is None:
                index = 0

            section_outputs[index]["questions"].append(entry)
            section_consumed[index].append(question_element["id"])
            section_consumed[index].extend([o["id"] for o in option_elements])
            if answer_element:
                section_consumed[index].append(answer_element["id"])

        elif rtype == "checkbox_option":
            checkbox_id = relation.get("checkbox_id")
            label_id = relation.get("label_id")

            if checkbox_id not in checkboxes:
                review_reasons.append(
                    f"LLM referenced unknown checkbox id {checkbox_id!r}"
                )
                continue

            checkbox = checkboxes[checkbox_id]

            if label_id and label_id in elements:
                label_element = elements[label_id]
                label_text = label_element["text"]
                label_index = section_for_element(label_id)
            else:
                label_element = None
                label_text = None
                label_index = None

            if label_index is None:
                # Associate checkbox to a section by its own position.
                label_index = None
                for index, section in enumerate(sections):
                    for bbox in section["region_bboxes"]:
                        if _point_in_bbox(checkbox["center"], bbox):
                            label_index = index
                            break
                    if label_index is not None:
                        break

            checked = relation.get("checked")
            if checked is None:
                checked = checkbox["state"] == "checked"
            elif checkbox["state"] == "uncertain":
                pass  # trust the LLM decision when cv is uncertain

            if label_element:
                section_consumed[label_index].append(label_element["id"])

            entry = {
                "label": label_text,
                "checked": bool(checked),
                "instruction": None,
                "confidence": checkbox["confidence"],
                "uncertain": checkbox["state"] == "uncertain",
                "reason": None,
                "checkbox_id": checkbox_id,
                "label_element_ids": ([label_element["id"]] if label_element else []),
                "bbox": _norm(checkbox["bbox"], width, height),
                "bbox_pixels": checkbox["bbox"],
            }

            if label_index is not None and label_index < len(section_outputs):
                section_outputs[label_index]["checkboxes"].append(entry)
            else:
                section_outputs[0]["checkboxes"].append(entry)

        elif rtype == "table":
            table = _build_table(
                doc_rep,
                relation.get("region_id"),
                bool(relation.get("has_header", True)),
            )

            if table is None:
                review_reasons.append(
                    f"Could not build table for region "
                    f"{relation.get('region_id')!r}"
                )
                continue

            region_id = relation.get("region_id")
            index = None

            for section_index, section in enumerate(sections):
                if region_id in section["region_ids"]:
                    index = section_index
                    break

            if index is None:
                for section_index, section in enumerate(sections):
                    region = regions.get(region_id)
                    if region and any(
                        _point_in_bbox(
                            (
                                region["center"]
                                if "center" in region
                                else [
                                    (region["bbox"][0] + region["bbox"][2]) / 2,
                                    (region["bbox"][1] + region["bbox"][3]) / 2,
                                ]
                            ),
                            bbox,
                        )
                        for bbox in section["region_bboxes"]
                    ):
                        index = section_index
                        break

            if index is None:
                index = 0

            for row in table["rows"]:
                for cell in row:
                    section_consumed[index].extend(cell["element_ids"])

            section_outputs[index]["tables"].append(table)

    # ----- other elements referenced by sections but not in fields -----
    for index, section in enumerate(sections):
        other = []
        for eid in section["element_ids"]:
            element = elements.get(eid)
            if element and eid not in section_consumed[index]:
                other.append(
                    {
                        "text": element["text"],
                        "element_id": eid,
                        "bbox": _norm(element["bbox"], width, height),
                        "bbox_pixels": element["bbox"],
                    }
                )
        section_outputs[index]["other_elements"] = other

    # ----- unassigned text -----
    referenced = set()
    for index in range(len(section_outputs)):
        referenced.update(section_consumed[index])

    unassigned = []
    for element in doc_rep["elements"]:
        if element["id"] in referenced:
            continue
        if any(element["id"] in section["element_ids"] for section in sections):
            continue
        unassigned.append(element)

    llm_unassigned = llm_data.get("unassigned_text_ids") or []
    for eid in llm_unassigned:
        element = elements.get(eid)
        if element and element["id"] not in referenced:
            if element not in unassigned:
                unassigned.append(element)

    unassigned_list = [
        {
            "text": e["text"],
            "element_id": e["id"],
            "bbox": _norm(e["bbox"], width, height),
            "bbox_pixels": e["bbox"],
            "confidence": e["confidence"],
        }
        for e in unassigned
    ]

    result = {
        "document_type": llm_data.get("document_type"),
        "sections": section_outputs,
        "unassigned_text": unassigned_list,
    }

    return result, review_reasons