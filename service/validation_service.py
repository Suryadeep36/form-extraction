def validate_extraction(result, doc_rep, review_reasons):
    review_reasons = list(review_reasons)

    confidences = []

    for section in result["sections"]:
        for field in section["fields"]:
            if field["confidence"] is not None:
                confidences.append(field["confidence"])
            if field["uncertain"]:
                review_reasons.append(f"Field {field['label']!r} has no reliable value")

        for checkbox in section["checkboxes"]:
            if checkbox["confidence"] is not None:
                confidences.append(checkbox["confidence"])

    for element in doc_rep["elements"]:
        if element["confidence"] is not None and element["confidence"] < 0.5:
            review_reasons.append(f"Low OCR confidence for {element['text']!r}")

    if result["unassigned_text"]:
        review_reasons.append(
            f"{len(result['unassigned_text'])} text element(s) unassigned"
        )

    if confidences:
        document_confidence = round(sum(confidences) / len(confidences), 4)
    else:
        document_confidence = None

    result["document_confidence"] = document_confidence
    result["requires_human_review"] = bool(review_reasons)
    result["review_reasons"] = review_reasons

    return result