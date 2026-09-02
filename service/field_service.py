"""
Form-field orchestration layer.

Runs the generic geometric field pipeline:
    1. detect input regions (lines / boxes / blank gaps)
    2. spatial compound splitting of OCR elements
    3. label/value association
    4. build first-class `fields`

Table regions are passed in so field detection never mistakes table borders
for input lines.
"""

from util.form_utils import (
    detect_input_regions,
    split_compound_elements,
    build_form_fields,
)


def process_form_fields(image, elements, table_bboxes=None, checkboxes=None):
    """
    Detect form fields for the document representation.

    Args:
        image: BGR ndarray in processed coordinate space.
        elements: raw OCR elements (with word boxes when available).
        table_bboxes: bboxes of detected tables (fields must avoid them).
        checkboxes: detected checkboxes (small boxes to avoid re-labeling).

    Returns:
        (fields, input_regions, elements)
        `elements` is the *updated* element list after compound splitting.
    """
    table_bboxes = table_bboxes or []
    checkboxes = checkboxes or []

    print("[FIELDS] Detecting input regions...")
    input_regions = detect_input_regions(
        image,
        elements=elements,
        table_bboxes=table_bboxes,
        checkboxes=checkboxes,
    )
    print(f"[FIELDS] Input regions detected: {len(input_regions)}")

    if input_regions:
        print("[FIELDS] Splitting compound OCR elements (spatial)...")
        elements = split_compound_elements(elements, input_regions)
    else:
        print(
            "[FIELDS] No input regions; compound OCR normalization deferred "
            "to the LLM-free fallback in the OCR service."
        )

    print("[FIELDS] Associating labels/values...")
    fields = build_form_fields(elements, input_regions) if input_regions else []
    print(f"[FIELDS] Fields: {len(fields)}")

    return fields, input_regions, elements