import json
import os
from fastapi import HTTPException

from util.llm_utils import (
  _format_regions_for_prompt,
  _format_candidates_for_prompt,
  _format_checkboxes_for_prompt,
  _format_elements_for_prompt
)

from service.gemini_service import (
    gemini_client
)

def build_structure_prompt(doc_rep, candidates):
    elements_repr = _format_elements_for_prompt(doc_rep)
    regions_repr = _format_regions_for_prompt(doc_rep)
    checkboxes_repr = _format_checkboxes_for_prompt(doc_rep)
    candidates_repr = _format_candidates_for_prompt(candidates)

    return f"""
You are a document understanding system. You are given OCR text, bounding
boxes, detected regions, checkboxes and candidate relationships from a
physical form. The form may belong to ANY domain.

YOUR ROLE
=========
Decide WHAT things mean and HOW they relate. Computer vision has already
decided WHERE things are. You must NEVER invent or reproduce coordinates.
You must ONLY reference the IDs that are provided to you.

INPUT DATA
==========

Document size: {doc_rep["image_width"]} x {doc_rep["image_height"]} pixels.
Coordinates below are normalized to [0,1]. (0,0) is the top-left, (1,1) is
the bottom-right.

OCR ELEMENTS (id "text" bbox=(x1,y1,x2,y2) conf=confidence)
-----------------------------------------------------------
{elements_repr}

REGIONS (bordered boxes detected by computer vision)
---------------------------------------------------
{regions_repr}

CHECKBOXES (cv-determined state)
-------------------------------
{checkboxes_repr}

CANDIDATE LABEL->VALUE RELATIONSHIPS (geometric scoring)
-------------------------------------------------------
{candidates_repr}

TASK
====
1. SECTIONS
   Group elements and regions into logical sections. Each section must have
   a meaningful name. Assign every meaningful element to exactly one section
   by listing its ID in "element_ids". A section may use "region_ids" for
   bordered regions it covers (optional).

   Use "region_ids" whenever a detected region clearly belongs to a section
   (e.g. a bordered box, a table box, a field group). This gives the section
   an exact bounding box. Do not force a region into a section if it does
   not belong there.

2. RELATIONSHIPS
   Decide relationships between elements using text content, spatial layout
   and the candidate list. Supported types:

   - "label_value":
       label_id: the label element (single id).
       value_ids: one or more elements that form the value.
     A value is usually immediately to the right of its label on the same
     row, or directly below it. Use the geometric candidates as a guide but
     do not blindly trust them.
     If the label element ALREADY contains the answer (e.g. "AGE: 21"),
     still create a label_value relationship, but you may leave value_ids
     empty -- the system will split the text.
     Do NOT confuse instructions, headings, or unrelated text with values.
     IMPORTANT: value_ids must contain ONLY the text that is the actual
     value. Never include neighboring option labels, instructions, or other
     fields' content.

   - "question":
       question_id: the question text element (single id).
       answer_id: the element containing the chosen answer/option (or null
                  if no answer is visible).
       option_ids: the elements that are the possible options for this
                   question (e.g. "YES ...", "NO ...").
     Use this when a label is a QUESTION whose possible answers are nearby
     options (often paired with checkboxes). Do NOT use "label_value" for
     such questions, and never stuff the options into a label_value value.

   - "checkbox_option":
       checkbox_id: a checkbox element.
       label_id: the text element that is the label/option for this checkbox
                 (usually immediately to the right of the checkbox).
       checked: the selected state (true/false). If the cv state is
                "uncertain", decide from the ink or leave as the cv value.
     Never attach instructional text to a checkbox label.

   - "table":
       region_id: a region that is a table.
       has_header: true if the first row is a header row.

3. UNASSIGNED TEXT
   If some element is truly irrelevant (watermark, page number, logo text),
   list its ID in "unassigned_text_ids". Prefer assigning everything.

RULES
=====
- Only use IDs that exist in the input. Never invent IDs.
- Never output coordinates, bounding boxes, or pixel values.
- Never invent text that was not observed.
- A heading is a section title, not a field label, unless the structure
  indicates otherwise.
- Do not treat every line or region boundary as a semantic section.
- Preserve table structure: elements inside a table region stay in the table.
- Preserve handwriting as observed; do not silently correct it.
- If a field is present but empty, still create the label_value relationship
  with empty value_ids.

OUTPUT FORMAT (JSON only)
=========================
{{
  "document_type": "short description or null",
  "sections": [
    {{
      "name": "Section name",
      "region_ids": ["r000"],
      "element_ids": ["t000", "t001"]
    }}
  ],
  "relationships": [
    {{
      "type": "label_value",
      "label_id": "t000",
      "value_ids": ["t001"]
    }},
    {{
      "type": "checkbox_option",
      "checkbox_id": "c000",
      "label_id": "t002",
      "checked": true
    }},
    {{
      "type": "question",
      "question_id": "t004",
      "answer_id": "t005",
      "option_ids": ["t005", "t006"]
    }},
    {{
      "type": "table",
      "region_id": "r001",
      "has_header": true
    }}
  ],
  "unassigned_text_ids": []
}}
"""

def run_pass_1_llm_splitter(compound_texts: list) -> dict:
    """
    Sends compound OCR strings to Gemini to split them
    into individual logical fields.
    """

    if not compound_texts:
        return {}

    input_json_str = json.dumps(compound_texts, indent=2)

    prompt = f"""
        I am providing a list of strings detected by an OCR engine.

        Some strings contain multiple form fields merged together
        (e.g., "City: NY State: NY").

        Split these compound strings into individual logical fields.

        Rules:
        1. If a string contains multiple logical fields, split it into separate strings.
        2. If a string represents only one field, return it as a single item.
        3. Do not alter spelling.
        4. Do not alter casing.
        5. Do not invent or add text.
        6. Preserve the original text as much as possible.
        7. Return ONLY a valid JSON object.
        8. The key must be the exact original string.
        9. The value must be an array of split strings.

        Input Strings:

        {input_json_str}
        """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "system_instruction": (
                    "You are an OCR data cleaning utility. "
                    "You output strict JSON objects without "
                    "markdown formatting or conversational text."
                ),
            },
        )

        llm_output = response.text

        print("Gemini response received.")

        if not llm_output:
            raise HTTPException(
                status_code=500, detail="LLM returned an empty response."
            )

        try:
            return json.loads(llm_output)

        except json.JSONDecodeError as e:
            print("Invalid JSON returned by LLM:")
            print(llm_output)

            raise HTTPException(
                status_code=500, detail="LLM returned invalid JSON: " + str(e)
            )

    except Exception as e:
        print(f"API Error in Pass 1: {e}")
        return {}