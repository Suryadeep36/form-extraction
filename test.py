from ollama import chat
prompt = """
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

Document size: 644 x 845 pixels.
Coordinates below are normalized to [0,1]. (0,0) is the top-left, (1,1) is
the bottom-right.

OCR ELEMENTS (id "text" bbox=(x1,y1,x2,y2) conf=confidence)
-----------------------------------------------------------
t000 "HANDWRITING SAMPLE FORM" bbox=(0.2873,0.0154,0.7003,0.0391) conf=0.981
t001 "NAME" bbox=(0.0745,0.0698,0.1382,0.0817) conf=1.000
t002 "DATE" bbox=(0.3618,0.0698,0.4224,0.0876) conf=0.998
t003 "CITY" bbox=(0.528,0.0686,0.5854,0.0899) conf=1.000
t004 "STATE ZIP" bbox=(0.7345,0.0722,0.854,0.0888) conf=0.995
t005 "8-3-89" bbox=(0.3416,0.0935,0.4627,0.116) conf=1.000
t006 "MinDen Ciry Mi 48456" bbox=(0.5155,0.0947,0.9193,0.1207) conf=0.727
t007 "This sample of handwriting is being collected for use in testing computer recognition of hand printed numbers" bbox=(0.0761,0.1219,0.9752,0.1456) conf=0.995
t008 "and letters. Please print the following characters in the boxes that appear below." bbox=(0.0497,0.1396,0.7003,0.1586) conf=1.000
t009 "0123456789" bbox=(0.0807,0.1562,0.2345,0.1716) conf=1.000
t010 "0123456789" bbox=(0.3727,0.1574,0.5248,0.1728) conf=1.000
t011 "0123456789" bbox=(0.6646,0.1574,0.8152,0.174) conf=1.000
t012 "0123456789" bbox=(0.0543,0.1811,0.2717,0.2024) conf=0.999
t013 "0123456>89" bbox=(0.3525,0.1751,0.5807,0.2036) conf=0.968
t014 "0123456789" bbox=(0.6553,0.1811,0.8758,0.2024) conf=0.993
t015 "87" bbox=(0.0714,0.2083,0.1025,0.2272) conf=1.000
t016 "701" bbox=(0.177,0.2083,0.2143,0.2272) conf=1.000
t017 "3752" bbox=(0.3075,0.2083,0.3556,0.2272) conf=1.000
t018 "80759" bbox=(0.4674,0.2107,0.5202,0.2284) conf=1.000
t019 "960941" bbox=(0.646,0.2107,0.7081,0.2284) conf=1.000
t020 "87" bbox=(0.0497,0.2272,0.1071,0.2568) conf=0.999
t021 "701" bbox=(0.163,0.2284,0.222,0.2556) conf=1.000
t022 "3752" bbox=(0.2873,0.2272,0.3866,0.2568) conf=1.000
t023 "80759" bbox=(0.4503,0.232,0.5637,0.258) conf=0.992
t024 "960941" bbox=(0.6335,0.232,0.7671,0.2568) conf=1.000
t025 "158" bbox=(0.0792,0.2604,0.1165,0.2817) conf=1.000
t026 "4586" bbox=(0.2112,0.2615,0.2578,0.2805) conf=0.999
t027 "32123" bbox=(0.3665,0.2604,0.427,0.2828) conf=1.000
t028 "832656" bbox=(0.5481,0.2627,0.6102,0.2817) conf=1.000
t029 "82" bbox=(0.8509,0.2639,0.8835,0.284) conf=1.000
t030 "158" bbox=(0.0543,0.2805,0.1304,0.3077) conf=1.000
t031 "4584" bbox=(0.1879,0.2817,0.2717,0.3077) conf=0.989
t032 "33123" bbox=(0.3463,0.2828,0.455,0.3077) conf=0.901
t033 "832656" bbox=(0.5404,0.2828,0.6801,0.3077) conf=0.958
t034 "82" bbox=(0.8261,0.2828,0.8835,0.3112) conf=1.000
t035 "7481" bbox=(0.0792,0.3136,0.1242,0.3325) conf=1.000
t036 "80539" bbox=(0.2376,0.3148,0.2919,0.3325) conf=1.000
t037 "419219" bbox=(0.4193,0.316,0.4814,0.3337) conf=1.000
t038 "67" bbox=(0.7143,0.316,0.7453,0.3361) conf=1.000
t039 "904" bbox=(0.823,0.3172,0.8618,0.3373) conf=1.000
t040 "7481" bbox=(0.059,0.3337,0.1413,0.3609) conf=1.000
t041 "80539" bbox=(0.2174,0.3361,0.3276,0.3621) conf=1.000
t042 "419219" bbox=(0.4037,0.3373,0.5373,0.3633) conf=0.999
t043 "67" bbox=(0.6957,0.3361,0.7484,0.3669) conf=1.000
t044 "904" bbox=(0.8012,0.3361,0.8773,0.3645) conf=1.000
t045 "61738" bbox=(0.0792,0.368,0.1351,0.3858) conf=1.000
t046 "729658" bbox=(0.2609,0.3692,0.323,0.3858) conf=1.000
t047 "75" bbox=(0.5606,0.3692,0.5916,0.3893) conf=1.000
t048 "390" bbox=(0.6724,0.3692,0.7096,0.3893) conf=1.000
t049 "5716" bbox=(0.8028,0.3716,0.8478,0.3893) conf=1.000
t050 "61738" bbox=(0.0528,0.3882,0.1693,0.4142) conf=1.000
t051 "729658" bbox=(0.2407,0.3917,0.4006,0.4154) conf=0.980
t052 "75" bbox=(0.5419,0.387,0.5947,0.4178) conf=0.997
t053 "390" bbox=(0.6444,0.3905,0.7143,0.4178) conf=0.998
t054 "57/6" bbox=(0.7733,0.3941,0.8618,0.4189) conf=0.994
t055 "109334" bbox=(0.0807,0.4201,0.1429,0.4391) conf=1.000
t056 "40" bbox=(0.3835,0.4213,0.4146,0.4414) conf=1.000
t057 "625" bbox=(0.486,0.4213,0.5248,0.4426) conf=0.999
t058 "4234" bbox=(0.6165,0.4213,0.6646,0.4414) conf=1.000
t059 "46002" bbox=(0.7764,0.4249,0.8292,0.4426) conf=1.000
t060 "109334" bbox=(0.0559,0.4438,0.205,0.4651) conf=1.000
t061 "40" bbox=(0.3587,0.4402,0.4146,0.4675) conf=1.000
t062 "675" bbox=(0.4643,0.4438,0.5342,0.4698) conf=1.000
t063 "4234" bbox=(0.5978,0.445,0.6957,0.471) conf=0.997
t064 "46002" bbox=(0.7593,0.4462,0.8696,0.4675) conf=1.000
t065 "gyxlakpdsbtzirumwfqjenhocv" bbox=(0.0807,0.4757,0.4705,0.4923) conf=0.992
t066 "9yxakfdsbtzirumwfgjenhocv" bbox=(0.0637,0.4994,0.7655,0.5243) conf=0.831
t067 "ZXSBNGECMYWQTKFLUOHPIRVDJA" bbox=(0.0776,0.529,0.5761,0.5503) conf=1.000
t068 "ZXSBNGECMYWQTKFLUOHPIRVDJA" bbox=(0.0621,0.5562,0.7811,0.5799) conf=0.992
t069 "Please print the following text in the box below:" bbox=(0.0481,0.587,0.4379,0.6059) conf=1.000
t070 "We, the People of the United States, in order to form a more perfect Union, establish Justice, insure domestic" bbox=(0.0481,0.6036,0.972,0.6237) conf=0.999
t071 "Tranquility, provide for the common Defense, promote the general Welfare, and secure the Blessings of Liberty to" bbox=(0.0481,0.6201,0.9736,0.6402) conf=0.990
t072 "ourselves and our posterity, do ordain and establish this CONSTITUTION for the United States of America." bbox=(0.0481,0.6367,0.9224,0.658) conf=0.996
t073 "we, the Peopte of the Umiteg States, In orderto" bbox=(0.0481,0.6686,0.9255,0.7018) conf=0.941
t074 "forma more perfect Umion, establish Justice" bbox=(0.0481,0.6947,0.9301,0.7231) conf=0.957
t075 "insore domestic Tranguility, Provide for the" bbox=(0.0481,0.7207,0.9301,0.7491) conf=0.943
t076 "common Defense,promote thegeneraL Welfare" bbox=(0.045,0.7444,0.9534,0.7751) conf=0.966
t077 "ana secure the Blessings ofhiberty to our-" bbox=(0.0466,0.7692,0.9425,0.7964) conf=0.945
t078 "selves and oor posterity, do ordain ang" bbox=(0.0466,0.7929,0.8602,0.8225) conf=0.958
t079 "establish fhis CoNsTiTuTioN For the" bbox=(0.0512,0.8213,0.8447,0.8485) conf=0.894
t080 "United States of America." bbox=(0.0559,0.8544,0.7003,0.8781) conf=0.952

REGIONS (bordered boxes detected by computer vision)
---------------------------------------------------
r000 bbox=(0.0435,0.6698,0.9612,0.955) size=591x241

CHECKBOXES (cv-determined state)
-------------------------------
c000 state="checked" conf=0.98 bbox=(0.8245,0.2698,0.882,0.316)
c001 state="checked" conf=0.98 bbox=(0.0466,0.213,0.104,0.2592)
c002 state="checked" conf=0.933 bbox=(0.5404,0.3882,0.5932,0.4189)
c003 state="checked" conf=0.98 bbox=(0.4767,0.8544,0.5047,0.8793)
c004 state="checked" conf=0.98 bbox=(0.2205,0.8592,0.2484,0.8805)
c005 state="checked" conf=0.98 bbox=(0.2112,0.7467,0.2422,0.7657)
c006 state="checked" conf=0.98 bbox=(0.3509,0.7751,0.3789,0.7953)
c007 state="checked" conf=0.98 bbox=(0.1429,0.5609,0.1724,0.5799)
c008 state="checked" conf=0.98 bbox=(0.5419,0.2864,0.5683,0.3065)
c009 state="checked" conf=0.98 bbox=(0.5932,0.7988,0.6165,0.8189)
c010 state="checked" conf=0.98 bbox=(0.3835,0.5609,0.4068,0.5787)

CANDIDATE LABEL->VALUE RELATIONSHIPS (geometric scoring)
-------------------------------------------------------
t002 -> t005(1.703), t006(1.481)
t003 -> t006(1.857)
t004 -> t006(1.354)
t006 -> t005(2.842), t007(1.49)
t013 -> t012(3.0), t014(3.0), t017(1.7), t018(1.369), t022(1.227)
t065 -> t066(1.807)
t066 -> t067(1.868), t068(1.232)
t067 -> t068(1.838)
t068 -> t069(1.812), t070(1.421)
t069 -> t070(2.057), t071(1.659), t072(1.262)
t074 -> t075(3.057), t076(2.486), t077(1.892), t078(1.324)
t075 -> t076(3.11), t077(2.516), t078(1.948), t079(1.267)
t076 -> t078(2.573), t079(1.889)
t077 -> t078(3.085), t079(2.4), t080(1.605)
t078 -> t079(3.023), t080(2.226)
t079 -> t080(2.853)

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
{
  "document_type": "short description or null",
  "sections": [
    {
      "name": "Section name",
      "region_ids": ["r000"],
      "element_ids": ["t000", "t001"]
    }
  ],
  "relationships": [
    {
      "type": "label_value",
      "label_id": "t000",
      "value_ids": ["t001"]
    },
    {
      "type": "checkbox_option",
      "checkbox_id": "c000",
      "label_id": "t002",
      "checked": true
    },
    {
      "type": "question",
      "question_id": "t004",
      "answer_id": "t005",
      "option_ids": ["t005", "t006"]
    },
    {
      "type": "table",
      "region_id": "r001",
      "has_header": true
    }
  ],
  "unassigned_text_ids": []
}
"""
response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "/no_think\n\n" + prompt
        }
    ],
    think=False,
    options={
        "temperature": 0,
        "num_ctx": 8192
    }
)


print("Prompt tokens:", response["prompt_eval_count"])
print(
    "Prompt processing:",
    response["prompt_eval_duration"] / 1e9,
    "sec"
)

print("Output tokens:", response["eval_count"])
print(
    "Generation:",
    response["eval_duration"] / 1e9,
    "sec"
)

result = response["message"]["content"]

print(result)
