import os
import json
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

from utils import (
    analyze_document,
    build_structure_prompt,
    resolve_structure,
    validate_extraction,
)


load_dotenv()


app = FastAPI(
    title="Document Extraction API",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

groq_client = Groq(
    api_key=groq_api_key
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Document Extraction API"
    }


# ---------------------------------------------------------
# LLM semantic interpretation (Pass 1 - structure)
#
# The LLM returns element/region/checkbox IDs only. Geometry is always
# resolved deterministically from the OCR/CV representation.
# ---------------------------------------------------------

def interpret_document(doc_rep, candidates):
    prompt = build_structure_prompt(doc_rep, candidates)
    print(prompt)
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document understanding system. "
                    "You interpret relationships and return JSON "
                    "referencing only the IDs you are given. "
                    "Never invent coordinates or IDs."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        response_format={
            "type": "json_object"
        }
    )

    llm_output = (
        completion
        .choices[0]
        .message
        .content
    )

    print(
        f"Tokens: "
        f"{completion.usage.prompt_tokens} input + "
        f"{completion.usage.completion_tokens} output = "
        f"{completion.usage.total_tokens} total"
    )

    if not llm_output:
        raise HTTPException(
            status_code=500,
            detail="LLM returned an empty response."
        )

    try:
        return json.loads(llm_output)
    except json.JSONDecodeError as e:
        print("Invalid JSON returned by LLM:")
        print(llm_output)
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON: " + str(e)
        )


# ---------------------------------------------------------
# Document extraction
#
# Pipeline:
#   1. Preprocess (orientation + deskew)
#   2. Layout analysis + OCR          -> document representation
#   3. Candidate field builder        -> geometric label/value candidates
#   4. LLM semantic interpretation    -> IDs only (sections, relationships)
#   5. Deterministic resolver         -> real bboxes from OCR/CV elements
#   6. Validation                     -> confidence + review reasons
# ---------------------------------------------------------

@app.post("/extract-document")
async def extract_document(
    image: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate file
    # -----------------------------------------------------

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type."
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported."
        )

    # -----------------------------------------------------
    # Create temporary file
    # -----------------------------------------------------

    suffix = os.path.splitext(
        image.filename or ""
    )[1]

    if not suffix:
        suffix = ".jpg"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    image_path = temp_file.name

    try:

        # -------------------------------------------------
        # Save uploaded image
        # -------------------------------------------------

        with temp_file:
            shutil.copyfileobj(
                image.file,
                temp_file
            )

        # -------------------------------------------------
        # 1. Preprocess + build document representation
        # -------------------------------------------------

        doc_rep, candidates = analyze_document(
            image_path,
            image_path=image_path,
        )

        print(
            f"OCR items: {len(doc_rep['elements'])} | "
            f"regions: {len(doc_rep['regions'])} | "
            f"checkboxes: {len(doc_rep['checkboxes'])}"
        )

        print(
            f"Candidate relationships: {len(candidates)}"
        )

        # -------------------------------------------------
        # 4. LLM semantic interpretation (IDs only)
        # -------------------------------------------------

        print("Sending document to Groq...")

        llm_data = interpret_document(doc_rep, candidates)

        # -------------------------------------------------
        # 5. Deterministic resolution (IDs -> real geometry)
        # -------------------------------------------------

        result, review_reasons = resolve_structure(
            llm_data,
            doc_rep,
            candidates,
        )

        # -------------------------------------------------
        # 6. Validation
        # -------------------------------------------------

        result = validate_extraction(
            result,
            doc_rep,
            review_reasons,
        )

        # -------------------------------------------------
        # Return final result
        # -------------------------------------------------

        return {
            "success": True,

            "filename": image.filename,

            "image": {
                "width": doc_rep["image_width"],
                "height": doc_rep["image_height"]
            },

            "preprocessing": doc_rep["preprocessing"],

            "ocr_items": len(doc_rep["elements"]),

            "detected_horizontal_lines": len(
                doc_rep["lines"]["horizontal"]
            ),

            "detected_regions": len(doc_rep["regions"]),

            "detected_checkboxes": len(doc_rep["checkboxes"]),

            "data": result,

            "document_representation": {
                "elements": doc_rep["elements"],
                "regions": doc_rep["regions"],
                "checkboxes": doc_rep["checkboxes"],
                "lines": doc_rep["lines"],
                "candidate_relationships": candidates,
            },

            "llm_output": llm_data,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Extraction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # -------------------------------------------------
        # Delete temporary files
        # -------------------------------------------------

        if os.path.exists(image_path):
            os.remove(image_path)
