import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from service.document_service import (
    analyze_document,
    interpret_document
)
from service.structure_service import (
    resolve_structure
)
from service.validation_service import (
    validate_extraction
)
from service.storage_service import (
    init_db,
    storage_enabled,
    save_document,
    list_documents,
    get_document,
    get_document_file,
    delete_document,
)
from service.template_service import (
    register_template,
    extract_filled,
    load_template,
    list_templates,
    get_template_image,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Document Extraction API",
    version="2.0.0",
    lifespan=lifespan,
)




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
# Health check
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Document Extraction API"
    }


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

        # Preserve the original filename / mime type for display + storage.
        content_type = image.content_type
        original_filename = image.filename or "document"

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

        print("Sending document to Gemini...")

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

        response_body = {
            "success": True,

            "filename": original_filename,

            "image": {
                "width": doc_rep["image_width"],
                "height": doc_rep["image_height"]
            },

            # Perspective-corrected / oriented image whose coordinate system
            # all overlay bboxes are normalized against. Render THIS image,
            # not the raw upload, so the overlays line up.
            "processed_image": doc_rep.get("processed_image_data_url"),

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

        # -------------------------------------------------
        # Persist upload + output (best-effort; never fails the request)
        # -------------------------------------------------

        if storage_enabled():
            try:
                # Persist the corrected image (the coordinate system all
                # overlays are normalized against) so the dashboard renders
                # accurate boxes. The original filename/mime are preserved.
                response_body["document_id"] = save_document(
                    original_filename,
                    content_type,
                    doc_rep.get("processed_image_data_url"),
                    doc_rep,
                    response_body,
                )
            except Exception as e:
                print(f"[STORAGE] save failed: {e}")
                response_body["document_id"] = None

        return response_body

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


# ---------------------------------------------------------
# Two-pass template pipeline
#
# Pass 1: register an EMPTY form -> build a template (field labels + value
#         regions). Pass 2: extract KV pairs from a FILLED form of the same
#         layout by warping it onto the template via homography alignment.
# Templates are stored as JSON + reference JPEG under uploads/templates/.
# ---------------------------------------------------------

def _save_upload(image: UploadFile):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")
    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with temp_file:
        shutil.copyfileobj(image.file, temp_file)
    return temp_file.name, image.filename or "document"


@app.post("/register-template")
async def register_form_template(image: UploadFile = File(...), name: str = ""):
    image_path, filename = _save_upload(image)
    try:
        template = register_template(image_path, name=name or None)
        return {"success": True, "template": template}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TEMPLATE] register failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register template: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.get("/templates")
async def templates_list():
    return {"templates": list_templates()}


@app.get("/templates/{template_id}")
async def templates_get(template_id: str):
    template = load_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found.")
    return {"template": template}


@app.get("/templates/{template_id}/image")
async def templates_image(template_id: str):
    payload = get_template_image(template_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Template image not found.")
    return Response(content=payload, media_type="image/jpeg")


@app.post("/extract-filled")
async def extract_filled_endpoint(
    image: UploadFile = File(...),
    template_id: str = Form(...),
):
    image_path, filename = _save_upload(image)
    try:
        template = load_template(template_id, with_reference=True)
        if template is None:
            raise HTTPException(status_code=404, detail="Template not found.")
        result = extract_filled(template, image_path)
        return {"success": True, "extraction": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TEMPLATE] extract failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract filled form: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


# ---------------------------------------------------------
# Document storage (dashboard)
# ---------------------------------------------------------

def _require_storage():
    if not storage_enabled():
        raise HTTPException(
            status_code=503,
            detail="Document storage is disabled (DATABASE_URL not set).",
        )


@app.get("/documents")
def documents_list(limit: int = 200):
    _require_storage()
    try:
        return {"documents": list_documents(limit)}
    except Exception as e:
        print(f"[STORAGE] list failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents.")


@app.get("/documents/{document_id}")
def documents_get(document_id: str):
    _require_storage()
    try:
        record = get_document(document_id)
    except Exception as e:
        print(f"[STORAGE] get failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load document.")
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return record


@app.get("/documents/{document_id}/image")
def documents_image(document_id: str):
    _require_storage()
    try:
        payload = get_document_file(document_id)
    except Exception as e:
        print(f"[STORAGE] image failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load image.")
    if payload is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    return Response(
        content=payload["image"],
        media_type=payload["mime_type"] or "image/jpeg",
    )


@app.delete("/documents/{document_id}")
def documents_delete(document_id: str):
    _require_storage()
    try:
        deleted = delete_document(document_id)
    except Exception as e:
        print(f"[STORAGE] delete failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document.")
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True}
