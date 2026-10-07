import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import Response

from service.template_service import (
    register_template,
    extract_filled,
    load_template,
    list_templates,
    get_template_image,
)
from service.auth_service import get_current_user

router = APIRouter()

def _save_upload(image: UploadFile):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")
    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with temp_file:
        shutil.copyfileobj(image.file, temp_file)
    return temp_file.name, image.filename or "document"

@router.post("/register-template")
def register_form_template(image: UploadFile = File(...), name: str = Form(""), user_id: str = Depends(get_current_user)):
    image_path, filename = _save_upload(image)
    try:
        template = register_template(user_id, image_path, name=name or None)
        template["id"] = template["template_id"]
        return {"success": True, "template": template}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TEMPLATE] register failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register template: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@router.get("/templates")
def templates_list(user_id: str = Depends(get_current_user)):
    return {"templates": list_templates(user_id)}

@router.get("/templates/{template_id}")
def templates_get(template_id: str, user_id: str = Depends(get_current_user)):
    template = load_template(user_id, template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found.")
    return {"template": template}

@router.get("/templates/{template_id}/image")
def templates_image(template_id: str, user_id: str = Depends(get_current_user)):
    payload = get_template_image(user_id, template_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Template image not found.")
    return Response(content=payload, media_type="image/jpeg")

@router.post("/extract-filled")
def extract_filled_endpoint(
    image: UploadFile = File(...),
    template_id: str = Form(...),
    user_id: str = Depends(get_current_user)
):
    image_path, filename = _save_upload(image)
    try:
        template = load_template(user_id, template_id, with_reference=True)
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
