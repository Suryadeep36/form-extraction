import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from service.template_service import load_template
from service.filled_form_service import (
    save_filled_form,
    list_filled_forms,
    load_filled_form,
    delete_filled_form,
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

@router.post("/filled-forms")
def filled_forms_create(
    image: UploadFile = File(...),
    template_id: str = Form(...),
    user_id: str = Depends(get_current_user)
):
    image_path, filename = _save_upload(image)
    try:
        template = load_template(user_id, template_id, with_reference=True)
        if template is None:
            raise HTTPException(status_code=404, detail="Template not found.")
        record = save_filled_form(user_id, template, image_path, source_filename=filename)
        return {"success": True, "filled_form": record}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FILLED] save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save filled form: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@router.get("/filled-forms")
def filled_forms_list(template_id: str = None, user_id: str = Depends(get_current_user)):
    return {"filled_forms": list_filled_forms(user_id, template_id)}

@router.get("/templates/{template_id}/filled-forms")
def filled_forms_list_for_template(template_id: str, user_id: str = Depends(get_current_user)):
    return {"filled_forms": list_filled_forms(user_id, template_id)}

@router.get("/filled-forms/{filled_id}")
def filled_forms_get(filled_id: str, user_id: str = Depends(get_current_user)):
    record = load_filled_form(user_id, filled_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Filled form not found.")
    return {"filled_form": record}

@router.delete("/filled-forms/{filled_id}")
def filled_forms_delete(filled_id: str, user_id: str = Depends(get_current_user)):
    if not delete_filled_form(user_id, filled_id):
        raise HTTPException(status_code=404, detail="Filled form not found.")
    return {"success": True}
