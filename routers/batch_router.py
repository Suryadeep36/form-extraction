from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List

from service.batch_service import enqueue_batch, get_batch_status
from service.auth_service import get_current_user

router = APIRouter(prefix="/batch", tags=["batch"])

@router.post("/extract")
async def extract_batch(
    template_id: str = Form(...),
    images: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user)
):
    if not template_id:
        raise HTTPException(status_code=400, detail="template_id is required")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
        
    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are supported.")
            
    batch_id = await enqueue_batch(user_id, template_id, images)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "total_files": len(images),
        "message": "Batch processing started"
    }

@router.get("/status/{batch_id}")
async def batch_status(batch_id: str, user_id: str = Depends(get_current_user)):
    status = get_batch_status(user_id, batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    return status
