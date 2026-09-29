import asyncio
import uuid
import os
import shutil
import tempfile
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from service.template_service import load_template, extract_filled
from service.filled_form_service import save_filled_form

# Global in-memory stores
batch_jobs = {}
batch_queue = asyncio.Queue()

# Workers configuration
MAX_WORKERS = 2
workers = []

async def worker_task(worker_id: int):
    print(f"[BATCH] Worker {worker_id} started")
    while True:
        job = await batch_queue.get()
        user_id = job["user_id"]
        batch_id = job["batch_id"]
        template_id = job["template_id"]
        image_path = job["image_path"]
        file_index = job["file_index"]
        
        try:
            # Check if batch was cancelled or not found
            if batch_id not in batch_jobs:
                continue
                
            print(f"[BATCH] Worker {worker_id} processing file {file_index} for batch {batch_id}")
            
            # Load template and run extraction in a thread pool to avoid blocking asyncio
            template = load_template(user_id, template_id)
            if not template:
                raise Exception("Template not found")
                
            record = await run_in_threadpool(save_filled_form, user_id, template, image_path, job["filename"])
            
            batch_jobs[batch_id]["results"][file_index] = {
                "success": True,
                "data": record["extraction"],
                "filename": job["filename"]
            }
            
        except Exception as e:
            print(f"[BATCH] Worker {worker_id} error: {e}")
            batch_jobs[batch_id]["results"][file_index] = {
                "success": False,
                "error": str(e),
                "filename": job["filename"]
            }
            batch_jobs[batch_id]["errors"] += 1
            
        finally:
            batch_jobs[batch_id]["completed"] += 1
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            batch_queue.task_done()
            
def start_workers():
    global workers
    if not workers:
        for i in range(MAX_WORKERS):
            task = asyncio.create_task(worker_task(i))
            workers.append(task)
            
async def enqueue_batch(user_id: str, template_id: str, files: list[UploadFile]):
    batch_id = str(uuid.uuid4())
    total_files = len(files)
    
    batch_jobs[batch_id] = {
        "user_id": user_id,
        "batch_id": batch_id,
        "template_id": template_id,
        "total": total_files,
        "completed": 0,
        "errors": 0,
        "results": [None] * total_files,
        "status": "processing"
    }
    
    for i, file in enumerate(files):
        suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
            
        await batch_queue.put({
            "user_id": user_id,
            "batch_id": batch_id,
            "template_id": template_id,
            "image_path": temp_file.name,
            "filename": file.filename,
            "file_index": i
        })
        
    return batch_id

def get_batch_status(user_id: str, batch_id: str):
    if batch_id not in batch_jobs:
        return None
    
    job = batch_jobs[batch_id]
    if job.get("user_id") != user_id:
        return None
        
    if job["completed"] == job["total"]:
        job["status"] = "completed"
        
    return job
