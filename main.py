import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service import blob_store
from service.batch_service import start_workers

from routers.batch_router import router as batch_router
from routers.template_router import router as template_router
from routers.filled_form_router import router as filled_form_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(blob_store.log_status())
    start_workers()
    yield


app = FastAPI(
    title="Document Extraction API",
    version="2.0.0",
    lifespan=lifespan,
)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        FRONTEND_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(batch_router, tags=["Batch Processing"])
app.include_router(template_router, tags=["Templates"])
app.include_router(filled_form_router, tags=["Filled Forms"])
