import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core import store
from app.routers import upload, orchestrate, specialist, dashboard, batch, geodata, farmer

logging.basicConfig(level=logging.INFO)
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic multimodal remote-sensing VQA assistant (SIH 2026)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    store.init_db()
    logging.getLogger("satquery").info(
        "SatQuery AI backend started. LLM providers configured: groq=%s openai=%s. "
        "MODEL_WEIGHTS_MODE=%s",
        bool(settings.GROQ_API_KEY), bool(settings.OPENAI_API_KEY),
        settings.MODEL_WEIGHTS_MODE,
    )


app.include_router(upload.router)
app.include_router(orchestrate.router)
app.include_router(specialist.router)
app.include_router(dashboard.router)
app.include_router(batch.router)
app.include_router(geodata.router)
app.include_router(farmer.router)

# Serve uploaded images back to the frontend for preview
import os
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "llm_available": bool(settings.GROQ_API_KEY or settings.OPENAI_API_KEY),
        "weights_mode": settings.MODEL_WEIGHTS_MODE,
    }
