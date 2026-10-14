from fastapi import APIRouter

from app.core import store
from app.config import get_settings

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/overview")
def overview():
    stats = store.dashboard_stats()
    return stats


@router.get("/history")
def get_history(limit: int = 100):
    return {"items": store.history(limit)}


@router.get("/models")
def models_status():
    s = get_settings()
    return {
        "active_models": [
            {"name": "BLIP-1 VQA", "id": s.VQA_MODEL_ID, "status": "configured"},
            {"name": "ResNet-50 (EuroSAT)", "id": s.LANDCOVER_BACKBONE, "status": "configured"},
            {"name": "MobileNet-V3 (Change Detection)", "id": s.CHANGE_DETECTION_BACKBONE, "status": "configured"},
            {"name": "Dual ResNet-18 (SAR Fusion)", "id": s.SAR_FUSION_BACKBONE, "status": "configured"},
            {"name": "SAM ViT-B (Segmentation)", "id": s.SAM_MODEL_TYPE, "status": "configured"},
        ],
        "weights_mode": s.MODEL_WEIGHTS_MODE,
        "llm_priority": s.LLM_PROVIDER_PRIORITY,
        "datasets": s.DATASET_SOURCES,
    }
