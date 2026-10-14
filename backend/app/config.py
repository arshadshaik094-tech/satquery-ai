"""
SatQuery AI — Central configuration.

All secrets are read from environment variables (.env). NOTHING is
hardcoded. If a key is missing, the relevant provider is simply
unavailable and the orchestrator falls back to whichever provider IS
configured (Groq OR OpenAI OR neither -> DEMO_MODE classical-CV fallback).
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------------- LLM providers (agentic orchestrator brain) ----------------
    # Leave blank / unset to disable a provider. At least one recommended.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Which provider to try first: "groq" | "openai" | "auto"
    LLM_PROVIDER_PRIORITY: str = os.getenv("LLM_PROVIDER_PRIORITY", "auto")

    # ---------------- ML weight behavior ----------------
    # "auto"   -> try to download real pretrained weights on first use (needs internet)
    # "local"  -> only use weights already cached under MODEL_CACHE_DIR, error if missing
    # "off"    -> never attempt to load deep weights; always use the classical-CV
    #             fallback pipeline (real algorithms: NDVI, k-means, Canny/SSIM diff,
    #             GrabCut-based water/building masks) — NOT random/fake numbers.
    MODEL_WEIGHTS_MODE: str = os.getenv("MODEL_WEIGHTS_MODE", "auto")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./model_cache")

    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cuda" if available on your machine

    # Fine-tuned model checkpoint paths (optional)
    CHANGE_HEAD_WEIGHTS: str = os.getenv("CHANGE_HEAD_WEIGHTS", "")
    EUROSAT_HEAD_WEIGHTS: str = os.getenv("EUROSAT_HEAD_WEIGHTS", "")
    SAR_FUSION_HEAD_WEIGHTS: str = os.getenv("SAR_FUSION_HEAD_WEIGHTS", "")
    SOIL_HEAD_WEIGHTS: str = os.getenv("SOIL_HEAD_WEIGHTS", "")
    PLANT_HEAD_WEIGHTS: str = os.getenv("PLANT_HEAD_WEIGHTS", "")

    # ---------------- Real model identifiers (as required by the brief) ----------------
    VQA_MODEL_ID: str = "Salesforce/blip-vqa-base"                 # BLIP-1 VQA
    LANDCOVER_BACKBONE: str = "resnet50.a1_in1k"                   # timm ResNet-50, fine-tuned head on EuroSAT
    CHANGE_DETECTION_BACKBONE: str = "mobilenetv3_large_100"       # timm MobileNet-V3
    SAR_FUSION_BACKBONE: str = "resnet18"                          # torchvision, dual-branch (optical+SAR)
    SAM_CHECKPOINT_URL: str = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )
    SAM_MODEL_TYPE: str = "vit_b"

    EUROSAT_CLASSES: list[str] = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    ]

    # ---------------- External live data sources (real, public, keyless) ----------------
    ESRI_WORLD_IMAGERY_TILE_TEMPLATE: str = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    NASA_EONET_URL: str = "https://eonet.gsfc.nasa.gov/api/v3/events"
    NASA_GIBS_WMTS_TEMPLATE: str = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
        "{layer}/default/{date}/{tile_matrix_set}/{z}/{y}/{x}.png"
    )
    NOMINATIM_URL: str = "https://nominatim.openstreetmap.org/search"
    NOMINATIM_USER_AGENT: str = "SatQueryAI/1.0 (SIH2026 project)"

    # ---------------- Dataset references (for provenance / citations in UI) --------------
    DATASET_SOURCES: dict = {
        "EuroSAT": "https://github.com/phelber/EuroSAT",
        "LEVIR-CD": "https://justchenhao.github.io/LEVIR/",
        "SEN12MS": "https://mediatum.ub.tum.de/1474000",
        "RSVQA-LR": "https://rsvqa.sylvainlobry.com/",
        "DOTA-v2": "https://captain-whu.github.io/DOTA/",
        "ISRO Bhuvan": "https://bhuvan.nrsc.gov.in/",
    }

    APP_NAME: str = "SatQuery AI"
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> "Settings":
    return Settings()
