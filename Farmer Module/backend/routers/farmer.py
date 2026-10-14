"""
Farmer Assist router.

Endpoints:
  POST /api/farmer/soil    -> soil type + texture + moisture-hint analysis
  POST /api/farmer/plant   -> plant/leaf health signal analysis
  POST /api/farmer/advise  -> combines both into general, non-personalized
                              agronomic pointers (never a prescription)

Wiring: add `from .routers import farmer` and
`app.include_router(farmer.router)` in your existing app/main.py — see
FARMER_MODULE_README.md for the exact lines.
"""
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from ..models.soil_classifier import classify_soil
from ..models.plant_health import classify_plant_health

router = APIRouter(prefix="/api/farmer", tags=["farmer"])

# General, publicly-known agronomic tendencies — NOT personalized advice,
# NOT a substitute for a local agricultural extension officer / KVK. Kept
# intentionally general so it can't be mistaken for a specific prescription.
_SOIL_CROP_NOTES = {
    "Black soil": "Retains moisture well; commonly associated with cotton, soybean, sugarcane in Indian agriculture.",
    "Alluvial soil": "Generally fertile, good water retention; widely used for wheat, rice, sugarcane.",
    "Red soil": "Lower natural fertility, drains faster; often needs added organic matter/fertilizer; used for groundnut, millets, pulses.",
    "Clay soil": "High water retention, can waterlog; benefits from drainage management.",
    "Laterite soil": "Low fertility, acidic tendency; often needs lime/organic amendment; used for cashew, tea in some regions.",
    "Sandy soil": "Drains quickly, low water/nutrient retention; benefits from frequent light irrigation and organic matter.",
}


class AdviseRequest(BaseModel):
    soil_type: str
    health_signal: str | None = None


@router.post("/soil")
async def analyze_soil(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    data = await file.read()
    try:
        result = classify_soil(data)
    except Exception as e:
        raise HTTPException(500, f"Soil analysis failed: {e}")
    return result


@router.post("/plant")
async def analyze_plant(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    data = await file.read()
    try:
        result = classify_plant_health(data)
    except Exception as e:
        raise HTTPException(500, f"Plant health analysis failed: {e}")
    return result


@router.post("/advise")
async def advise(req: AdviseRequest):
    soil_note = _SOIL_CROP_NOTES.get(
        req.soil_type,
        "No general note available for this soil type in our reference table.",
    )
    pointers = [f"Soil ({req.soil_type}): {soil_note}"]

    if req.health_signal:
        if "stress" in req.health_signal.lower() or "discoloration" in req.health_signal.lower():
            pointers.append(
                "Plant signal suggests possible stress/discoloration — check for nutrient "
                "deficiency, water stress, or early disease symptoms. Consider a closer "
                "in-person inspection or contact your local Krishi Vigyan Kendra (KVK)."
            )
        elif "healthy" in req.health_signal.lower():
            pointers.append("Plant signal looks healthy from this image — continue routine monitoring.")
        else:
            pointers.append("Plant signal was inconclusive from this image — try a clearer, well-lit close-up.")

    return {
        "pointers": pointers,
        "disclaimer": (
            "These are general, publicly-known agronomic tendencies for reference only — "
            "not a personalized recommendation, not a diagnosis, and not a substitute for "
            "a soil test or an agricultural extension officer."
        ),
    }
