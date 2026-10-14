import time
from fastapi import APIRouter, HTTPException

from app.schemas import (
    VQARequest, ChangeDetectionRequest, FusionRequest, GroundingRequest,
    FloodDetectionRequest,
)
from app.models.vqa_blip import get_vqa_engine
from app.models.landcover_resnet import get_landcover_engine
from app.models.change_detection import get_change_engine
from app.models.sar_fusion import get_fusion_engine
from app.models.segmentation_sam import get_sam_engine
from app.models.flood_detection import detect_flood
from app.utils.image_utils import load_image, pil_to_b64
from app.core import store

router = APIRouter(prefix="/api", tags=["specialist"])


def _make_trace(tool: str, reason: str, t0: float, summary: str) -> dict:
    return {
        "tool": tool,
        "reason": reason,
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "output_summary": str(summary)[:200],
    }


@router.post("/vqa")
def vqa(req: VQARequest):
    t_start = time.time()
    try:
        img = load_image(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    t0 = time.time()
    result = get_vqa_engine().answer(img, req.question)
    vqa_trace = _make_trace(
        "BLIP-VQA (Salesforce/blip-vqa-base)",
        f"Open-ended VQA: '{req.question}'",
        t0,
        f"answer={result.get('answer')} conf={result.get('confidence', 0):.2f}",
    )

    t0 = time.time()
    lc = get_landcover_engine().classify(img)
    lc_trace = _make_trace(
        "ResNet-50 EuroSAT land-cover",
        "Scene classification on uploaded image",
        t0,
        f"label={lc.get('label')} conf={lc.get('confidence', 0):.2f}",
    )

    processing_time_s = round(time.time() - t_start, 2)
    store.log_analysis(
        "describe_image", req.question, result.get("answer", ""),
        result.get("confidence", 0.5), processing_time_s, {"vqa": result, "landcover": lc},
    )
    return {
        "vqa": result,
        "landcover": lc,
        "tools_used": [vqa_trace, lc_trace],
        "processing_time_s": processing_time_s,
    }


@router.post("/change-detection")
def change_detection(req: ChangeDetectionRequest):
    t_start = time.time()
    try:
        t1 = load_image(req.image_t1_path)
        t2 = load_image(req.image_t2_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    t0 = time.time()
    result = get_change_engine().detect(t1, t2)
    overlay_b64 = pil_to_b64(result.pop("overlay_image"))
    mask_b64 = pil_to_b64(result.pop("mask_image"))
    trace = _make_trace(
        "MobileNetV3 Siamese change detector",
        "Bi-temporal LAB-space differencing + Otsu threshold",
        t0,
        f"change={result.get('change_percent')}% trend={result.get('trend')} engine={result.get('engine')}",
    )

    processing_time_s = round(time.time() - t_start, 2)
    store.log_analysis(
        "detect_change", req.question or "change detection",
        result.get("trend", ""), 0.7, processing_time_s, result,
    )
    return {
        **result,
        "overlay_image_b64": overlay_b64,
        "mask_image_b64": mask_b64,
        "tools_used": [trace],
        "processing_time_s": processing_time_s,
    }


@router.post("/fusion")
def fusion(req: FusionRequest):
    t_start = time.time()
    try:
        opt = load_image(req.optical_path)
        sar = load_image(req.sar_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    t0 = time.time()
    result = get_fusion_engine().fuse(opt, sar)
    overlay_b64 = pil_to_b64(result.pop("fused_overlay"))
    trace = _make_trace(
        "Dual ResNet-18 SAR/Optical fusion",
        "SAR backscatter thresholding + optical water heuristic",
        t0,
        f"water={result.get('water_percent')}% built_up={result.get('built_up_percent')}% engine={result.get('engine')}",
    )

    processing_time_s = round(time.time() - t_start, 2)
    store.log_analysis(
        "fuse_optical_sar", req.question or "fusion analysis",
        f"water {result.get('water_percent')}%, built-up {result.get('built_up_percent')}%",
        0.7, processing_time_s, result,
    )
    return {
        **result,
        "overlay_image_b64": overlay_b64,
        "tools_used": [trace],
        "processing_time_s": processing_time_s,
    }


@router.post("/grounding")
def grounding(req: GroundingRequest):
    t_start = time.time()
    try:
        img = load_image(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    t0 = time.time()
    result = get_sam_engine().ground(img, req.referring_expression)
    overlay_b64 = pil_to_b64(result.pop("overlay_image"))
    trace = _make_trace(
        "SAM ViT-B grounding",
        f"Referring expression: '{req.referring_expression}'",
        t0,
        f"target={result.get('target')} coverage={result.get('coverage_percent')}% engine={result.get('engine')}",
    )

    processing_time_s = round(time.time() - t_start, 2)
    store.log_analysis(
        "ground_referring_expression", req.referring_expression,
        f"highlighted {result.get('target')}", result.get("confidence", 0.5),
        processing_time_s, result,
    )
    return {
        **result,
        "overlay_image_b64": overlay_b64,
        "tools_used": [trace],
        "processing_time_s": processing_time_s,
    }


@router.post("/flood-detection")
def flood_detection(req: FloodDetectionRequest):
    t_start = time.time()
    try:
        img = load_image(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    t0 = time.time()
    result = detect_flood(img, req.resolution_m or 10.0)
    overlay_b64 = pil_to_b64(result.pop("overlay_image"))
    mask_b64    = pil_to_b64(result.pop("mask_image"))

    trace = _make_trace(
        "Flood/Water Extent Detector",
        "NDWI-proxy + HSV heuristic + morphological cleanup",
        t0,
        f"area={result.get('flood_area_km2')} km² "
        f"coverage={result.get('flood_percent')}% "
        f"conf={result.get('confidence')}",
    )

    processing_time_s = round(time.time() - t_start, 2)
    store.log_analysis(
        "flood_detection",
        "Detect flood affected areas",
        f"{result.get('flood_area_km2')} km² affected",
        result.get("confidence", 0.7),
        processing_time_s,
        result,
    )
    return {
        **result,
        "overlay_image_b64": overlay_b64,
        "mask_image_b64": mask_b64,
        "tools_used": [trace],
        "processing_time_s": processing_time_s,
    }
