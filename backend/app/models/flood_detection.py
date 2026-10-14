"""
Flood / water extent detection.

Uses SegFormer-B2 pretrained on ADE20K (nvidia/segformer-b2-finetuned-ade-512-512).
ADE20K water classes: water(21), sea(26), river(60), lake(128), waterfall(113).

Model is pre-loaded in a background thread at startup so first inference is fast.
Falls back to classical CV if model is not yet loaded or unavailable.
"""
from __future__ import annotations
import logging
import threading
import numpy as np
from PIL import Image

logger = logging.getLogger("satquery")

WATER_CLASS_IDS = {21, 26, 60, 113, 128}
MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"

_model  = None
_proc   = None
_ready  = False   # True once loading succeeds
_failed = False   # True if loading failed permanently


def _preload():
    global _model, _proc, _ready, _failed
    try:
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        logger.info(f"Pre-loading SegFormer water-detection model ({MODEL_ID}) …")
        _proc  = SegformerImageProcessor.from_pretrained(MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        _model.eval()
        _ready = True
        logger.info("SegFormer water-detection model ready.")
    except Exception as exc:
        _failed = True
        logger.warning(f"SegFormer preload failed: {exc} — classical CV fallback will be used.")


# Start background load immediately when module is imported
threading.Thread(target=_preload, daemon=True).start()


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_flood(img: Image.Image, resolution_m: float = 10.0) -> dict:
    """
    Detect flood / water extent in a satellite RGB image.
    Uses SegFormer if loaded, otherwise classical CV fallback.
    """
    if _ready:
        return _segformer(img, resolution_m)
    return _classical(img, resolution_m)


# ── SegFormer path ─────────────────────────────────────────────────────────────

def _segformer(img: Image.Image, resolution_m: float) -> dict:
    import torch
    import torch.nn.functional as F
    import cv2

    rgb  = img.convert("RGB")
    W, H = rgb.size

    inputs = _proc(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits           # (1, C, H/4, W/4)

    logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = logits_up.argmax(dim=1).squeeze(0).numpy()

    water_mask = np.zeros((H, W), dtype=np.uint8)
    for cls_id in WATER_CLASS_IDS:
        water_mask[pred == cls_id] = 255

    # Clean up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k, iterations=2)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN,  k, iterations=1)

    pixel_count    = int(np.sum(water_mask > 0))
    total_pixels   = H * W
    flood_percent  = round(pixel_count / total_pixels * 100, 2)
    flood_area_km2 = round(pixel_count * (resolution_m ** 2) / 1_000_000, 2)
    confidence     = round(min(0.94, max(0.70, 0.70 + flood_percent / 150)), 2)

    overlay  = np.array(rgb).copy().astype(np.float32)
    flood_px = water_mask > 0
    overlay[flood_px, 0] = overlay[flood_px, 0] * 0.20
    overlay[flood_px, 1] = overlay[flood_px, 1] * 0.40 + 50
    overlay[flood_px, 2] = np.clip(overlay[flood_px, 2] * 0.50 + 140, 0, 255)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 130, 255), 2)

    return {
        "overlay_image":  Image.fromarray(overlay),
        "mask_image":     Image.fromarray(water_mask),
        "flood_area_km2": flood_area_km2,
        "flood_percent":  flood_percent,
        "confidence":     confidence,
        "pixel_count":    pixel_count,
        "engine": "SegFormer-B2 semantic segmentation (ADE20K water/sea/river/lake classes)",
        "note": "Water detected using SegFormer-B2 pretrained on ADE20K dataset.",
    }


# ── Classical fallback ─────────────────────────────────────────────────────────

def _classical(img: Image.Image, resolution_m: float) -> dict:
    import cv2

    rgb = np.array(img.convert("RGB"))
    H, W = rgb.shape[:2]
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    brightness = (r + g + b) / 3.0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1]

    is_veg      = (g > r + 8) & (g > b + 8) & (g > 45)
    is_bright   = brightness > 200
    not_water   = is_veg | is_bright

    water_blue   = (b > r + 5) & (b > g + 5) & (brightness > 30) & (~is_veg)
    water_dark   = (brightness < 70) & (sat < 80)
    water_turbid = (
        (brightness > 40) & (brightness < 180) & (sat < 110) &
        (np.abs(r - g) < 45) & (b < r + 20) & (~is_veg)
    )

    water_mask = ((water_blue | water_dark | water_turbid).astype(np.uint8) * 255)
    water_mask[not_water] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN,  k, iterations=1)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k, iterations=2)

    pixel_count    = int(np.sum(water_mask > 0))
    total_pixels   = H * W
    flood_percent  = round(pixel_count / total_pixels * 100, 2)
    flood_area_km2 = round(pixel_count * (resolution_m ** 2) / 1_000_000, 2)
    confidence     = round(min(0.72, max(0.55, 0.55 + flood_percent / 200)), 2)

    overlay  = rgb.copy().astype(np.float32)
    flood_px = water_mask > 0
    overlay[flood_px, 0] = overlay[flood_px, 0] * 0.20
    overlay[flood_px, 1] = overlay[flood_px, 1] * 0.40 + 50
    overlay[flood_px, 2] = np.clip(overlay[flood_px, 2] * 0.50 + 140, 0, 255)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 130, 255), 2)

    status = "loading" if not _failed else "unavailable"
    return {
        "overlay_image":  Image.fromarray(overlay),
        "mask_image":     Image.fromarray(water_mask),
        "flood_area_km2": flood_area_km2,
        "flood_percent":  flood_percent,
        "confidence":     confidence,
        "pixel_count":    pixel_count,
        "engine": f"Classical CV fallback (SegFormer model {status})",
        "note": (
            "SegFormer model is still loading in the background. "
            "Wait ~30 seconds and retry for deep learning detection."
            if status == "loading" else
            "Using classical water detection (NDWI-proxy + HSV heuristics)."
        ),
    }
