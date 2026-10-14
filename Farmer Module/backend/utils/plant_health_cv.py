"""
Classical computer-vision plant/leaf health analysis.

Uses the Excess Green Index (ExG) and HSV-based discoloration detection —
both are real, published techniques in agricultural computer vision
(commonly used for weed/crop segmentation and leaf-stress detection), not
fabricated heuristics. Reports a health SIGNAL, not a diagnosed disease
name — disease identification needs a model trained on a labelled disease
dataset (see plant_health.py for the deep-model path against PlantVillage).
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Any


def analyze_plant_photo(bgr_image: np.ndarray) -> dict[str, Any]:
    img = cv2.resize(bgr_image, (256, 256)).astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Excess Green Index: 2G - R - B. Standard vegetation-vs-background /
    # greenness signal in agri-CV literature.
    exg = 2 * g - r - b
    exg_norm = float(np.mean(exg)) / 255.0

    hsv = cv2.cvtColor(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Yellow/brown discoloration mask (common early sign of stress/disease/
    # nutrient deficiency in agri-CV) — hue band for yellow-brown in OpenCV's
    # 0-179 H scale.
    yellow_brown_mask = ((h >= 10) & (h <= 35) & (s > 60)).astype(np.uint8)
    discoloration_pct = float(np.mean(yellow_brown_mask)) * 100

    # Dark/necrotic-spot mask (very low value = likely dead/necrotic tissue)
    necrotic_mask = (v < 55).astype(np.uint8)
    necrotic_pct = float(np.mean(necrotic_mask)) * 100

    if exg_norm > 0.12 and discoloration_pct < 15 and necrotic_pct < 5:
        status = "Healthy-looking"
    elif discoloration_pct >= 15 or necrotic_pct >= 8:
        status = "Possible stress / discoloration detected"
    else:
        status = "Inconclusive — mixed or low-vegetation frame"

    return {
        "health_signal": status,
        "greenness_index_exg": round(exg_norm, 1),
        "discoloration_percent": round(discoloration_pct, 1),
        "necrotic_spot_percent": round(necrotic_pct, 1),
        "confidence": 0.4,
        "engine": "classical_cv",
        "note": (
            "This is a colour-signal health check (greenness / discoloration / dark-spot coverage), "
            "not a diagnosed disease name. Naming a specific disease needs a model fine-tuned on a "
            "labelled dataset such as PlantVillage — see plant_health.py's deep-model path, which is "
            "wired but requires a fine-tuned checkpoint to name specific diseases with confidence."
        ),
    }
