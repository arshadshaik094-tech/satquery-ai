"""
Classical computer-vision soil analysis.

IMPORTANT — read before wiring this in:
True soil bulk density (g/cm^3) is a physical lab measurement (dry mass /
volume). It CANNOT be derived from an RGB photo, and this module makes no
attempt to fabricate one. What a photo genuinely supports, and what this
module actually computes:

  1. Colour-based soil-type hinting (Alluvial / Black / Red / Laterite /
     Sandy-ish) using real HSV/LAB colour statistics — a recognised first
     signal in soil-science literature, but NOT a lab-grade classification.
  2. A texture "coarseness" proxy using GLCM (grey-level co-occurrence
     matrix) contrast/homogeneity — a real, standard texture-analysis
     technique — reported as a qualitative bucket (fine/medium/coarse
     apparent texture), not a particle-size measurement.
  3. An apparent-moisture heuristic from brightness/darkness — real signal
     (wet soil is measurably darker), reported as a rough tendency, not a
     percentage moisture content.

Every function returns "engine": "classical_cv" and a "note" stating the
limitation, exactly like the rest of the SatQuery AI backend.
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Any

try:
    from skimage.feature import graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# Reference colour ranges are broad heuristics drawn from common soil-colour
# literature (Munsell-adjacent hue bands), not a calibrated instrument.
_SOIL_COLOR_HINTS = [
    # (name, HSV hue range, typical description)
    ("Black / Regur soil", (0, 179), "dark, low-value regardless of hue — checked separately"),
    ("Red soil",           (0, 15),  "iron-oxide reddish-brown hue"),
    ("Red soil",           (165, 179), "iron-oxide reddish-brown hue (wraps hue=0)"),
    ("Laterite soil",      (10, 25), "reddish-yellow, high iron/aluminium hue band"),
    ("Alluvial soil",      (20, 45), "pale yellow-brown, low saturation"),
    ("Sandy soil",         (25, 50), "light tan/beige, low saturation, high value"),
]


def analyze_soil_photo(bgr_image: np.ndarray) -> dict[str, Any]:
    img = cv2.resize(bgr_image, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    # --- 1. Colour-based type hint ---
    if v_mean < 70 and s_mean < 90:
        type_hint = "Black / Regur soil"
        type_reason = "low brightness, low-to-moderate saturation — consistent with dark clay-rich (regur) soils"
    else:
        type_hint, type_reason = "Alluvial soil", "default: moderate hue/saturation, no strong red/laterite signal"
        for name, (lo, hi), reason in _SOIL_COLOR_HINTS:
            if name in ("Black / Regur soil",):
                continue
            if lo <= h_mean <= hi:
                type_hint, type_reason = name, reason
                break

    # --- 2. Texture coarseness proxy (GLCM) ---
    if _HAS_SKIMAGE:
        glcm = graycomatrix(gray, distances=[3], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, "contrast")[0, 0])
        homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
        if contrast > 400:
            texture_bucket = "coarse apparent texture (sandy/gritty-looking)"
        elif contrast > 150:
            texture_bucket = "medium apparent texture (loamy-looking)"
        else:
            texture_bucket = "fine apparent texture (clay-looking, smooth)"
    else:
        contrast, homogeneity = None, None
        texture_bucket = "unavailable — scikit-image not installed"

    # --- 3. Apparent moisture tendency (darker => more likely moist) ---
    if v_mean < 60:
        moisture_hint = "appears dark — often correlates with higher moisture, but shadows/lighting can cause this too"
    elif v_mean > 160:
        moisture_hint = "appears bright/dry-looking"
    else:
        moisture_hint = "moderate brightness — inconclusive from colour alone"

    return {
        "soil_type_hint": type_hint,
        "type_reason": type_reason,
        "texture_apparent": texture_bucket,
        "glcm_contrast": round(contrast, 1) if contrast is not None else None,
        "glcm_homogeneity": round(homogeneity, 3) if homogeneity is not None else None,
        "moisture_hint": moisture_hint,
        "hsv_mean": {"h": round(h_mean, 1), "s": round(s_mean, 1), "v": round(v_mean, 1)},
        "confidence": 0.42,  # deliberately modest — this is a colour/texture heuristic, not a trained classifier
        "engine": "classical_cv",
        "note": (
            "Soil TYPE is a colour/texture heuristic only, not a lab or trained-model classification. "
            "Soil DENSITY (g/cm^3) is not estimated at all — it cannot be derived from a photo and this "
            "system will not fabricate that number. For a verified soil type or density reading, use a "
            "physical soil test kit or send a sample to a local agricultural lab / KVK (Krishi Vigyan Kendra)."
        ),
    }
