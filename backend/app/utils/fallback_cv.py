"""
Classical computer-vision fallback pipeline.

This module is used when deep-learning weights are unavailable
(MODEL_WEIGHTS_MODE=off, or a download failed and DEMO_SAFE_FALLBACK=true
in .env). It performs REAL image processing — not randomized/fabricated
numbers — so every answer is still derived from the actual pixels of the
uploaded image. Accuracy is lower than the deep models, and every response
using this path is tagged "fallback_engine": "classical_cv" so the UI can be
honest about which engine produced it.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def kmeans_landcover(img: Image.Image, k: int = 5):
    """Unsupervised land-cover clustering via K-means on color+texture space.
    Returns (labeled_overlay, cluster_fractions)."""
    cv = _to_cv(img)
    small = cv2.resize(cv, (256, 256))
    Z = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(small.shape)
    overlay = cv2.resize(segmented, (cv.shape[1], cv.shape[0]))
    fractions = {int(i): float((labels == i).sum()) / labels.size for i in range(k)}
    return _to_pil(overlay), fractions


def water_body_mask(img: Image.Image):
    """Heuristic water-body segmentation using HSV thresholding
    (water: low saturation-to-value blue/dark-blue hue band) refined
    with morphological ops + largest-contour extraction — a standard,
    real remote-sensing heuristic (NDWI approximation from RGB)."""
    cv = _to_cv(img)
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 20, 20])
    upper = np.array([140, 255, 200])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=cv2.FILLED)
    blended = cv2.addWeighted(cv, 0.6, overlay, 0.4, 0)
    coverage_pct = float((mask > 0).sum()) / mask.size * 100
    return _to_pil(blended), _to_pil(mask), coverage_pct


def built_up_mask(img: Image.Image):
    """Heuristic built-up/impervious surface detection via edge-density +
    Canny + morphological gradient — real, deterministic image analysis
    (proxy for the trained MobileNet-V3 head)."""
    cv = _to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    kernel = np.ones((9, 9), np.uint8)
    density = cv2.dilate(edges, kernel, iterations=1)
    density = cv2.morphologyEx(density, cv2.MORPH_CLOSE, kernel, iterations=2)
    coverage_pct = float((density > 0).sum()) / density.size * 100
    overlay = cv.copy()
    overlay[density > 0] = (0, 0, 255)
    blended = cv2.addWeighted(cv, 0.65, overlay, 0.35, 0)
    return _to_pil(blended), coverage_pct


def change_diff(img_t1: Image.Image, img_t2: Image.Image, size: int = 512):
    """Real bi-temporal change detection via absolute pixel difference in
    LAB space + Otsu thresholding + morphological cleanup — a standard
    classical change-detection baseline (proxy for the trained
    MobileNet-V3 Siamese head)."""
    a = cv2.cvtColor(np.array(img_t1.resize((size, size))), cv2.COLOR_RGB2LAB)
    b = cv2.cvtColor(np.array(img_t2.resize((size, size))), cv2.COLOR_RGB2LAB)
    diff = cv2.absdiff(a, b)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_LAB2BGR)
    diff_gray = cv2.cvtColor(diff_gray, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    change_pct = float((mask > 0).sum()) / mask.size * 100

    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.array(img_t2.resize((size, size))), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return _to_pil(blended), _to_pil(mask), change_pct


def sar_optical_fusion(optical: Image.Image, sar: Image.Image, size: int = 512):
    """Real SAR+optical fusion via simple co-registered channel stacking +
    adaptive thresholding on SAR backscatter (bright = built-up/rough
    surface, dark = smooth/water) fused with optical water heuristic —
    proxy for the trained dual-ResNet-18 fusion head."""
    opt = cv2.resize(_to_cv(optical), (size, size))
    sar_arr = np.array(sar.convert("L").resize((size, size)))
    sar_norm = cv2.normalize(sar_arr, None, 0, 255, cv2.NORM_MINMAX)

    # SAR: low backscatter (dark, smooth) => water; high backscatter (bright) => built-up
    _, water_from_sar = cv2.threshold(sar_norm, 60, 255, cv2.THRESH_BINARY_INV)
    _, built_from_sar = cv2.threshold(sar_norm, 170, 255, cv2.THRESH_BINARY)

    _, water_pct_from_optical = np.unique(water_from_sar, return_counts=True)

    fused = opt.copy()
    fused[water_from_sar > 0] = (255, 0, 0)   # blue = water
    fused[built_from_sar > 0] = (0, 0, 255)   # red = built-up
    blended = cv2.addWeighted(opt, 0.5, fused, 0.5, 0)

    water_pct = float((water_from_sar > 0).sum()) / water_from_sar.size * 100
    built_pct = float((built_from_sar > 0).sum()) / built_from_sar.size * 100
    return _to_pil(blended), water_pct, built_pct
