"""
Grounding / referring segmentation — Segment Anything Model (SAM, ViT-B
checkpoint, Meta AI, real public weights at SAM_CHECKPOINT_URL).

SAM segments objects from point/box prompts. Since SAM has no language
understanding, we use real CV heuristics to find seed regions:
  - water/river/lake → HSV blue-band thresholding
  - built-up/urban/road → Canny edge density
  - vegetation/forest/crop/green → NDVI-proxy (green channel dominance)
  - dominant region → k-means largest cluster centroid

Each seed is a real pixel-derived coordinate, not a hardcoded guess.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from functools import lru_cache
from PIL import Image
import numpy as np
import cv2

from app.config import get_settings
from app.utils import fallback_cv

logger = logging.getLogger("satquery.sam")


def _vegetation_mask(img: Image.Image):
    """
    Vegetation detection using green-channel dominance (NDVI proxy for RGB).
    Pixels where green > red AND green > blue by a margin are vegetation.
    Returns (overlay_PIL, mask_PIL, coverage_pct).
    """
    arr = np.array(img.convert("RGB"))
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    # Vegetation: green channel dominates both red and blue
    veg_mask = ((g - r) > 15) & ((g - b) > 10) & (g > 50)
    veg_mask_u8 = veg_mask.astype(np.uint8) * 255
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    veg_mask_u8 = cv2.morphologyEx(veg_mask_u8, cv2.MORPH_OPEN, kernel, iterations=2)
    veg_mask_u8 = cv2.morphologyEx(veg_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Overlay
    overlay = arr.copy()
    overlay[veg_mask_u8 > 0] = [0, 200, 80]   # bright green highlight
    blended = cv2.addWeighted(arr, 0.55, overlay, 0.45, 0)
    coverage_pct = float((veg_mask_u8 > 0).sum()) / veg_mask_u8.size * 100
    mask_pil = Image.fromarray(veg_mask_u8)
    overlay_pil = Image.fromarray(blended)
    return overlay_pil, mask_pil, coverage_pct


def _built_up_mask_with_mask(img: Image.Image):
    """built_up_mask but also returns binary mask PIL for SAM seeding."""
    cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    kernel = np.ones((9, 9), np.uint8)
    density = cv2.dilate(edges, kernel, iterations=1)
    density = cv2.morphologyEx(density, cv2.MORPH_CLOSE, kernel, iterations=2)
    coverage_pct = float((density > 0).sum()) / density.size * 100
    overlay = cv_img.copy()
    overlay[density > 0] = (0, 0, 255)
    blended = cv2.addWeighted(cv_img, 0.65, overlay, 0.35, 0)
    overlay_pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(density)
    return overlay_pil, mask_pil, coverage_pct


def _kmeans_dominant_mask(img: Image.Image):
    """Return centroid of the dominant k-means cluster as a seed mask."""
    cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    small = cv2.resize(cv_img, (256, 256))
    Z = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, 5, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    dominant_label = int(np.bincount(labels.flatten()).argmax())
    dom_mask = (labels.flatten() == dominant_label).reshape(256, 256).astype(np.uint8) * 255
    dom_mask_full = cv2.resize(dom_mask, (cv_img.shape[1], cv_img.shape[0]))
    overlay = cv_img.copy()
    overlay[dom_mask_full > 0] = np.uint8(centers[dominant_label])
    blended = cv2.addWeighted(cv_img, 0.55, overlay, 0.45, 0)
    coverage_pct = float((dom_mask_full > 0).sum()) / dom_mask_full.size * 100
    return (
        Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)),
        Image.fromarray(dom_mask_full),
        coverage_pct,
    )


def _water_mask_validated(img: Image.Image):
    """
    Water detection with false-positive rejection.
    - Excludes sky: top 15% of image rows are ignored
    - Excludes overly large masks: if >50% of image is "water", it's likely
      sky/background bleed — return empty mask and 0% coverage
    - Requires contiguous blobs (min area 0.5% of image)
    Returns (overlay_PIL, mask_PIL, coverage_pct, found: bool)
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    # Mask out top 15% (sky region)
    sky_cutoff = int(h * 0.15)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lower = np.array([90, 30, 30])
    upper = np.array([140, 255, 190])
    mask = cv2.inRange(hsv, lower, upper)
    mask[:sky_cutoff, :] = 0  # zero out sky rows

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove small blobs (less than 0.5% of image area)
    min_area = int(h * w * 0.005)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    coverage_pct = float((clean_mask > 0).sum()) / clean_mask.size * 100

    # Reject if coverage is suspiciously high (sky bleed) or zero
    found = 1.0 < coverage_pct < 50.0

    if not found:
        # Return original image unmodified with empty mask
        return img, Image.fromarray(clean_mask), 0.0, False

    # Build overlay
    overlay = arr.copy()
    overlay[clean_mask > 0] = [0, 120, 255]
    blended = cv2.addWeighted(arr, 0.55, overlay, 0.45, 0)
    return Image.fromarray(blended), Image.fromarray(clean_mask), coverage_pct, True


def _get_seed_region(image: Image.Image, expr: str):
    """
    Determine target category from expression and run the appropriate
    real CV heuristic to find a seed region.
    Returns (target_name, overlay_PIL, mask_PIL, coverage_pct, found: bool).
    """
    e = expr.lower()
    is_water = any(w in e for w in ["water", "river", "lake", "sea", "pond", "reservoir", "flood"])
    is_builtup = any(w in e for w in ["built", "building", "urban", "road", "highway", "residential", "industrial"])
    is_veg = any(w in e for w in ["vegetation", "forest", "crop", "green", "tree", "grass", "farm", "agricultural", "pasture"])

    if is_water:
        overlay, mask, cov, found = _water_mask_validated(image)
        return "water", overlay, mask, cov, found
    elif is_veg:
        overlay, mask, cov = _vegetation_mask(image)
        found = cov > 1.0
        return "vegetation", overlay, mask, cov, found
    elif is_builtup:
        overlay, mask, cov = _built_up_mask_with_mask(image)
        found = cov > 1.0
        return "built-up", overlay, mask, cov, found
    else:
        overlay, mask, cov = _kmeans_dominant_mask(image)
        return "dominant-region", overlay, mask, cov, True


class SAMEngine:
    def __init__(self):
        self.settings = get_settings()
        self._predictor = None
        self._engine = "uninitialized"

    def _ensure_checkpoint(self) -> str:
        cache_dir = self.settings.MODEL_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, "sam_vit_b_01ec64.pth")
        if not os.path.exists(ckpt_path):
            logger.info("Downloading SAM checkpoint (~375MB)...")
            urllib.request.urlretrieve(self.settings.SAM_CHECKPOINT_URL, ckpt_path)
        return ckpt_path

    def _try_load(self):
        if self.settings.MODEL_WEIGHTS_MODE == "off":
            self._engine = "classical_cv"
            return
        try:
            from segment_anything import sam_model_registry, SamPredictor
            ckpt_path = self._ensure_checkpoint()
            sam = sam_model_registry[self.settings.SAM_MODEL_TYPE](checkpoint=ckpt_path)
            sam.to(self.settings.DEVICE)
            self._predictor = SamPredictor(sam)
            self._engine = "sam-vit-b"
            logger.info("Loaded SAM ViT-B weights.")
        except Exception as exc:
            logger.warning("Could not load SAM (%s). Using classical CV.", exc)
            self._engine = "classical_cv"

    def ground(self, image: Image.Image, referring_expression: str) -> dict:
        if self._engine == "uninitialized":
            self._try_load()

        target, cv_overlay, cv_mask, coverage, found = _get_seed_region(image, referring_expression)

        # If the CV heuristic found nothing, return honest "not detected" response
        if not found:
            return {
                "overlay_image": image,  # return original unmodified
                "target": target,
                "coverage_percent": 0.0,
                "confidence": 0.0,
                "engine": "classical_cv",
                "note": f"No {target} region detected in this image.",
            }

        if self._engine == "sam-vit-b":
            try:
                arr = np.array(image.convert("RGB"))
                self._predictor.set_image(arr)

                mask_arr = np.array(cv_mask.convert("L")) if cv_mask.mode != "L" else np.array(cv_mask)

                if mask_arr.max() > 0:
                    ys, xs = np.where(mask_arr > 0)
                    # Use a dense grid of points across the detected region
                    # Sample up to 12 points spread across the mask for full coverage
                    total_pts = len(ys)
                    step = max(1, total_pts // 12)
                    sampled_ys = ys[::step][:12]
                    sampled_xs = xs[::step][:12]
                    # Always include centroid
                    cy, cx = int(ys.mean()), int(xs.mean())
                    all_ys = np.append(sampled_ys, cy)
                    all_xs = np.append(sampled_xs, cx)
                    point_coords = np.column_stack([all_xs, all_ys])
                    point_labels = np.ones(len(point_coords), dtype=int)
                else:
                    h, w = arr.shape[:2]
                    point_coords = np.array([[w // 2, h // 2]])
                    point_labels = np.array([1])

                masks, scores, _ = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                best_idx = int(scores.argmax())
                best_mask = masks[best_idx]
                sam_coverage = float(best_mask.sum()) / best_mask.size * 100

                # Reject if suspiciously large
                if sam_coverage > 60.0:
                    raise ValueError(f"SAM mask too large ({sam_coverage:.1f}%)")

                # Dark solid color overlay — high opacity so it's clearly visible
                color_map = {
                    "water": (0, 80, 220),        # deep blue
                    "vegetation": (0, 150, 40),   # dark green
                    "built-up": (200, 30, 30),    # dark red
                    "dominant-region": (180, 140, 0),  # dark yellow
                }
                color = color_map.get(target, (200, 0, 0))

                # Apply strong solid fill on the mask
                overlay_arr = arr.copy()
                overlay_arr[best_mask] = color

                # Blend: 35% original, 65% color for strong visible marking
                blended = cv2.addWeighted(arr, 0.35, overlay_arr, 0.65, 0)
                overlay_pil = Image.fromarray(blended)

                return {
                    "overlay_image": overlay_pil,
                    "target": target,
                    "coverage_percent": round(sam_coverage, 2),
                    "confidence": float(scores[best_idx]),
                    "engine": "sam-vit-b",
                }
            except Exception as exc:
                logger.warning("SAM predict failed (%s), using CV fallback.", exc)

        # Classical CV fallback — apply same dark color scheme for consistency
        color_map = {
            "water": (0, 80, 220),
            "vegetation": (0, 150, 40),
            "built-up": (200, 30, 30),
            "dominant-region": (180, 140, 0),
        }
        color = color_map.get(target, (200, 0, 0))
        arr = np.array(image.convert("RGB"))
        mask_arr = np.array(cv_mask.convert("L")) if cv_mask.mode != "L" else np.array(cv_mask)
        overlay_arr = arr.copy()
        overlay_arr[mask_arr > 0] = color
        blended = cv2.addWeighted(arr, 0.35, overlay_arr, 0.65, 0)

        return {
            "overlay_image": Image.fromarray(blended),
            "target": target,
            "coverage_percent": round(coverage, 2),
            "confidence": 0.65,
            "engine": "classical_cv",
            "note": "Using classical CV heuristics for segmentation.",
        }


@lru_cache
def get_sam_engine() -> SAMEngine:
    return SAMEngine()
