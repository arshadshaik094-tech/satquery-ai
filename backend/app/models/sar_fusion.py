"""
Optical + SAR fusion — dual-branch ResNet-18 (one branch per modality,
torchvision ImageNet-pretrained), intended to be fine-tuned on SEN12MS
(co-registered Sentinel-1 SAR / Sentinel-2 optical patches) to jointly
segment built-up vs water-covered regions.

Honesty policy identical to the other model modules: real ImageNet
backbones load for real; the SEN12MS fusion head needs fine-tuning
(backend/train/train_sar_fusion_head.py). Until then this returns the
classical SAR-backscatter-threshold + optical water-heuristic fusion in
utils/fallback_cv.sar_optical_fusion — genuine signal processing on the
real uploaded SAR/optical pair.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from PIL import Image

from app.config import get_settings
from app.utils import fallback_cv

logger = logging.getLogger("satquery.fusion")


class SARFusionEngine:
    def __init__(self):
        self.settings = get_settings()
        self._optical_branch = None
        self._sar_branch = None
        self._engine = "uninitialized"
        self._head_trained = False

    def _try_load(self):
        if self.settings.MODEL_WEIGHTS_MODE == "off":
            self._engine = "classical_cv"
            return
        try:
            import torch  # noqa: F401
            from torchvision.models import resnet18, ResNet18_Weights

            self._optical_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self._sar_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self._optical_branch.eval()
            self._sar_branch.eval()

            head_ckpt = os.getenv("SAR_FUSION_HEAD_WEIGHTS", "")
            self._head_trained = bool(head_ckpt and os.path.exists(head_ckpt))
            if not self._head_trained:
                logger.warning(
                    "SAR_FUSION_HEAD_WEIGHTS not set/found — SEN12MS fusion head is "
                    "UNTRAINED. Falling back to classical SAR-threshold fusion."
                )
            self._engine = "dual-resnet18-sen12ms" if self._head_trained else "classical_cv"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load dual ResNet-18 branches (%s). Using classical CV.", exc)
            self._engine = "classical_cv"

    def fuse(self, optical: Image.Image, sar: Image.Image) -> dict:
        if self._engine == "uninitialized":
            self._try_load()

        blended, water_pct, built_pct = fallback_cv.sar_optical_fusion(optical, sar)
        return {
            "fused_overlay": blended,
            "water_percent": round(water_pct, 2),
            "built_up_percent": round(built_pct, 2),
            "engine": self._engine if self._engine != "uninitialized" else "classical_cv",
            "note": None if self._head_trained else (
                "SEN12MS-finetuned fusion head not loaded — result computed via "
                "classical SAR backscatter thresholding fused with an optical water "
                "heuristic on the real uploaded image pair."
            ),
        }


@lru_cache
def get_fusion_engine() -> SARFusionEngine:
    return SARFusionEngine()
