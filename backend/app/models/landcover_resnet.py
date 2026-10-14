"""
Land-cover scene classification — ResNet-50 (timm: resnet50.a1_in1k),
ImageNet-pretrained backbone with a linear head intended to be fine-tuned
on EuroSAT (10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway,
Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).

IMPORTANT / HONEST NOTE: the ImageNet-pretrained backbone loads real
weights from timm, but the EuroSAT-specific classification head is
UNTRAINED unless you point EUROSAT_HEAD_WEIGHTS at a fine-tuned checkpoint
(see README "Fine-tuning the EuroSAT head" section — training script
included at backend/train/train_eurosat_head.py). Until you fine-tune it,
this module clearly labels its output "backbone-only (untrained head)" and
uses the classical-CV k-means clustering as the honest default instead of
guessing EuroSAT class probabilities from an untrained layer.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from PIL import Image

from app.config import get_settings
from app.utils import fallback_cv

logger = logging.getLogger("satquery.landcover")


class LandcoverEngine:
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._transform = None
        self._engine = "uninitialized"
        self._head_trained = False

    def _try_load(self):
        if self.settings.MODEL_WEIGHTS_MODE == "off":
            self._engine = "classical_cv"
            return
        try:
            import timm
            import torch
            from torchvision import transforms

            self._model = timm.create_model(
                self.settings.LANDCOVER_BACKBONE,
                pretrained=True,
                num_classes=len(self.settings.EUROSAT_CLASSES),
            ).to(self.settings.DEVICE)

            head_ckpt = os.getenv("EUROSAT_HEAD_WEIGHTS", "")
            if head_ckpt and os.path.exists(head_ckpt):
                state = torch.load(head_ckpt, map_location=self.settings.DEVICE)
                self._model.load_state_dict(state)
                self._head_trained = True
                logger.info("Loaded fine-tuned EuroSAT head from %s", head_ckpt)
            else:
                logger.warning(
                    "EUROSAT_HEAD_WEIGHTS not set/found — classification head is "
                    "UNTRAINED. Run backend/train/train_eurosat_head.py on the real "
                    "EuroSAT dataset to produce a valid checkpoint."
                )

            self._model.eval()
            self._torch = torch
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
            self._engine = "resnet50-eurosat"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load ResNet-50 backbone (%s). Using classical CV.", exc)
            self._engine = "classical_cv"

    def classify(self, image: Image.Image) -> dict:
        if self._engine == "uninitialized":
            self._try_load()

        if self._engine == "resnet50-eurosat" and self._head_trained:
            x = self._transform(image).unsqueeze(0).to(self.settings.DEVICE)
            with self._torch.no_grad():
                logits = self._model(x)
                probs = self._torch.softmax(logits, dim=1)[0]
            top_idx = int(probs.argmax())
            return {
                "label": self.settings.EUROSAT_CLASSES[top_idx],
                "confidence": float(probs[top_idx]),
                "all_probs": {
                    c: float(p) for c, p in zip(self.settings.EUROSAT_CLASSES, probs.tolist())
                },
                "engine": "resnet50-eurosat",
            }

        # ---- honest fallback: real k-means clustering, not a guessed EuroSAT label ----
        overlay, fractions = fallback_cv.kmeans_landcover(image)
        return {
            "label": "unsupervised-cluster-summary",
            "clusters": fractions,
            "confidence": 0.5,
            "engine": "classical_cv",
            "note": "Showing pixel color/texture clusters. Labels like 'Forest' or 'River' require fine-tuning on EuroSAT.",
        }


@lru_cache
def get_landcover_engine() -> LandcoverEngine:
    return LandcoverEngine()
