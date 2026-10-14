"""
Bi-temporal change detection.

Two paths:
1. Fine-tuned SiameseMobileNetV3 (trained on LEVIR-CD) — loads from
   CHANGE_HEAD_WEIGHTS env var if the checkpoint exists.
2. Classical CV fallback — LAB-space diff + Otsu thresholding.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
from PIL import Image

from app.utils import fallback_cv

logger = logging.getLogger("satquery.change")


class SiameseChangeNet:
    """Wraps the trained SiameseMobileNetV3 for inference."""

    def __init__(self, ckpt_path: str):
        import torch
        import torch.nn as nn
        import timm

        class _SiameseChangeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(
                    "mobilenetv3_large_100", pretrained=False, features_only=True
                )
                feat_ch = self.backbone.feature_info.channels()[-1]
                self.decoder = nn.Sequential(
                    nn.Conv2d(feat_ch, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                )

            def forward(self, a, b):
                fa = self.backbone(a)[-1]
                fb = self.backbone(b)[-1]
                return self.decoder(torch.abs(fa - fb))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _SiameseChangeNet().to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.val_iou  = ckpt.get("val_iou", 0.0)
        self.epoch    = ckpt.get("epoch", "?")
        logger.info(
            "Loaded LEVIR-CD change head (epoch %s, val_IoU=%.4f) on %s",
            self.epoch, self.val_iou, self.device,
        )

    def predict(self, img_t1: Image.Image, img_t2: Image.Image):
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        a = tf(img_t1.convert("RGB")).unsqueeze(0).to(self.device)
        b = tf(img_t2.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(a, b)
            logits = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Binary mask at 0.5 threshold
        mask_arr = (prob > 0.5).astype(np.uint8) * 255

        # Resize mask back to original image size
        orig_w, orig_h = img_t1.size
        mask_img = Image.fromarray(mask_arr).resize((orig_w, orig_h), Image.NEAREST)
        mask_arr_full = np.array(mask_img)

        # Build overlay
        orig_arr = np.array(img_t1.convert("RGB"))
        overlay  = orig_arr.copy().astype(np.float32)
        changed  = mask_arr_full > 0
        # Highlight changed pixels in red
        overlay[changed, 0] = np.clip(overlay[changed, 0] * 0.3 + 180, 0, 255)
        overlay[changed, 1] = np.clip(overlay[changed, 1] * 0.3, 0, 255)
        overlay[changed, 2] = np.clip(overlay[changed, 2] * 0.3, 0, 255)
        overlay_img = Image.fromarray(overlay.astype(np.uint8))

        change_pct = float(changed.sum()) / changed.size * 100
        return overlay_img, mask_img, round(change_pct, 2)


class ChangeDetectionEngine:
    def __init__(self):
        self._siamese = None
        self._loaded  = False
        self._trained = False

    def _try_load(self):
        if self._loaded:
            return
        self._loaded = True

        ckpt_path = os.getenv("CHANGE_HEAD_WEIGHTS", "").strip()
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Also try relative to this file's location
            here = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.join(here, "..", "..", "..", "weights", "change_levir.pt")
            alt = os.path.normpath(alt)
            if os.path.exists(alt):
                ckpt_path = alt
            else:
                logger.warning(
                    "CHANGE_HEAD_WEIGHTS not set or file not found — "
                    "using classical LAB-diff fallback."
                )
                return

        try:
            self._siamese = SiameseChangeNet(ckpt_path)
            self._trained = True
        except Exception as exc:
            logger.warning("Failed to load LEVIR-CD head (%s) — using classical CV.", exc)

    @staticmethod
    def _images_are_identical(img_t1: Image.Image, img_t2: Image.Image, threshold: float = 0.5) -> bool:
        """
        Returns True when the two images are visually identical or near-identical.
        Compares mean absolute difference in LAB space on a small thumbnail
        so it's fast and colour-aware. A mean diff < threshold (default 0.5
        out of ~100) is treated as no meaningful change.
        """
        size = (128, 128)
        a = np.array(img_t1.convert("RGB").resize(size), dtype=np.float32)
        b = np.array(img_t2.convert("RGB").resize(size), dtype=np.float32)
        mean_diff = float(np.mean(np.abs(a - b)))
        return mean_diff < threshold

    def detect(self, img_t1: Image.Image, img_t2: Image.Image) -> dict:
        self._try_load()

        # Short-circuit: identical or near-identical images → no change
        if self._images_are_identical(img_t1, img_t2):
            # Return a blank (all-black) mask and the original image as overlay
            orig_w, orig_h = img_t1.size
            blank_mask    = Image.fromarray(np.zeros((orig_h, orig_w), dtype=np.uint8))
            overlay_img   = img_t1.convert("RGB").copy()
            return {
                "change_percent": 0.0,
                "trend":          "unchanged",
                "overlay_image":  overlay_img,
                "mask_image":     blank_mask,
                "engine":         "identity_check",
                "note": (
                    "Both images appear identical — no change detected."
                ),
            }

        if self._trained and self._siamese:
            try:
                overlay, mask, change_pct = self._siamese.predict(img_t1, img_t2)
                trend = (
                    "increased"    if change_pct > 8 else
                    "unchanged"    if change_pct < 2 else
                    "minor change"
                )
                return {
                    "change_percent": change_pct,
                    "trend":          trend,
                    "overlay_image":  overlay,
                    "mask_image":     mask,
                    "engine": (
                        f"SiameseMobileNetV3 fine-tuned on LEVIR-CD "
                        f"(epoch {self._siamese.epoch}, "
                        f"val_IoU={self._siamese.val_iou:.4f})"
                    ),
                    "note": None,
                }
            except Exception as exc:
                logger.warning("Siamese inference failed (%s) — falling back to CV.", exc)

        # Classical fallback
        overlay, mask, change_pct = fallback_cv.change_diff(img_t1, img_t2)
        trend = (
            "increased"    if change_pct > 8 else
            "unchanged"    if change_pct < 2 else
            "minor change"
        )
        return {
            "change_percent": round(change_pct, 2),
            "trend":          trend,
            "overlay_image":  overlay,
            "mask_image":     mask,
            "engine": "classical_cv",
            "note": (
                "LEVIR-CD fine-tuned head not loaded — result computed via "
                "LAB-space differencing + Otsu thresholding."
            ),
        }


@lru_cache
def get_change_engine() -> ChangeDetectionEngine:
    return ChangeDetectionEngine()
