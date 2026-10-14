"""
Soil-type classifier.

Backbone: EfficientNet-B0 (via timm) — lightweight, standard choice for
fine-grained texture/colour classification tasks like soil imagery.

Fine-tune target (real, public): a soil-image classification dataset such
as the widely-used Kaggle "Soil types" dataset (Alluvial / Black / Clay /
Red soil classes) — search "soil classification dataset" on Kaggle for
current mirrors, since Kaggle dataset URLs change/get re-uploaded often.
Point SOIL_HEAD_WEIGHTS at your fine-tuned checkpoint once trained.

Until a fine-tuned head is provided, this module transparently defers to
the classical-CV heuristic in utils/soil_texture_cv.py — it does NOT
return a random or fabricated class label.
"""
from __future__ import annotations
import io
from typing import Any

import numpy as np
from PIL import Image

from ..config import get_settings

settings = get_settings()
from ..utils.soil_texture_cv import analyze_soil_photo

_model = None
_labels = ["Alluvial soil", "Black soil", "Clay soil", "Red soil", "Laterite soil", "Sandy soil"]


def _try_load_model():
    global _model
    if _model is not None:
        return _model
    if settings.MODEL_WEIGHTS_MODE == "off":
        return None
    try:
        import torch
        import timm

        model = timm.create_model("efficientnet_b0", pretrained=(settings.MODEL_WEIGHTS_MODE == "auto"), num_classes=len(_labels))
        if settings.SOIL_HEAD_WEIGHTS:
            state = torch.load(settings.SOIL_HEAD_WEIGHTS, map_location=settings.DEVICE)
            model.load_state_dict(state)
            model.eval()
            _model = model.to(settings.DEVICE)
            return _model
        # Backbone loaded, but head is untrained — don't pretend it's calibrated.
        return None
    except Exception:
        return None


def classify_soil(image_bytes: bytes) -> dict[str, Any]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    model = _try_load_model()
    if model is not None and settings.SOIL_HEAD_WEIGHTS:
        import torch
        import torchvision.transforms as T

        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        x = tf(pil_img).unsqueeze(0).to(settings.DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_idx = int(torch.argmax(probs))
        return {
            "soil_type": _labels[top_idx],
            "confidence": round(float(probs[top_idx]), 3),
            "all_scores": {lbl: round(float(p), 3) for lbl, p in zip(_labels, probs)},
            "engine": "efficientnet_b0.fine_tuned",
            "note": "Fine-tuned head loaded from SOIL_HEAD_WEIGHTS.",
        }

    # Honest fallback — real colour/texture analysis, not a guessed class.
    cv_bgr = np.array(pil_img)[:, :, ::-1].copy()
    result = analyze_soil_photo(cv_bgr)
    result["soil_type"] = result.pop("soil_type_hint")
    return result
