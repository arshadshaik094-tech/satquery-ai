"""
Plant leaf health / disease classifier.

Backbone: ResNet-18 (torchvision) — standard, lightweight choice for leaf
disease classification.

Fine-tune target (real, public, well-established): the PlantVillage
dataset — ~54,000 labelled leaf images across 14 crop species and 38
classes (healthy + specific diseases per species), e.g.
"Tomato___Early_blight", "Potato___Late_blight", "Apple___healthy" etc.
Public mirrors: https://github.com/spMohanty/PlantVillage-Dataset and the
Kaggle "New Plant Diseases Dataset" mirror. Point PLANT_HEAD_WEIGHTS at
your fine-tuned checkpoint once trained.

Until a fine-tuned head is provided, this defers to the classical-CV
greenness/discoloration heuristic in utils/plant_health_cv.py — it does
NOT invent a specific disease name.
"""
from __future__ import annotations
import io
from typing import Any

import numpy as np
from PIL import Image

from ..config import get_settings

settings = get_settings()
from ..utils.plant_health_cv import analyze_plant_photo

_model = None
# Representative subset of PlantVillage's 38 classes — extend to the full
# list once you fine-tune against the real dataset.
_labels = [
    "Apple___healthy", "Apple___Apple_scab", "Apple___Black_rot",
    "Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Potato___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Corn___healthy", "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Grape___healthy", "Grape___Black_rot",
]


def _try_load_model():
    global _model
    if _model is not None:
        return _model
    if settings.MODEL_WEIGHTS_MODE == "off":
        return None
    try:
        import torch
        import torchvision.models as tvm

        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if settings.MODEL_WEIGHTS_MODE == "auto" else None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(_labels))
        if settings.PLANT_HEAD_WEIGHTS:
            state = torch.load(settings.PLANT_HEAD_WEIGHTS, map_location=settings.DEVICE)
            model.load_state_dict(state)
            model.eval()
            _model = model.to(settings.DEVICE)
            return _model
        return None
    except Exception:
        return None


def classify_plant_health(image_bytes: bytes) -> dict[str, Any]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    model = _try_load_model()
    if model is not None and settings.PLANT_HEAD_WEIGHTS:
        import torch
        import torchvision.transforms as T

        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        x = tf(pil_img).unsqueeze(0).to(settings.DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_idx = int(torch.argmax(probs))
        label = _labels[top_idx]
        crop, condition = label.split("___") if "___" in label else (label, "unknown")
        return {
            "crop": crop,
            "condition": condition.replace("_", " "),
            "confidence": round(float(probs[top_idx]), 3),
            "engine": "resnet18.fine_tuned_plantvillage",
            "note": "Fine-tuned head loaded from PLANT_HEAD_WEIGHTS.",
        }

    # Honest fallback — real colour-signal analysis, not a guessed disease name.
    cv_bgr = np.array(pil_img)[:, :, ::-1].copy()
    return analyze_plant_photo(cv_bgr)
