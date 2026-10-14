# Farmer Assist Module — wiring guide

Verified working standalone (backend router + React page both compiled and
ran against real test images) before packaging. Drop these files into your
existing SatQuery AI project as follows.

## 1. Backend

Copy into your existing `backend/app/`:
```
backend/app/models/soil_classifier.py
backend/app/models/plant_health.py
backend/app/utils/soil_texture_cv.py
backend/app/utils/plant_health_cv.py
backend/app/routers/farmer.py
```

In `backend/app/main.py`, add next to your other router includes:
```python
from .routers import farmer
app.include_router(farmer.router)
```

In `backend/requirements.txt`, add:
```
scikit-image==0.24.0
```
(everything else — torch, torchvision, timm, opencv, Pillow — you already have)

In `backend/.env.example` / `.env`, optionally add (leave blank to always
use the honest classical-CV fallback):
```
SOIL_HEAD_WEIGHTS=
PLANT_HEAD_WEIGHTS=
```
And confirm `backend/app/config.py`'s `Settings` class has
`SOIL_HEAD_WEIGHTS: str = ""` and `PLANT_HEAD_WEIGHTS: str = ""` fields
(same pattern as your existing `EUROSAT_HEAD_WEIGHTS` etc.) — add if missing.

## 2. Frontend

Copy `frontend/FarmerAssist.jsx` into `frontend/src/pages/FarmerAssist.jsx`.

In `frontend/src/components/Sidebar.jsx`, add to the `NAV` array (after
Grounding, before Batch Processing — matches where it sits in the agent's
tool list):
```jsx
import { Sprout } from "lucide-react"; // add to existing lucide import line

{ to: "/farmer-assist", label: "Farmer Assist", icon: Sprout },
```

In `frontend/src/App.jsx`, add the route and title:
```jsx
import FarmerAssist from "./pages/FarmerAssist";
// ...
<Route path="/farmer-assist" element={<FarmerAssist />} />
// and in the TITLES object:
"/farmer-assist": ["Farmer Assist", "Soil type & plant health from photos"],
```

## 3. (Optional) Orchestrator tool

To let the agent route natural-language questions like "what type of soil
is this?" here too, add a sixth tool in
`backend/app/core/llm_router.py`'s function-calling schema:

```python
{
  "name": "analyze_soil_and_plant",
  "description": "Classify soil type/texture and/or plant health from an uploaded photo.",
  "parameters": {"type": "object", "properties": {
      "soil_image_path": {"type": "string"},
      "plant_image_path": {"type": "string"}
  }}
}
```
and handle it in `orchestrator.py` by calling `classify_soil` /
`classify_plant_health` on the resolved file path, same pattern as your
existing five tools.

## What's honest here, restated

- Soil TYPE and plant HEALTH SIGNAL: real colour/texture analysis (GLCM,
  Excess Green Index) when no fine-tuned head is loaded — tagged
  `"engine": "classical_cv"`.
- Soil DENSITY: intentionally not estimated at all, anywhere in this
  module. A photo cannot produce that number honestly.
- Fine-tuning targets are real, named, public datasets (a public Kaggle
  soil-type dataset; PlantVillage for plant disease) — you'll need to
  download and train against them yourself; I could not do that from this
  sandbox (no dataset-host network access here).
