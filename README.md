# SatQuery AI

An agentic multimodal Vision-Language assistant for satellite and aerial imagery analysis. Upload an image, ask a question in plain English, and the system routes it to the right specialist model and returns a grounded, explainable answer with a visual overlay.

Built for **SIH 2026**.

---

## Features

### Single Image Analysis
Upload any satellite or aerial image and ask a natural-language question about it. Combines BLIP-based VQA with ResNet-50 land-cover classification (EuroSAT) to give you both a direct answer and a scene label.

### Change Detection
Upload two images of the same area taken at different times. The system computes a bi-temporal change map using a Siamese MobileNetV3 architecture, highlights changed regions as a heatmap overlay, and reports the percentage area changed along with a trend label (increased / minor change / unchanged).

### Optical + SAR Fusion
Upload a co-registered optical (RGB) and SAR (grayscale backscatter) image pair of the same geographic area. The system fuses both modalities to identify water-covered and built-up regions, outputs a colour overlay, and reports percentage coverage for each class.

### Flood & Water Detection
Upload a satellite image of a potentially flood-affected area. The system segments water extent using an NDWI-proxy combined with HSV-space morphological analysis, overlays the detected flood mask, and estimates the affected area in km² given the image resolution.

### Grounding / Referring Expression Segmentation
Upload an image and describe a specific object or region (e.g. "the river" or "the industrial area"). SAM (Segment Anything Model, ViT-B) segments and highlights exactly that region in the image.

### Agentic Orchestration
All features are also accessible through a single unified query endpoint (`POST /api/orchestrate`). An LLM (Groq llama3-70b-8192, with OpenAI GPT-4o as fallback) reads your query and which images are attached, selects the right specialist tool via function-calling, runs it, and then narrates the model's real numeric output into a natural-language answer. The LLM only narrates — it never invents the underlying numbers.

### Live Earth Events Dashboard
The dashboard shows real-time disaster event pins from NASA EONET v3 and GDACS on an Esri World Imagery satellite basemap, with an optional Bhuvan ISRO LULC overlay. Includes stat cards for total analyses, change detections, VQA queries, and average model confidence from your session.

### Batch Processing
Queue multiple images for analysis and process them together.

### Analysis History
Every analysis run is persisted in SQLite and browsable in the History page with query, task type, result, and timestamp.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React (Vite), Tailwind CSS, React Leaflet |
| Backend | Python, FastAPI, Uvicorn |
| AI / ML | PyTorch, HuggingFace Transformers, timm |
| Models | BLIP-1 (VQA), ResNet-50 EuroSAT (land-cover), MobileNetV3 Siamese (change detection), Dual ResNet-18 (SAR fusion), SAM ViT-B (segmentation), SegFormer-B2 (water/flood) |
| LLM Orchestration | Groq (llama3-70b-8192) primary · OpenAI GPT-4o fallback |
| Data Sources | Esri World Imagery, NASA EONET v3, GDACS, Bhuvan ISRO (LULC WMS) |
| Persistence | SQLite (via store.py) |

---

## Project Structure

```
satquery-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, CORS, static uploads
│   │   ├── config.py                Settings, model IDs, env vars
│   │   ├── schemas.py               Pydantic request/response models
│   │   ├── core/
│   │   │   ├── orchestrator.py      Agentic tool selection + execution
│   │   │   ├── llm_router.py        Groq / OpenAI provider-agnostic client
│   │   │   ├── keyword_router.py    Deterministic fallback router (no LLM key needed)
│   │   │   └── store.py             SQLite history + batch queue
│   │   ├── models/
│   │   │   ├── vqa_blip.py          BLIP-1 VQA engine
│   │   │   ├── landcover_resnet.py  ResNet-50 EuroSAT land-cover classifier
│   │   │   ├── change_detection.py  Siamese MobileNetV3 change detector
│   │   │   ├── sar_fusion.py        Dual ResNet-18 SAR + optical fusion
│   │   │   ├── segmentation_sam.py  SAM ViT-B grounding
│   │   │   └── flood_detection.py   SegFormer flood / water extent detector
│   │   ├── routers/
│   │   │   ├── orchestrate.py       POST /api/orchestrate
│   │   │   ├── specialist.py        POST /api/vqa, /change-detection, /fusion, /grounding, /flood-detection
│   │   │   ├── upload.py            POST /api/upload
│   │   │   ├── dashboard.py         GET  /api/dashboard/overview
│   │   │   ├── batch.py             Batch queue endpoints
│   │   │   └── geodata.py           NASA EONET + GDACS proxy endpoints
│   │   ├── services/
│   │   │   └── external_data.py     Esri / NASA / GDACS / Nominatim API clients
│   │   └── utils/
│   │       ├── fallback_cv.py       Classical CV algorithms (LAB diff, HSV masking, K-means)
│   │       └── image_utils.py       PIL helpers, base64 encoding
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── src/
        ├── pages/                   Dashboard, SingleImageAnalysis, ChangeDetection,
        │                            OpticalSarAnalysis, GroundingVQA, FloodDetection,
        │                            BatchProcessing, History, Datasets, Models,
        │                            Reports, Settings, HelpDocs
        ├── components/              Sidebar, Topbar, StatCard, UploadDropzone,
        │                            ImageCompareSlider
        └── lib/api.js               Axios client mapping to all backend routes
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Groq API key (free at [console.groq.com](https://console.groq.com)) — or an OpenAI key

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and add your GROQ_API_KEY
```

`.env` key settings:

```env
GROQ_API_KEY=your_key_here
MODEL_WEIGHTS_MODE=auto   # downloads pretrained weights on first use
DEVICE=cpu
```

```bash
uvicorn app.main:app --reload --port 8000
```

Health check: `http://localhost:8000/api/health`  
API docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
# runs at http://localhost:5174
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/api/upload` | Upload an image, get back a server path |
| POST | `/api/orchestrate` | Unified agentic query (LLM routes to the right model) |
| POST | `/api/vqa` | Single image VQA + land-cover classification |
| POST | `/api/change-detection` | Bi-temporal change detection |
| POST | `/api/fusion` | Optical + SAR fusion |
| POST | `/api/grounding` | Referring expression segmentation |
| POST | `/api/flood-detection` | Flood / water extent detection |
| GET | `/api/dashboard/overview` | Session stats |
| GET | `/api/health` | Backend status |
