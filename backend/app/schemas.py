from typing import Optional, Any
from pydantic import BaseModel, Field


class ToolTrace(BaseModel):
    """One step the orchestrator took — shown in the UI for explainability."""
    tool: str
    reason: str
    duration_ms: float
    output_summary: str


class OrchestratorQuery(BaseModel):
    query: str
    session_id: Optional[str] = None
    image_t1_path: Optional[str] = None      # single image OR "earlier" bitemporal frame
    image_t2_path: Optional[str] = None      # "later" bitemporal frame
    optical_path: Optional[str] = None       # optical frame for fusion
    sar_path: Optional[str] = None           # SAR frame for fusion
    lat: Optional[float] = None
    lon: Optional[float] = None


class OrchestratorResponse(BaseModel):
    answer: str
    confidence: float
    processing_time_s: float
    tools_used: list[ToolTrace]
    task_type: str
    overlay_image_b64: Optional[str] = None
    mask_image_b64: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class VQARequest(BaseModel):
    image_path: str
    question: str


class ChangeDetectionRequest(BaseModel):
    image_t1_path: str
    image_t2_path: str
    question: Optional[str] = None


class FusionRequest(BaseModel):
    optical_path: str
    sar_path: str
    question: Optional[str] = None


class GroundingRequest(BaseModel):
    image_path: str
    referring_expression: str


class BatchTaskCreate(BaseModel):
    task_type: str  # "vqa" | "change_detection" | "fusion" | "single_image" | "grounding"
    payload: dict[str, Any]


class BatchTaskStatus(BaseModel):
    task_id: str
    task_type: str
    status: str   # queued | processing | completed | failed
    progress: int
    result: Optional[dict[str, Any]] = None


class FloodDetectionRequest(BaseModel):
    image_path: str
    resolution_m: Optional[float] = 10.0   # metres per pixel (default Sentinel-2)
