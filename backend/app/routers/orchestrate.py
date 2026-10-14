from fastapi import APIRouter, HTTPException

from app.schemas import OrchestratorQuery, OrchestratorResponse
from app.core.orchestrator import get_orchestrator
from app.core import store

router = APIRouter(prefix="/api/orchestrate", tags=["orchestrate"])


@router.post("", response_model=OrchestratorResponse)
def orchestrate(q: OrchestratorQuery):
    if not q.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    if not any([q.image_t1_path, q.optical_path]):
        raise HTTPException(
            400,
            "At least one image is required (image_t1_path for single/bi-temporal, "
            "or optical_path+sar_path for fusion).",
        )
    try:
        result = get_orchestrator().run(q)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    store.log_analysis(
        task_type=result.task_type,
        query=q.query,
        answer=result.answer,
        confidence=result.confidence,
        processing_time_s=result.processing_time_s,
        extra=result.extra,
    )
    return result
