from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import BatchTaskCreate
from app.core import store
from app.utils.image_utils import load_image
from app.models.vqa_blip import get_vqa_engine
from app.models.landcover_resnet import get_landcover_engine
from app.models.change_detection import get_change_engine
from app.models.sar_fusion import get_fusion_engine
from app.models.segmentation_sam import get_sam_engine

router = APIRouter(prefix="/api/batch", tags=["batch"])


def _run_task(task_id: str, task_type: str, payload: dict):
    store.update_batch_task(task_id, "processing", 10)
    try:
        if task_type == "vqa":
            img = load_image(payload["image_path"])
            store.update_batch_task(task_id, "processing", 50)
            result = get_vqa_engine().answer(img, payload["question"])
        elif task_type == "single_image":
            img = load_image(payload["image_path"])
            store.update_batch_task(task_id, "processing", 50)
            result = get_landcover_engine().classify(img)
        elif task_type == "change_detection":
            t1 = load_image(payload["image_t1_path"])
            t2 = load_image(payload["image_t2_path"])
            store.update_batch_task(task_id, "processing", 50)
            res = get_change_engine().detect(t1, t2)
            res.pop("overlay_image", None)
            res.pop("mask_image", None)
            result = res
        elif task_type == "fusion":
            opt = load_image(payload["optical_path"])
            sar = load_image(payload["sar_path"])
            store.update_batch_task(task_id, "processing", 50)
            res = get_fusion_engine().fuse(opt, sar)
            res.pop("fused_overlay", None)
            result = res
        elif task_type == "grounding":
            img = load_image(payload["image_path"])
            store.update_batch_task(task_id, "processing", 50)
            res = get_sam_engine().ground(img, payload["referring_expression"])
            res.pop("overlay_image", None)
            result = res
        else:
            raise ValueError(f"Unknown batch task_type '{task_type}'")

        store.update_batch_task(task_id, "completed", 100, result)
    except Exception as exc:  # noqa: BLE001
        store.update_batch_task(task_id, "failed", 100, {"error": str(exc)})


@router.post("")
def create_batch(task: BatchTaskCreate, background_tasks: BackgroundTasks):
    task_id = store.create_batch_task(task.task_type, task.payload)
    background_tasks.add_task(_run_task, task_id, task.task_type, task.payload)
    return {"task_id": task_id, "status": "queued"}


@router.get("/{task_id}")
def get_task(task_id: str):
    t = store.get_batch_task(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t


@router.get("")
def list_tasks():
    return {"items": store.list_batch_tasks()}
