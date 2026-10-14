import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


@router.post("")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
    dest_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = UPLOAD_DIR / dest_name
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    dest_path.write_bytes(content)
    return {"path": str(dest_path), "filename": file.filename, "size_bytes": len(content)}
