import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    """Load an image from disk as RGB. Raises FileNotFoundError with a clear
    message if the path is wrong — never silently substitutes a blank/fake image."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Image not found at '{path}'. Upload the file first via /api/upload."
        )
    return Image.open(p).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


def resize_pair(img_a: Image.Image, img_b: Image.Image, size: int = 512):
    """Co-register two frames onto a common grid via simple resize.
    (Real geometric co-registration for user uploads without EXIF/geo metadata
    uses SIFT/ORB keypoint alignment — see services/coregistration.py.)"""
    a = img_a.resize((size, size))
    b = img_b.resize((size, size))
    return a, b
