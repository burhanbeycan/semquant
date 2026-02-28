from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io as skio

from .utils import as_gray_float


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from disk and return grayscale float in [0,1]."""
    img = skio.imread(str(path))
    return as_gray_float(img)


def save_png(path: str | Path, img: np.ndarray) -> None:
    """Save an array to PNG. Handles float [0,1] and bool masks."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(img)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    skio.imsave(str(p), arr, check_contrast=False)
