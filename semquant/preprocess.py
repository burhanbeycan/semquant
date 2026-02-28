from __future__ import annotations

import numpy as np
from skimage import exposure, filters


def preprocess(img: np.ndarray, gaussian_sigma: float = 1.0, use_clahe: bool = True) -> np.ndarray:
    """Basic SEM-friendly preprocessing: optional CLAHE + Gaussian blur."""
    x = np.asarray(img, dtype=np.float32)
    if use_clahe:
        x = exposure.equalize_adapthist(x, clip_limit=0.01)
    if gaussian_sigma and gaussian_sigma > 0:
        x = filters.gaussian(x, sigma=float(gaussian_sigma), preserve_range=True)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return x
