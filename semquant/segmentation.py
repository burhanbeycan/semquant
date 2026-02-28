from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from skimage import filters, morphology, segmentation


ThresholdMethod = Literal["otsu", "yen", "sauvola"]


def threshold_mask(img: np.ndarray, method: ThresholdMethod = "otsu") -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if method == "otsu":
        t = filters.threshold_otsu(x)
        return x >= t
    if method == "yen":
        t = filters.threshold_yen(x)
        return x >= t
    if method == "sauvola":
        h, w = x.shape
        win = int(max(31, min(h, w) // 12 * 2 + 1))
        t = filters.threshold_sauvola(x, window_size=win, k=0.2)
        return x >= t
    raise ValueError(f"Unknown threshold method: {method}")


def clean_binary_mask(
    mask: np.ndarray,
    min_object_area_px: int = 80,
    remove_border_objects: bool = True,
) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    m = morphology.remove_small_objects(m, min_size=int(min_object_area_px))
    m = morphology.binary_closing(m, morphology.disk(2))
    m = morphology.binary_opening(m, morphology.disk(1))
    if remove_border_objects:
        m = segmentation.clear_border(m)
    return m


def exclude_bottom(img: np.ndarray, fraction: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop out the bottom fraction of an image."""
    x = np.asarray(img)
    h, w = x.shape
    frac = float(fraction)
    frac = min(max(frac, 0.0), 0.9)
    maxr = int(round(h * (1.0 - frac)))
    cropped = x[:maxr, :]
    bbox = (0, 0, maxr, w)
    return cropped, bbox
