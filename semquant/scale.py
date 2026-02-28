from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from skimage import measure, morphology


@dataclass
class ScaleBarDetection:
    bar_length_px: int
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col) in full image coords
    polarity: str  # "bright" or "dark"


def _candidates_from_mask(mask: np.ndarray, min_area_px: int, min_aspect_ratio: float):
    lab = measure.label(mask)
    props = measure.regionprops(lab)
    cands = []
    for p in props:
        if p.area < min_area_px:
            continue
        minr, minc, maxr, maxc = p.bbox
        h = maxr - minr
        w = maxc - minc
        if h <= 0 or w <= 0:
            continue
        if w <= h:
            continue
        ar = w / h
        if ar < min_aspect_ratio:
            continue
        cands.append((w, (minr, minc, maxr, maxc)))
    return cands


def detect_scale_bar(
    img: np.ndarray,
    search_bottom_fraction: float = 0.22,
    min_aspect_ratio: float = 8.0,
    min_area_px: int = 250,
) -> Optional[ScaleBarDetection]:
    """Detect a horizontal SEM scale bar near the bottom of the image."""
    x = np.asarray(img)
    if x.ndim != 2:
        raise ValueError("detect_scale_bar expects a 2D grayscale image.")
    h, w = x.shape
    frac = float(search_bottom_fraction)
    frac = min(max(frac, 0.05), 0.6)
    y0 = int(round(h * (1.0 - frac)))
    roi = x[y0:h, :]

    p_hi = float(np.percentile(roi, 99.5))
    p_lo = float(np.percentile(roi, 0.5))
    bright = roi >= p_hi
    dark = roi <= p_lo

    bright = morphology.binary_closing(bright, morphology.rectangle(3, 15))
    dark = morphology.binary_closing(dark, morphology.rectangle(3, 15))

    for mask_name, mask in [("bright", bright), ("dark", dark)]:
        mask = morphology.remove_small_objects(mask, min_size=min_area_px)
        cands = _candidates_from_mask(mask, min_area_px=min_area_px, min_aspect_ratio=min_aspect_ratio)
        if not cands:
            continue
        cands.sort(key=lambda t: t[0], reverse=True)
        bar_w, (minr, minc, maxr, maxc) = cands[0]
        bbox_full = (minr + y0, minc, maxr + y0, maxc)
        return ScaleBarDetection(bar_length_px=int(bar_w), bbox=bbox_full, polarity=mask_name)

    return None
