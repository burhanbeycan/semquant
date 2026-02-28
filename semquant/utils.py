from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def versions_info() -> Dict[str, Any]:
    import numpy
    import pandas
    import scipy
    import skimage
    import matplotlib
    import reportlab

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "matplotlib": matplotlib.__version__,
        "reportlab": reportlab.__version__,
    }


_UNIT_TO_UM = {
    "nm": 1e-3,
    "um": 1.0,
    "µm": 1.0,
    "mm": 1e3,
}


def unit_to_um_factor(unit: str) -> float:
    u = unit.strip().lower()
    if u not in _UNIT_TO_UM:
        raise ValueError(f"Unsupported unit '{unit}'. Use one of: {sorted(_UNIT_TO_UM)}")
    return _UNIT_TO_UM[u]


def robust_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


@dataclass
class Calibration:
    pixel_size_um: Optional[float]  # micrometers per pixel
    source: str  # e.g. "pixel_size_cli", "scale_bar_detected", "uncalibrated"

    def is_calibrated(self) -> bool:
        return self.pixel_size_um is not None and math.isfinite(self.pixel_size_um) and self.pixel_size_um > 0


def as_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert an image to 2D grayscale float32 in [0, 1]."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2])
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr
