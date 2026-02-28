from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

from .utils import Calibration, robust_percentile


def regionprops_dataframe(
    mask: np.ndarray,
    calibration: Calibration,
    kind: str = "object",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Measure connected components in a binary mask."""
    m = np.asarray(mask, dtype=bool)
    lab = measure.label(m)
    props = measure.regionprops_table(
        lab,
        properties=(
            "label",
            "area",
            "perimeter",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "solidity",
            "eccentricity",
        ),
    )
    df = pd.DataFrame(props)
    if df.empty:
        summary = {
            f"{kind}_count": 0,
            f"{kind}_area_fraction": float(m.mean()),
            "calibrated": calibration.is_calibrated(),
            "pixel_size_um": calibration.pixel_size_um,
            "calibration_source": calibration.source,
        }
        return df, summary

    # Derived metrics
    per = df["perimeter"].replace(0, np.nan)
    df["circularity"] = 4.0 * math.pi * df["area"] / (per * per)
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"].replace(0, np.nan)

    # Keep original px columns
    df["area_px"] = df["area"]
    df["perimeter_px"] = df["perimeter"]
    df["equivalent_diameter_px"] = df["equivalent_diameter"]
    df["major_axis_length_px"] = df["major_axis_length"]
    df["minor_axis_length_px"] = df["minor_axis_length"]

    # Scale to physical units if calibrated
    if calibration.is_calibrated():
        px_um = calibration.pixel_size_um
        df["area_um2"] = df["area"] * (px_um * px_um)
        df["perimeter_um"] = df["perimeter"] * px_um
        df["equivalent_diameter_um"] = df["equivalent_diameter"] * px_um
        df["major_axis_length_um"] = df["major_axis_length"] * px_um
        df["minor_axis_length_um"] = df["minor_axis_length"] * px_um

    diam = df["equivalent_diameter_um"] if calibration.is_calibrated() else df["equivalent_diameter_px"]
    diam = diam.to_numpy(dtype=float)

    summary = {
        f"{kind}_count": int(df.shape[0]),
        f"{kind}_area_fraction": float(m.mean()),
        f"{kind}_diameter_mean": float(np.nanmean(diam)),
        f"{kind}_diameter_median": float(np.nanmedian(diam)),
        f"{kind}_diameter_std": float(np.nanstd(diam)),
        f"{kind}_diameter_p10": robust_percentile(diam, 10),
        f"{kind}_diameter_p90": robust_percentile(diam, 90),
        "calibrated": calibration.is_calibrated(),
        "pixel_size_um": calibration.pixel_size_um,
        "calibration_source": calibration.source,
    }
    return df, summary


def porosity_percent(pore_mask: np.ndarray) -> float:
    m = np.asarray(pore_mask, dtype=bool)
    return float(m.mean() * 100.0)


def fiber_diameter_distribution(
    fiber_mask: np.ndarray,
    calibration: Calibration,
    sample_stride_px: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray]:
    """Estimate fiber diameter distribution using skeleton + distance transform."""
    m = np.asarray(fiber_mask, dtype=bool)
    if m.sum() == 0:
        df = pd.DataFrame({"diameter_px": []})
        summary = {
            "fiber_skeleton_points": 0,
            "fiber_area_fraction": float(m.mean()),
            "calibrated": calibration.is_calibrated(),
            "pixel_size_um": calibration.pixel_size_um,
            "calibration_source": calibration.source,
        }
        skel = np.zeros_like(m, dtype=bool)
        dt = np.zeros_like(m, dtype=float)
        return df, summary, skel, dt

    dt = ndi.distance_transform_edt(m)
    skel = morphology.skeletonize(m)

    ys, xs = np.where(skel)
    if ys.size == 0:
        df = pd.DataFrame({"diameter_px": []})
        summary = {
            "fiber_skeleton_points": 0,
            "fiber_area_fraction": float(m.mean()),
            "calibrated": calibration.is_calibrated(),
            "pixel_size_um": calibration.pixel_size_um,
            "calibration_source": calibration.source,
        }
        return df, summary, skel, dt

    stride = max(1, int(sample_stride_px))
    if stride > 1:
        idx = np.arange(0, ys.size, stride)
        ys, xs = ys[idx], xs[idx]

    diam_px = 2.0 * dt[ys, xs].astype(np.float32)

    df = pd.DataFrame({"diameter_px": diam_px})
    if calibration.is_calibrated():
        df["diameter_um"] = df["diameter_px"] * calibration.pixel_size_um

    diam = df["diameter_um"].to_numpy() if calibration.is_calibrated() else df["diameter_px"].to_numpy()

    summary = {
        "fiber_skeleton_points": int(df.shape[0]),
        "fiber_area_fraction": float(m.mean()),
        "fiber_diameter_mean": float(np.nanmean(diam)),
        "fiber_diameter_median": float(np.nanmedian(diam)),
        "fiber_diameter_std": float(np.nanstd(diam)),
        "fiber_diameter_p10": robust_percentile(diam, 10),
        "fiber_diameter_p90": robust_percentile(diam, 90),
        "calibrated": calibration.is_calibrated(),
        "pixel_size_um": calibration.pixel_size_um,
        "calibration_source": calibration.source,
    }
    return df, summary, skel, dt
