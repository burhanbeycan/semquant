from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .io import read_image, save_png
from .measurements import fiber_diameter_distribution, porosity_percent, regionprops_dataframe
from .plotting import save_cdf, save_histogram, save_image, save_labeled, save_overlay
from .preprocess import preprocess
from .scale import detect_scale_bar
from .segmentation import clean_binary_mask, exclude_bottom, threshold_mask
from .utils import Calibration, ensure_dir, load_yaml, save_json, save_yaml, unit_to_um_factor, versions_info


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_out_dir(out_dir: str | Path, image_path: str | Path) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and out_dir.is_file():
        raise ValueError("out_dir must be a directory")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _compute_calibration(
    img: np.ndarray,
    config: Dict[str, Any],
    pixel_size_um: Optional[float] = None,
    scale_bar_length: Optional[float] = None,
    scale_bar_unit: str = "um",
) -> Tuple[Calibration, Dict[str, Any]]:
    """Compute pixel calibration from direct pixel size or detected scale bar."""
    scale_cfg = config.get("scale", {})
    direct = scale_cfg.get("pixel_size_um_per_px", None)
    if pixel_size_um is not None:
        direct = pixel_size_um

    if direct is not None:
        try:
            direct = float(direct)
            if not (direct > 0):
                raise ValueError
            return Calibration(pixel_size_um=direct, source="pixel_size"), {"scale_bar_detected": False}
        except Exception:
            pass

    sb_cfg = scale_cfg.get("scale_bar", {})
    enabled = bool(sb_cfg.get("enabled", True))
    if not enabled:
        return Calibration(pixel_size_um=None, source="uncalibrated"), {"scale_bar_detected": False}

    known_length = sb_cfg.get("known_length", None)
    unit = sb_cfg.get("unit", "um")
    if scale_bar_length is not None:
        known_length = scale_bar_length
    if scale_bar_unit is not None:
        unit = scale_bar_unit

    info: Dict[str, Any] = {"scale_bar_detected": False}

    if known_length is None:
        return Calibration(pixel_size_um=None, source="uncalibrated"), info

    det = detect_scale_bar(
        img,
        search_bottom_fraction=float(sb_cfg.get("search_bottom_fraction", 0.22)),
        min_aspect_ratio=float(sb_cfg.get("min_aspect_ratio", 8.0)),
        min_area_px=int(sb_cfg.get("min_area_px", 250)),
    )
    if det is None or det.bar_length_px <= 0:
        return Calibration(pixel_size_um=None, source="uncalibrated"), info

    try:
        known_um = float(known_length) * unit_to_um_factor(str(unit))
        px_um = known_um / float(det.bar_length_px)
        info.update(
            {
                "scale_bar_detected": True,
                "scale_bar_length_px": int(det.bar_length_px),
                "scale_bar_bbox": det.bbox,
                "scale_bar_polarity": det.polarity,
                "scale_bar_known_length": float(known_length),
                "scale_bar_unit": str(unit),
            }
        )
        return Calibration(pixel_size_um=float(px_um), source="scale_bar_detected"), info
    except Exception:
        return Calibration(pixel_size_um=None, source="uncalibrated"), info


def analyze(
    image_path: str | Path,
    out_dir: str | Path,
    config: Dict[str, Any] | str | Path,
    *,
    mode: Optional[str] = None,
    pixel_size_um: Optional[float] = None,
    scale_bar_length: Optional[float] = None,
    scale_bar_unit: str = "um",
) -> Dict[str, Any]:
    """Analyze one SEM image and write figures/tables to out_dir.

    Returns:
        summary dict (also written to tables/summary.csv and run_info/summary.json).
    """
    image_path = Path(image_path)
    out_dir = _resolve_out_dir(out_dir, image_path)

    if isinstance(config, (str, Path)):
        config_dict = load_yaml(config)
    else:
        config_dict = dict(config)

    analysis_cfg = config_dict.get("analysis", {})
    cleanup_cfg = config_dict.get("cleanup", {})
    output_cfg = config_dict.get("output", {})

    if mode is None:
        mode = str(analysis_cfg.get("mode", "pores")).lower()
    mode = str(mode).lower()
    if mode not in {"pores", "particles", "fibers"}:
        raise ValueError("mode must be one of: pores, particles, fibers")

    dpi = int(output_cfg.get("dpi", 300))

    # Create output folders
    figs_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")
    masks_dir = ensure_dir(out_dir / "masks")
    report_dir = ensure_dir(out_dir / "report")
    runinfo_dir = ensure_dir(out_dir / "run_info")

    # Load image
    img = read_image(image_path)

    # Calibration
    calib, calib_info = _compute_calibration(
        img, config_dict, pixel_size_um=pixel_size_um, scale_bar_length=scale_bar_length, scale_bar_unit=scale_bar_unit
    )

    # Crop bottom (exclude scale bar / legend)
    excl_frac = float(analysis_cfg.get("exclude_bottom_fraction", 0.12))
    img_roi, bbox_roi = exclude_bottom(img, excl_frac)

    # If scale bar bbox is known, crop above it as well (more precise than a fixed fraction)
    if calib_info.get("scale_bar_detected") and "scale_bar_bbox" in calib_info:
        minr, minc, maxr, maxc = calib_info["scale_bar_bbox"]
        # crop to just above the detected bar (with a small margin)
        margin = 3
        maxr2 = max(1, int(minr - margin))
        if maxr2 < img_roi.shape[0]:
            img_roi = img[:maxr2, :]

    # Preprocess
    img_prep = preprocess(img_roi, gaussian_sigma=float(analysis_cfg.get("gaussian_sigma", 1.0)), use_clahe=True)

    # Threshold → binary mask
    thr_method = str(analysis_cfg.get("threshold_method", "otsu")).lower()
    mask = threshold_mask(img_prep, method=thr_method)  # True=bright regions by default

    invert_flag = bool(analysis_cfg.get("invert", False))
    if invert_flag:
        mask = ~mask
    else:
        # heuristic: pores are typically dark → after threshold, mask often becomes "solid".
        if mode == "pores" and float(mask.mean()) > 0.5:
            mask = ~mask

    # Clean mask
    mask = clean_binary_mask(
        mask,
        min_object_area_px=int(cleanup_cfg.get("min_object_area_px", 80)),
        remove_border_objects=bool(cleanup_cfg.get("remove_border_objects", True)),
    )

    # Save base images
    save_image(figs_dir / "01_original.png", img_roi, dpi=dpi)
    save_png(masks_dir / "mask.png", mask)
    save_image(figs_dir / "02_mask.png", mask.astype(np.float32), dpi=dpi)
    save_overlay(figs_dir / "03_overlay.png", img_roi, mask, dpi=dpi)
    labeled = save_labeled(figs_dir / "04_labeled.png", img_roi, mask, dpi=dpi)

    # Measurements
    summary: Dict[str, Any] = {
        "image": str(image_path.name),
        "mode": mode,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "roi_height_px": int(img_roi.shape[0]),
        "roi_width_px": int(img_roi.shape[1]),
        "roi_area_px": int(img_roi.shape[0] * img_roi.shape[1]),
        "calibrated": calib.is_calibrated(),
        "pixel_size_um": calib.pixel_size_um,
        "calibration_source": calib.source,
    }
    summary.update(calib_info)

    # Physical ROI area if calibrated
    if calib.is_calibrated():
        area_um2 = float(summary["roi_area_px"]) * (calib.pixel_size_um ** 2)
        summary["roi_area_um2"] = area_um2
        summary["roi_area_mm2"] = area_um2 / 1e6

    measurements_df = pd.DataFrame()
    if mode in {"pores", "particles"}:
        kind = "pore" if mode == "pores" else "particle"
        measurements_df, meas_summary = regionprops_dataframe(mask, calib, kind=kind)
        summary.update(meas_summary)

        if mode == "pores":
            summary["porosity_percent"] = porosity_percent(mask)

        # Density (# / mm^2) if calibrated
        if calib.is_calibrated() and summary.get("roi_area_mm2", 0) > 0:
            density = float(meas_summary.get(f"{kind}_count", 0)) / float(summary["roi_area_mm2"])
            summary[f"{kind}_density_per_mm2"] = density

        # Save tables
        measurements_csv = tables_dir / "measurements.csv"
        measurements_df.to_csv(measurements_csv, index=False)

        # Plot distributions (equivalent diameter)
        diam = (
            measurements_df.get("equivalent_diameter_um")
            if calib.is_calibrated()
            else measurements_df.get("equivalent_diameter_px")
        )
        if diam is not None and len(diam) > 0:
            xlabel = "Equivalent diameter (µm)" if calib.is_calibrated() else "Equivalent diameter (px)"
            save_histogram(figs_dir / "hist_diameter.png", diam.to_numpy(), f"{kind.title()} diameter", xlabel, dpi=dpi)
            save_cdf(figs_dir / "cdf_diameter.png", diam.to_numpy(), f"{kind.title()} diameter CDF", xlabel, dpi=dpi)

    elif mode == "fibers":
        df, meas_summary, skel, dt = fiber_diameter_distribution(mask, calib, sample_stride_px=1)
        summary.update(meas_summary)

        # Save fiber samples
        df.to_csv(tables_dir / "measurements.csv", index=False)

        diam = df["diameter_um"] if calib.is_calibrated() and "diameter_um" in df else df["diameter_px"]
        xlabel = "Fiber diameter (µm)" if calib.is_calibrated() else "Fiber diameter (px)"
        save_histogram(figs_dir / "hist_diameter.png", diam.to_numpy(), "Fiber diameter", xlabel, dpi=dpi)
        save_cdf(figs_dir / "cdf_diameter.png", diam.to_numpy(), "Fiber diameter CDF", xlabel, dpi=dpi)

        # Save skeleton visualization
        save_image(figs_dir / "05_skeleton.png", skel.astype(np.float32), dpi=dpi)

    # Save summary table
    pd.DataFrame([summary]).to_csv(tables_dir / "summary.csv", index=False)
    save_json(summary, runinfo_dir / "summary.json")

    # Save config and versions for reproducibility
    save_yaml(config_dict, runinfo_dir / "config_used.yaml")
    save_json(versions_info(), runinfo_dir / "versions.json")

    # Markdown report
    report_md = report_dir / "report.md"
    _write_markdown_report(report_md, summary)

    # Optional PDF report
    if bool(output_cfg.get("make_pdf_report", True)):
        try:
            from .report import build_pdf_report

            key_figs = [
                figs_dir / "03_overlay.png",
                figs_dir / "hist_diameter.png",
            ]
            build_pdf_report(
                report_dir / "report.pdf",
                title=f"SEMQuant report — {image_path.name}",
                summary=summary,
                key_figures=[str(p) for p in key_figs],
            )
        except Exception as e:
            # Don't fail the whole pipeline if PDF creation fails.
            summary["pdf_report_error"] = str(e)

    return summary


def _write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append(f"# SEMQuant report — {summary.get('image','')}")
    lines.append("")
    lines.append(f"- Mode: **{summary.get('mode','')}**")
    lines.append(f"- Timestamp: {summary.get('timestamp','')}")
    lines.append(f"- Calibrated: **{summary.get('calibrated')}** (pixel_size_um={summary.get('pixel_size_um')})")
    if summary.get("scale_bar_detected"):
        lines.append(
            f"- Scale bar: detected, {summary.get('scale_bar_length_px')} px = "
            f"{summary.get('scale_bar_known_length')} {summary.get('scale_bar_unit')}"
        )
    lines.append("")
    lines.append("## Key metrics")
    lines.append("")
    for k in sorted(summary.keys()):
        if k in {"image", "mode", "timestamp"}:
            continue
        lines.append(f"- **{k}**: {summary[k]}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- Figures: `figures/`")
    lines.append("- Tables: `tables/`")
    lines.append("- Mask: `masks/mask.png`")
    lines.append("- Run info: `run_info/` (config + versions)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
