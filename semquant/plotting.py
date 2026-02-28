from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color, measure, segmentation

# Headless backend for servers/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_image(path: str | Path, img: np.ndarray, dpi: int = 300, cmap: str = "gray") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(p), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(path: str | Path, img: np.ndarray, mask: np.ndarray, dpi: int = 300) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    overlay = segmentation.mark_boundaries(img, mask.astype(bool), mode="thick")
    plt.figure()
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(p), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_labeled(path: str | Path, img: np.ndarray, mask: np.ndarray, dpi: int = 300) -> np.ndarray:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lab = measure.label(mask.astype(bool))
    rgb = color.label2rgb(lab, image=img, bg_label=0, alpha=0.35)
    plt.figure()
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(p), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    return lab


def save_histogram(
    path: str | Path,
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 30,
    dpi: int = 300,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    plt.figure()
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(str(p), dpi=dpi)
    plt.close()


def save_cdf(
    path: str | Path,
    values: np.ndarray,
    title: str,
    xlabel: str,
    dpi: int = 300,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if v.size == 0:
        v = np.array([0.0])
    xs = np.sort(v)
    ys = np.linspace(0, 1, xs.size, endpoint=True)
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(str(p), dpi=dpi)
    plt.close()
