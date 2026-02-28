from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from skimage.io import imsave

from semquant.pipeline import analyze


def make_stripe(h=256, w=256, thickness=20):
    img = np.ones((h, w), dtype=np.float32) * 0.1
    y0 = h // 2 - thickness // 2
    y1 = y0 + thickness
    img[y0:y1, :] = 0.9
    return img, thickness


def test_fiber_diameter_close_to_thickness_px():
    img, thickness = make_stripe(thickness=22)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = td / "fibers.png"
        imsave(str(img_path), (img * 255).astype(np.uint8), check_contrast=False)

        cfg = {
            "analysis": {
                "mode": "fibers",
                "threshold_method": "otsu",
                "invert": False,
                "gaussian_sigma": 0.0,
                "exclude_bottom_fraction": 0.0,
            },
            "cleanup": {"min_object_area_px": 10, "remove_border_objects": False},
            "scale": {"pixel_size_um_per_px": None, "scale_bar": {"enabled": False}},
            "output": {"dpi": 120, "make_pdf_report": False, "save_intermediate": True},
        }

        out_dir = td / "out"
        summary = analyze(img_path, out_dir=out_dir, config=cfg, mode="fibers", pixel_size_um=None)

        # Expected in pixels since uncalibrated
        got = float(summary["fiber_diameter_median"])
        assert abs(got - thickness) < 3.0
