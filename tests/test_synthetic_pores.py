from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from skimage import draw, util
from skimage.io import imsave

from semquant.pipeline import analyze


def make_grid_pores(h=256, w=256, r=10, step=48):
    img = np.ones((h, w), dtype=np.float32) * 0.9
    mask = np.zeros((h, w), dtype=bool)
    ys = list(range(step // 2, h - step // 2, step))
    xs = list(range(step // 2, w - step // 2, step))
    for y in ys:
        for x in xs:
            rr, cc = draw.disk((y, x), r, shape=img.shape)
            mask[rr, cc] = True
            img[rr, cc] = 0.1
    img = util.random_noise(img, mode="gaussian", var=0.001)
    img = np.clip(img, 0, 1)
    return img, mask


def test_porosity_close_to_ground_truth():
    img, gt_mask = make_grid_pores()
    expected = gt_mask.mean() * 100.0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = td / "pores.png"
        imsave(str(img_path), (img * 255).astype(np.uint8), check_contrast=False)

        cfg = {
            "analysis": {
                "mode": "pores",
                "threshold_method": "otsu",
                "invert": False,
                "gaussian_sigma": 0.0,
                "exclude_bottom_fraction": 0.0,
            },
            "cleanup": {"min_object_area_px": 10, "remove_border_objects": False},
            "scale": {"pixel_size_um_per_px": 0.01, "scale_bar": {"enabled": False}},
            "output": {"dpi": 120, "make_pdf_report": False, "save_intermediate": True},
        }

        out_dir = td / "out"
        summary = analyze(img_path, out_dir=out_dir, config=cfg, mode="pores", pixel_size_um=0.01)

        got = float(summary["porosity_percent"])
        assert abs(got - expected) < 4.0
