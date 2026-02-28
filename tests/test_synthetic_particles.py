from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from skimage import draw
from skimage.io import imsave

from semquant.pipeline import analyze


def make_particles(h=256, w=256, r=8, step=40):
    img = np.ones((h, w), dtype=np.float32) * 0.1
    ys = list(range(step // 2, h - step // 2, step))
    xs = list(range(step // 2, w - step // 2, step))
    count = 0
    for y in ys:
        for x in xs:
            rr, cc = draw.disk((y, x), r, shape=img.shape)
            img[rr, cc] = 0.9
            count += 1
    return img, count


def test_particle_count_matches_grid():
    img, expected_count = make_particles()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = td / "particles.png"
        imsave(str(img_path), (img * 255).astype(np.uint8), check_contrast=False)

        cfg = {
            "analysis": {
                "mode": "particles",
                "threshold_method": "otsu",
                "invert": False,
                "gaussian_sigma": 0.0,
                "exclude_bottom_fraction": 0.0,
            },
            "cleanup": {"min_object_area_px": 10, "remove_border_objects": False},
            "scale": {"pixel_size_um_per_px": 0.02, "scale_bar": {"enabled": False}},
            "output": {"dpi": 120, "make_pdf_report": False, "save_intermediate": True},
        }

        out_dir = td / "out"
        summary = analyze(img_path, out_dir=out_dir, config=cfg, mode="particles", pixel_size_um=0.02)

        got = int(summary["particle_count"])
        assert got == expected_count
