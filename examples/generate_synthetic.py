from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage import draw, filters, util
from skimage.io import imsave


def synthetic_pores(h: int = 512, w: int = 512, n: int = 80, r_range=(6, 25), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.ones((h, w), dtype=np.float32) * 0.85
    for _ in range(n):
        r = int(rng.integers(r_range[0], r_range[1]))
        cy = int(rng.integers(r, h - r))
        cx = int(rng.integers(r, w - r))
        rr, cc = draw.disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = 0.15  # dark pores
    img = filters.gaussian(img, sigma=1.0)
    img = util.random_noise(img, mode="gaussian", var=0.002)
    return np.clip(img, 0, 1)


def synthetic_particles(h: int = 512, w: int = 512, n: int = 60, r_range=(6, 20), seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.ones((h, w), dtype=np.float32) * 0.15
    for _ in range(n):
        r = int(rng.integers(r_range[0], r_range[1]))
        cy = int(rng.integers(r, h - r))
        cx = int(rng.integers(r, w - r))
        rr, cc = draw.disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = 0.85  # bright particles
    img = filters.gaussian(img, sigma=1.0)
    img = util.random_noise(img, mode="gaussian", var=0.002)
    return np.clip(img, 0, 1)


def synthetic_fibers(h: int = 512, w: int = 512, n: int = 25, thickness: int = 10, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.ones((h, w), dtype=np.float32) * 0.2
    for _ in range(n):
        y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
        y1, x1 = int(rng.integers(0, h)), int(rng.integers(0, w))
        rr, cc = draw.line(y0, x0, y1, x1)
        for t in range(-thickness // 2, thickness // 2 + 1):
            rr2 = np.clip(rr + t, 0, h - 1)
            cc2 = np.clip(cc + t, 0, w - 1)
            img[rr2, cc2] = 0.85
    img = filters.gaussian(img, sigma=1.2)
    img = util.random_noise(img, mode="gaussian", var=0.003)
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pores", "particles", "fibers"], default="pores")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "pores":
        img = synthetic_pores(seed=args.seed)
    elif args.mode == "particles":
        img = synthetic_particles(seed=args.seed)
    else:
        img = synthetic_fibers(seed=args.seed)

    imsave(str(out), (img * 255).astype(np.uint8), check_contrast=False)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
