from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .defaults import load_default_config
from .pipeline import analyze
from .utils import load_yaml


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="YAML config file (optional)")
    p.add_argument("--mode", type=str, default=None, help="pores | particles | fibers (overrides config)")
    p.add_argument("--pixel-size", type=float, default=None, help="Pixel size in µm/px (overrides scale bar)")
    p.add_argument(
        "--scale-bar-length",
        type=float,
        default=None,
        help="Real length of the scale bar (numeric). Used with auto bar detection.",
    )
    p.add_argument("--scale-bar-unit", type=str, default="um", help="Scale bar unit: nm | um | mm")


def _load_config(path: Optional[str]):
    if path is None:
        return load_default_config()
    return load_yaml(path)


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze(
        args.image,
        out_dir=out_dir,
        config=cfg,
        mode=args.mode,
        pixel_size_um=args.pixel_size,
        scale_bar_length=args.scale_bar_length,
        scale_bar_unit=args.scale_bar_unit,
    )
    print("Done.")
    print(f"Summary written to: {out_dir / 'tables' / 'summary.csv'}")
    return 0


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def cmd_batch(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    in_dir = Path(args.folder)
    if not in_dir.exists():
        print(f"Input folder does not exist: {in_dir}", file=sys.stderr)
        return 2

    images = sorted([p for p in in_dir.rglob("*") if p.is_file() and _is_image(p)])
    if args.pattern:
        import fnmatch

        images = [p for p in images if fnmatch.fnmatch(p.name, args.pattern)]

    if not images:
        print("No images found.", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for img in tqdm(images, desc="Analyzing"):
        out_dir = out_root / img.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = analyze(
            img,
            out_dir=out_dir,
            config=cfg,
            mode=args.mode,
            pixel_size_um=args.pixel_size,
            scale_bar_length=args.scale_bar_length,
            scale_bar_unit=args.scale_bar_unit,
        )
        summaries.append(summary)

    # Aggregate summary across images
    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(out_root / "summary_all.csv", index=False)

    print("Batch complete.")
    print(f"Results: {out_root}")
    if summaries:
        print(f"Aggregated summary: {out_root / 'summary_all.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semquant", description="Automated SEM image quantification.")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("analyze", help="Analyze a single SEM image.")
    p1.add_argument("image", type=str, help="Path to an image file")
    p1.add_argument("--out", type=str, required=True, help="Output directory")
    _add_common_args(p1)
    p1.set_defaults(func=cmd_analyze)

    p2 = sub.add_parser("batch", help="Analyze all images in a folder (recursive).")
    p2.add_argument("folder", type=str, help="Folder containing images")
    p2.add_argument("--out", type=str, required=True, help="Output root directory")
    p2.add_argument("--pattern", type=str, default=None, help="Optional filename pattern, e.g. '*.tif'")
    _add_common_args(p2)
    p2.set_defaults(func=cmd_batch)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
