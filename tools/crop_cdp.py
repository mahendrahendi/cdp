#!/usr/bin/env python3
"""
Crop CDP region from phone captures by finding the largest dense dark region.
Uses Otsu thresholding + morphological closing + connected components.

Usage:
  python tools/crop_cdp.py --input /path/to/image.png --output /path/to/crop.png
  python tools/crop_cdp.py --input_dir /path/to/images --output_dir /path/to/crops
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage import io, color, filters, morphology, measure, util


def find_cdp_bbox(image: np.ndarray, min_area_frac: float = 0.002) -> tuple[int, int, int, int]:
    """Return bbox (min_row, min_col, max_row, max_col) of the CDP-like region."""
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0

    # Dark regions (ink) should be True
    thresh = filters.threshold_otsu(gray)
    dark = gray < thresh

    # Close small gaps to make the CDP a single component
    dark = morphology.closing(dark, morphology.square(5))

    # Remove tiny specks
    min_area = int(min_area_frac * dark.shape[0] * dark.shape[1])
    dark = morphology.remove_small_objects(dark, min_size=max(10, min_area))

    labeled = measure.label(dark)
    props = measure.regionprops(labeled)

    if not props:
        raise ValueError("No foreground region found; check input image quality.")

    # Prefer large, roughly square regions
    best = None
    best_score = -1.0
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        h = maxr - minr
        w = maxc - minc
        if h == 0 or w == 0:
            continue
        aspect = w / h
        area = p.area
        square_penalty = abs(1.0 - aspect)
        score = area / (1.0 + square_penalty)
        if score > best_score:
            best_score = score
            best = p

    if best is None:
        best = max(props, key=lambda p: p.area)

    return best.bbox


def expand_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin_frac: float) -> tuple[int, int, int, int]:
    minr, minc, maxr, maxc = bbox
    h = maxr - minr
    w = maxc - minc
    margin_h = int(h * margin_frac)
    margin_w = int(w * margin_frac)
    minr = max(0, minr - margin_h)
    minc = max(0, minc - margin_w)
    maxr = min(shape[0], maxr + margin_h)
    maxc = min(shape[1], maxc + margin_w)
    return minr, minc, maxr, maxc


def crop_one(input_path: Path, output_path: Path, margin: float) -> None:
    image = io.imread(str(input_path))
    bbox = find_cdp_bbox(image)
    bbox = expand_bbox(bbox, image.shape[:2], margin)
    minr, minc, maxr, maxc = bbox
    cropped = image[minr:maxr, minc:maxc]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(output_path), util.img_as_ubyte(cropped))


def iter_images(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop CDP region from phone captures.")
    parser.add_argument("--input", type=str, help="Path to one input image.")
    parser.add_argument("--output", type=str, help="Path to output image.")
    parser.add_argument("--input_dir", type=str, help="Directory of input images.")
    parser.add_argument("--output_dir", type=str, help="Directory to write cropped images.")
    parser.add_argument("--margin", type=float, default=0.05, help="Extra margin around crop (fraction).")
    args = parser.parse_args()

    if args.input and args.output:
        crop_one(Path(args.input), Path(args.output), args.margin)
        return

    if args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        for src in iter_images(input_dir):
            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            crop_one(src, dst, args.margin)
        return

    raise SystemExit("Provide either --input/--output or --input_dir/--output_dir.")


if __name__ == "__main__":
    main()
