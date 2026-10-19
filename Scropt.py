
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def restore_image(rgb, mode, factor, gamma, clip_limit):
    """Input and output are RGB uint8 arrays."""
    x = rgb.astype(np.float32) / 255.0

    if mode == "scale":
        if factor <= 0:
            raise ValueError("factor must be positive")
        # Inverse of a known multiplicative brightness reduction.
        y = np.clip(x / factor, 0.0, 1.0)

    elif mode == "gamma":
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        y = np.power(x, gamma)

    elif mode == "clahe":
        # Apply CLAHE to luminance, not independently to RGB.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        luminance, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(8, 8)
        )
        luminance = clahe.apply(luminance)
        result = cv2.merge((luminance, a, b))
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result

    elif mode == "retinex":
        # Simple single-scale Retinex-like baseline.
        # Not a full multiscale or learned Retinex model.
        illumination = cv2.GaussianBlur(
            x, (0, 0), sigmaX=30, sigmaY=30
        )
        illumination = np.maximum(illumination, 1.0 / 255.0)

        y = x / illumination

        # Robust normalization avoids a few extreme pixels
        # dominating the output. This changes global tone.
        lo, hi = np.percentile(y, (1, 99.5))
        if hi > lo:
            y = (y - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.round(y * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["scale", "gamma", "clahe", "retinex"],
        default="scale"
    )
    parser.add_argument(
        "--factor", type=float, default=0.25,
        help="Known original-to-degraded brightness multiplier"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.6
    )
    parser.add_argument(
        "--clip-limit", type=float, default=2.0
    )
    parser.add_argument(
        "--recursive", action="store_true"
    )
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise NotADirectoryError(src)

    # Avoid accidentally processing the output folder as input.
    if dst == src or src in dst.parents:
        raise ValueError(
            "Output must be outside the input directory."
        )

    files = (
        src.rglob("*") if args.recursive else src.iterdir()
    )
    files = sorted(
        p for p in files
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not files:
        print("No supported images found.")
        return

    dst.mkdir(parents=True, exist_ok=True)
    report_path = dst / "processing_report.csv"
    successes = 0

    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as report:
        writer = csv.DictWriter(
            report,
            fieldnames=["file", "status", "width", "height", "error"]
        )
        writer.writeheader()

        for path in tqdm(files, desc="Restoring"):
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    rgb = np.asarray(im.convert("RGB"))

                restored = restore_image(
                    rgb,
                    mode=args.mode,
                    factor=args.factor,
                    gamma=args.gamma,
                    clip_limit=args.clip_limit
                )

                # Preserve subfolder structure in recursive mode.
                rel = path.relative_to(src)
                out_path = dst / rel
                out_path = out_path.with_suffix(".png")
                out_path.parent.mkdir(
                    parents=True, exist_ok=True
                )

                # PNG avoids introducing another lossy JPEG encode.
                Image.fromarray(restored).save(out_path)

                writer.writerow({
                    "file": str(rel),
                    "status": "ok",
                    "width": restored.shape[1],
                    "height": restored.shape[0],
                    "error": ""
                })
                successes += 1

            except Exception as e:
                writer.writerow({
                    "file": str(path),
                    "status": "error",
                    "width": "",
                    "height": "",
                    "error": str(e)
                })
                print(f"\nFailed: {path}: {e}")

    print(f"Restored: {successes}/{len(files)}")
    print(f"Output: {dst}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
