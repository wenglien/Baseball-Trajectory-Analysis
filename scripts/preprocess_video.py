"""Preprocess video(s) to 120 FPS and 1080p using ffmpeg.

This script calls ffmpeg. It supports frame interpolation (smooth) or
simple duplication to reach 120 FPS, and scales/pads to 1920x1080 while
preserving aspect ratio.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import math


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install ffmpeg and ensure it's in PATH.", file=sys.stderr)
        sys.exit(1)


def build_filter(method: str, resolution: str) -> str:
    # Choose target dimensions based on resolution
    if resolution == "4k":
        target_w, target_h = 3840, 2160
    else:
        target_w, target_h = 1920, 1080

    # Use minterpolate for interpolation; otherwise simple fps duplication.
    scale_pad = f"scale={target_w}:-2,pad={target_w}:{target_h}:({target_w}-iw)/2:({target_h}-ih)/2"
    if method == "interpolate":
        return (
            "minterpolate=fps=120:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1," + scale_pad
        )
    return "fps=120," + scale_pad


def process_file(infile: str, outfile: str, method: str, crf: int, preset: str, resolution: str) -> None:
    vf = build_filter(method, resolution)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        infile,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        outfile,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        raise SystemExit(proc.returncode)


def preprocess_videos(
    inputs,
    output: str | None = None,
    method: str = "interpolate",
    resolution: str = "1080p",
    crf: int = 18,
    preset: str = "medium",
) -> list:
    """
    Preprocess one or more input files or directories.

    Args:
        inputs: single path or iterable of paths (file or directory)
        output: output file or directory (if multiple inputs, treated as directory)
        method: 'interpolate' or 'duplicate'
        resolution: '1080p' or '4k'
        crf: x264 CRF
        preset: x264 preset

    Returns:
        list of output paths (as strings) processed
    """
    check_ffmpeg()

    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    out_paths = []

    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            continue

        def needs_preprocessing_file(path: Path, resolution: str, target_fps: int = 120) -> bool:
            """Return True if the file at `path` needs preprocessing to reach target fps/resolution."""
            try:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return True
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
            except Exception:
                return True

            # fps tolerance: allow small float rounding
            if not math.isfinite(fps) or fps <= 0:
                return True
            fps_ok = abs(fps - target_fps) <= 0.5

            if resolution == "4k":
                res_ok = (width >= 3840 and height >= 2160)
            else:
                res_ok = (width >= 1920 and height >= 1080)

            return not (fps_ok and res_ok)

        if p.is_dir():
            suffix = "_120fps_4k" if resolution == "4k" else "_120fps_1080p"
            outdir = Path(output) if output else p / f"processed{suffix}"
            outdir.mkdir(parents=True, exist_ok=True)
            for f in p.iterdir():
                if f.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                    if needs_preprocessing_file(f, resolution):
                        out_file = outdir / (f.stem + suffix + ".mp4")
                        process_file(str(f), str(out_file), method, crf, preset, resolution)
                        out_paths.append(str(out_file))
                    else:
                        # skip processing and use original file
                        out_paths.append(str(f))
                        print(f"Skipping preprocessing for {f.name}: already {target_fps}fps/{resolution}")
        else:
            suffix = "_120fps_4k" if resolution == "4k" else "_120fps_1080p"
            if needs_preprocessing_file(p, resolution):
                if output and Path(output).is_dir():
                    out_file = Path(output) / (p.stem + suffix + ".mp4")
                else:
                    out_file = Path(output) if output else p.with_name(p.stem + suffix + ".mp4")
                process_file(str(p), str(out_file), method, crf, preset, resolution)
                out_paths.append(str(out_file))
            else:
                out_paths.append(str(p))
                print(f"Skipping preprocessing for {p.name}: already {120}fps/{resolution}")

    return out_paths


def main():
    parser = argparse.ArgumentParser(
        description="Increase FPS to 120 and scale to 1080p using ffmpeg."
    )
    parser.add_argument("input", help="input file or directory (or multiple, comma-separated)")
    parser.add_argument("-o", "--output", help="output file or directory")
    parser.add_argument(
        "--method",
        choices=["interpolate", "duplicate"],
        default="interpolate",
        help="use frame interpolation (smooth) or duplicate frames",
    )
    parser.add_argument(
        "--resolution",
        choices=["1080p", "4k"],
        default="1080p",
        help="target resolution: 1080p or 4k",
    )
    parser.add_argument("--crf", type=int, default=18, help="CRF for x264 (lower = better quality)")
    parser.add_argument("--preset", default="medium", help="x264 preset (e.g. fast, medium, slow)")
    args = parser.parse_args()

    # support comma-separated multiple inputs from CLI
    inputs = [s for s in args.input.split(",")]
    outputs = preprocess_videos(
        inputs, output=args.output, method=args.method, resolution=args.resolution, crf=args.crf, preset=args.preset
    )

    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()

