#!/usr/bin/env python3
"""
Render a vertical pressed-cell bar from a live Pokemon-app dump CSV.

The bar is the same quantity the backend uses to fire a capture: how many
taxels are at or below PRESSURE_THRESHOLD. The threshold line is at
PRESSED_CELL_MIN (4). The bar turns accent-colored when that line is crossed.

Usage:
    venv/bin/python viz_pressure_bar.py data/pokemon_pinch_data_2.csv
    venv/bin/python viz_pressure_bar.py data/pokemon_hwi_data_2.csv --output hwi_bar.mp4
"""

from pathlib import Path
import argparse
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

# Same rule as streamHandposeAndGlove.py
NUM_SENSORS = 16 * 16
PRESSURE_THRESHOLD = 400.0
PRESSED_CELL_MIN = 4

FPS = 30
WIDTH = 360
HEIGHT = 720
BAR_MAX = 12  # headroom so the threshold line sits mid-bar, not at the top

BG = (18, 18, 18)
TRACK = (42, 42, 42)
FILL_BELOW = (60, 60, 240)     # red in BGR
FILL_ABOVE = (80, 200, 80)     # green in BGR when threshold is reached
THRESHOLD_LINE = (240, 240, 240)
TEXT = (230, 230, 230)
MUTED = (140, 140, 140)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a pressed-cell threshold bar from a dump CSV."
    )
    parser.add_argument("csv", type=Path, help="Pokemon dump CSV (s_0..s_255, pc_ts)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 (default: <csv_stem>_pressure_bar.mp4)",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def pressed_cells(sensor_row: np.ndarray) -> int:
    valid = sensor_row[sensor_row > 0]
    return int(np.sum(valid <= PRESSURE_THRESHOLD))


def load_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    missing = [c for c in ("pc_ts",) + tuple(f"s_{i}" for i in range(NUM_SENSORS)) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing[:6]}")
    ts = df["pc_ts"].to_numpy(dtype=float)
    sensors = df[[f"s_{i}" for i in range(NUM_SENSORS)]].to_numpy(dtype=float)
    counts = np.array([pressed_cells(row) for row in sensors], dtype=float)
    return ts, counts


def sample_at(ts: np.ndarray, values: np.ndarray, t: float) -> float:
    if t <= ts[0]:
        return float(values[0])
    if t >= ts[-1]:
        return float(values[-1])
    i = int(np.searchsorted(ts, t))
    t0, t1 = ts[i - 1], ts[i]
    if t1 <= t0:
        return float(values[i])
    alpha = (t - t0) / (t1 - t0)
    return float((1 - alpha) * values[i - 1] + alpha * values[i])


def put(img, text, xy, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(
        img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def draw_frame(n_pressed: float, t_rel: float) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    crossed = n_pressed >= PRESSED_CELL_MIN
    fill = FILL_ABOVE if crossed else FILL_BELOW
    n_show = int(round(n_pressed))

    put(img, "Tactile Pressure Trigger", (28, 48), 0.7, TEXT, 2)
    put(img, f"{n_show} / {PRESSED_CELL_MIN}", (28, 108), 1.4, fill, 3)

    top = 150
    bottom = HEIGHT - 50
    left = 40
    right = 160
    track_h = bottom - top

    cv2.rectangle(img, (left, top), (right, bottom), TRACK, thickness=-1)

    frac = np.clip(n_pressed / BAR_MAX, 0.0, 1.0)
    fill_top = int(bottom - frac * track_h)
    cv2.rectangle(img, (left, fill_top), (right, bottom), fill, thickness=-1)

    thresh_y = int(bottom - (PRESSED_CELL_MIN / BAR_MAX) * track_h)
    cv2.line(img, (left - 8, thresh_y), (right + 16, thresh_y), THRESHOLD_LINE, 3)
    put(img, "threshold", (right + 28, thresh_y - 4), 0.5, THRESHOLD_LINE, 1)
    put(
        img,
        f"(ADC <= {int(PRESSURE_THRESHOLD)})",
        (right + 28, thresh_y + 22),
        0.45,
        THRESHOLD_LINE,
        1,
    )
    return img


def main():
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    out_path = args.output or Path(f"{csv_path.stem}_pressure_bar.mp4")

    ts, counts = load_series(csv_path)
    duration = max(ts[-1] - ts[0], 1.0 / args.fps)
    n_frames = int(np.floor(duration * args.fps)) + 1
    n_cross = int(np.sum(counts >= PRESSED_CELL_MIN))
    print(
        f"{csv_path.name}: {len(ts)} samples, {duration:.1f}s, "
        f"max pressed={counts.max():.0f}, frames with n>={PRESSED_CELL_MIN}={n_cross}"
    )

    # H.264 + yuv420p so Cursor, QuickTime, and browsers can preview the file.
    # OpenCV's default mp4v (MPEG-4 Visual) often fails to play in the editor.
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it to write H.264 mp4s.")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(args.fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n_frames):
            t = ts[0] + i / args.fps
            n = sample_at(ts, counts, t)
            proc.stdin.write(draw_frame(n, t - ts[0]).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed with code {rc} while writing {out_path}")
    print(f"Wrote {out_path} ({n_frames} frames @ {args.fps} fps, H.264)")


if __name__ == "__main__":
    main()
