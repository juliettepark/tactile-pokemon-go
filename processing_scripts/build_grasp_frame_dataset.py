#!/usr/bin/env python3
"""
Build a single-frame grasp dataset from live Pokemon-app dumps.

Writes right-hand bone columns (get_right_bone_headers) plus label to:
    data/grasp_classifier/grasp_frames.csv

Drops the first and last 10% of each source file.
"""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import argparse
import pandas as pd

from tactile_utils.tactile_handpose_utils import get_right_bone_headers

RIGHT_BONE_HEADERS = get_right_bone_headers()

DEFAULT_SOURCES = {
    "precision_pinch": _project_root / "data" / "pokemon_pinch_data_2.csv",
    "heavy_wrap": _project_root / "data" / "pokemon_hwi_data_2.csv",
}

OUTPUT_CSV = _project_root / "data" / "grasp_classifier" / "grasp_frames.csv"
END_TRIM_FRACTION = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description="Build right-hand grasp frame dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    return parser.parse_args()


def middle_slice(n_rows: int, trim_fraction: float = END_TRIM_FRACTION):
    if n_rows <= 0:
        return 0, 0
    start = int(n_rows * trim_fraction)
    stop = int(n_rows * (1.0 - trim_fraction))
    if stop <= start:
        return 0, n_rows
    return start, stop


def frames_from_file(path: Path, label: str):
    if not path.exists():
        print(f"[skip] missing file: {path}")
        return []

    df = pd.read_csv(path)
    missing = [c for c in RIGHT_BONE_HEADERS if c not in df.columns]
    if missing:
        print(f"[skip] missing right-hand bone columns in {path}: {missing[:4]}...")
        return []

    start, stop = middle_slice(len(df))
    mid = df.iloc[start:stop]
    out = mid[RIGHT_BONE_HEADERS].copy()
    out["label"] = label
    out["source_file"] = str(path.relative_to(_project_root))
    print(f"{label}: {path.name} {len(df)} raw -> {len(out)} kept (trimmed [{start}:{stop}])")
    return out.to_dict("records")


def main():
    args = parse_args()
    all_rows = []
    for label, csv_path in DEFAULT_SOURCES.items():
        all_rows.extend(frames_from_file(csv_path, label))

    if not all_rows:
        raise SystemExit("No frames extracted. Check that the live dump CSVs exist.")

    out_df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    print(out_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
