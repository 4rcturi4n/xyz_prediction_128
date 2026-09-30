# scripts/00_make_paths_relative.py
#
# One-off fix: video_path (and xl_path) in video_xy_dataset.csv,
# video_z_dataset.csv, and every kfold_splits_*/fold_N_{train,val}.csv
# were baked in as absolute Windows paths into the OTHER repo
# (C:\Users\michris\Desktop\xy_early_prediction\...), since these CSVs
# were originally copied from there. The video files now live inside
# THIS repo's own data/Early prediction/ folder (including on a Linux
# server, where backslash Windows paths don't resolve at all), so we
# rewrite both columns to be forward-slash paths relative to repo root.
#
# Safe to re-run: no-ops on rows already rewritten.

import os
import glob

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_PREFIX = "C:\\Users\\michris\\Desktop\\xy_early_prediction\\"


def make_relative(path: str) -> str:
    if not isinstance(path, str):
        return path
    if path.startswith(OLD_PREFIX):
        path = path[len(OLD_PREFIX):]
    return path.replace("\\", "/")


def fix_csv(path: str):
    df = pd.read_csv(path)
    changed = False
    for col in ("video_path", "xl_path"):
        if col in df.columns:
            new_col = df[col].apply(make_relative)
            if not new_col.equals(df[col]):
                df[col] = new_col
                changed = True
    if changed:
        df.to_csv(path, index=False)
        print(f"rewrote {path}")


def main():
    targets = [
        os.path.join(ROOT, "data", "processed", "video_xy_dataset.csv"),
        os.path.join(ROOT, "data", "processed", "video_z_dataset.csv"),
    ]
    targets += glob.glob(os.path.join(ROOT, "data", "processed", "kfold_splits_*", "fold_*_*.csv"))

    for path in targets:
        fix_csv(path)
    print(f"\nChecked {len(targets)} CSV files.")


if __name__ == "__main__":
    main()
