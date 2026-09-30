# scripts/02b_build_loo_batch_splits.py
#
# Leave-one-batch-out splits: each "batch" is one recording session (the
# video_id prefix, e.g. #250617Video(14-21)#, @250624Video@) -- 6 distinct
# batches in the 106-video set. Fold N holds out batch N entirely as
# validation, trains on the other 5 batches. Alongside (not instead of)
# the stratified 5-fold CV in kfold_splits_xy / kfold_splits_z.
#
# Run after kfold_splits_xy / kfold_splits_z exist (needs the 106-video
# xy dataset with fold membership matching z, and the z dataset).

import os
import re
import json

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XY_CSV = os.path.join(ROOT, "data", "processed", "video_xy_dataset.csv")
Z_CSV = os.path.join(ROOT, "data", "processed", "video_z_dataset.csv")


def batch_of(video_id: str) -> str:
    m = re.match(r"^([#@][^#@]*[#@])", video_id)
    return m.group(1) if m else video_id.split("\\")[0]


def build(target_csv: str, out_dir: str, keep_ids=None):
    df = pd.read_csv(target_csv)
    if keep_ids is not None:
        df = df[df["video_id"].isin(keep_ids)].reset_index(drop=True)
    df["batch"] = df["video_id"].apply(batch_of)

    batches = sorted(df["batch"].unique())
    os.makedirs(out_dir, exist_ok=True)
    print(f"{out_dir}: {len(batches)} batches -> {batches}")

    for i, held_out in enumerate(batches, start=1):
        val_df = df[df["batch"] == held_out].drop(columns=["batch"])
        train_df = df[df["batch"] != held_out].drop(columns=["batch"])

        val_df.to_csv(os.path.join(out_dir, f"fold_{i}_val.csv"), index=False)
        train_df.to_csv(os.path.join(out_dir, f"fold_{i}_train.csv"), index=False)

        meta = {"fold": i, "held_out_batch": held_out,
                "train_video_ids": train_df["video_id"].tolist(),
                "val_video_ids": val_df["video_id"].tolist()}
        with open(os.path.join(out_dir, f"fold_{i}_split.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"  fold_{i}: held out {held_out:<25} train={len(train_df):3d} val={len(val_df):3d}")

    return len(batches)


def main():
    z_ids = set(pd.read_csv(Z_CSV)["video_id"])

    n_xy = build(XY_CSV, os.path.join(ROOT, "data", "processed", "kfold_splits_xy_loo_batch"), keep_ids=z_ids)
    n_z = build(Z_CSV, os.path.join(ROOT, "data", "processed", "kfold_splits_z_loo_batch"))

    assert n_xy == n_z, f"x/y and z batch counts differ: {n_xy} vs {n_z}"
    print(f"\nDone. {n_xy} leave-one-batch-out folds written for both x/y and z.")


if __name__ == "__main__":
    main()
