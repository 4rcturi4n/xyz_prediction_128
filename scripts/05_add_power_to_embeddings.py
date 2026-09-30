# scripts/05_add_power_to_embeddings.py
#
# Adds power_mW to each cached embedding payload, looked up by video_id
# from the corresponding split CSV. Runs for all four payload sets (x/y
# and z, 5-fold and leave-one-batch-out).
#
# Run after 04_build_fold_payloads.py.

import os
import glob

import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_power_lookup(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["video_id"], df["power_mW"].astype(float)))


def enrich(embeddings_dir, split_dir, out_dir, n_splits):
    for fold in range(1, n_splits + 1):
        for split_name in ("train", "val"):
            src_path = os.path.join(embeddings_dir, f"fold_{fold}", f"{split_name}_embeddings.pt")
            csv_path = os.path.join(split_dir, f"fold_{fold}_{split_name}.csv")
            if not os.path.exists(src_path):
                continue

            payload = torch.load(src_path, map_location="cpu", weights_only=False)
            power_lookup = load_power_lookup(csv_path)
            video_ids = payload["video_ids"]
            missing = [v for v in video_ids if v not in power_lookup]
            if missing:
                raise ValueError(f"{src_path}: {len(missing)} video_id(s) missing power_mW, e.g. {missing[:3]}")
            payload["power_mW"] = torch.tensor([power_lookup[v] for v in video_ids], dtype=torch.float32)

            fold_out = os.path.join(out_dir, f"fold_{fold}")
            os.makedirs(fold_out, exist_ok=True)
            torch.save(payload, os.path.join(fold_out, f"{split_name}_embeddings.pt"))
            print(f"{out_dir} fold_{fold}/{split_name}: {len(video_ids)} rows enriched")


def n_folds_in(split_dir):
    return len(glob.glob(os.path.join(split_dir, "fold_*_train.csv")))


def main():
    jobs = [
        ("dinov2_embeddings_xy", "kfold_splits_xy", "dinov2_embeddings_xy_with_power"),
        ("dinov2_embeddings_z", "kfold_splits_z", "dinov2_embeddings_z_with_power"),
        ("dinov2_embeddings_xy_loo_batch", "kfold_splits_xy_loo_batch", "dinov2_embeddings_xy_loo_batch_with_power"),
        ("dinov2_embeddings_z_loo_batch", "kfold_splits_z_loo_batch", "dinov2_embeddings_z_loo_batch_with_power"),
    ]
    for emb_name, split_name, out_name in jobs:
        embeddings_dir = os.path.join(ROOT, "results", emb_name)
        split_dir = os.path.join(ROOT, "data", "processed", split_name)
        out_dir = os.path.join(ROOT, "results", out_name)
        n_splits = n_folds_in(split_dir)
        print(f"\n=== {emb_name} ({n_splits} folds) ===")
        enrich(embeddings_dir, split_dir, out_dir, n_splits)
    print("\nDone.")


if __name__ == "__main__":
    main()
