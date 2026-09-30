# scripts/04_build_fold_payloads.py
#
# Builds fold-organized {train,val}_embeddings.pt payloads (the format
# every downstream training script expects) from the master DINOv2 cache
# (results/dinov2_master_cache/), for BOTH split schemes:
#   - kfold_splits_xy / kfold_splits_z            (5-fold stratified)
#   - kfold_splits_xy_loo_batch / kfold_splits_z_loo_batch  (leave-one-batch-out)
# No DINOv2 forward pass here -- pure tensor lookup/assembly, so re-running
# this after changing fold membership is fast.
#
# Run after 03_extract_dinov2_embeddings_master.py.
#
# Writes:
#   results/dinov2_embeddings_xy/fold_N/{train,val}_embeddings.pt
#   results/dinov2_embeddings_z/fold_N/{train,val}_embeddings.pt
#   results/dinov2_embeddings_xy_loo_batch/fold_N/{train,val}_embeddings.pt
#   results/dinov2_embeddings_z_loo_batch/fold_N/{train,val}_embeddings.pt

import os
import glob

import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "results", "dinov2_master_cache")


def safe_name(video_id: str) -> str:
    return video_id.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_cached(video_id: str, variant: str) -> torch.Tensor:
    path = os.path.join(CACHE_DIR, f"{safe_name(video_id)}__{variant}.pt")
    return torch.load(path, map_location="cpu", weights_only=False)["embedding"]


def build_payload(df: pd.DataFrame, variants, target_cols, error_cols, target_stats=None):
    """target_cols/error_cols: {axis_name: column_name}. target_stats: {axis_name: (mean, std)},
    computed from this call's own df if None (i.e. call once on train, reuse for val)."""
    if target_stats is None:
        target_stats = {a: (float(df[c].mean()), float(df[c].std() + 1e-8)) for a, c in target_cols.items()}

    embeddings, video_ids, variant_names, is_augmented = [], [], [], []
    phys = {a: [] for a in target_cols}
    norm = {a: [] for a in target_cols}
    errs = {a: [] for a in error_cols}

    for _, row in df.iterrows():
        vid = row["video_id"]
        for variant in variants:
            embeddings.append(load_cached(vid, variant))
            video_ids.append(vid)
            variant_names.append(variant)
            is_augmented.append(variant != "clean")
            for a, c in target_cols.items():
                v = float(row[c])
                phys[a].append(v)
                norm[a].append((v - target_stats[a][0]) / target_stats[a][1])
            for a, c in error_cols.items():
                v = row.get(c)
                errs[a].append(float(v) if pd.notna(v) else float("nan"))

    payload = {
        "embeddings": torch.stack(embeddings, dim=0),
        "video_ids": video_ids, "variant_names": variant_names, "is_augmented": is_augmented,
    }
    for a in target_cols:
        payload[f"targets_{a}"] = torch.tensor(norm[a], dtype=torch.float32)
        payload[f"targets_{a}_phys"] = torch.tensor(phys[a], dtype=torch.float32)
        payload[f"target_{a}_mean"], payload[f"target_{a}_std"] = target_stats[a]
    for a in error_cols:
        payload[f"errors_{a}"] = torch.tensor(errs[a], dtype=torch.float32)
    return payload


def build_target(split_dir: str, out_dir: str, target_cols, error_cols, n_splits: int):
    for fold in range(1, n_splits + 1):
        train_csv = os.path.join(split_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(split_dir, f"fold_{fold}_val.csv")
        if not os.path.exists(train_csv):
            break
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        train_payload = build_payload(train_df, ["clean", "mirror"], target_cols, error_cols)
        target_stats = {a: (train_payload[f"target_{a}_mean"], train_payload[f"target_{a}_std"]) for a in target_cols}
        val_payload = build_payload(val_df, ["clean"], target_cols, error_cols, target_stats=target_stats)

        fold_out = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_out, exist_ok=True)
        torch.save(train_payload, os.path.join(fold_out, "train_embeddings.pt"))
        torch.save(val_payload, os.path.join(fold_out, "val_embeddings.pt"))
        print(f"{out_dir} fold_{fold}: train {tuple(train_payload['embeddings'].shape)}, "
              f"val {tuple(val_payload['embeddings'].shape)}")


def n_folds_in(split_dir: str) -> int:
    return len(glob.glob(os.path.join(split_dir, "fold_*_train.csv")))


def main():
    xy_target_cols = {"x": "x_resolution", "y": "y_resolution"}
    xy_error_cols = {"x": "x_error", "y": "y_error"}
    z_target_cols = {"z": "z_resolution"}
    z_error_cols = {"z": "z_error"}

    jobs = [
        ("kfold_splits_xy", "dinov2_embeddings_xy", xy_target_cols, xy_error_cols),
        ("kfold_splits_z", "dinov2_embeddings_z", z_target_cols, z_error_cols),
        ("kfold_splits_xy_loo_batch", "dinov2_embeddings_xy_loo_batch", xy_target_cols, xy_error_cols),
        ("kfold_splits_z_loo_batch", "dinov2_embeddings_z_loo_batch", z_target_cols, z_error_cols),
    ]
    for split_name, out_name, target_cols, error_cols in jobs:
        split_dir = os.path.join(ROOT, "data", "processed", split_name)
        out_dir = os.path.join(ROOT, "results", out_name)
        n_splits = n_folds_in(split_dir)
        print(f"\n=== {split_name} -> {out_name} ({n_splits} folds) ===")
        build_target(split_dir, out_dir, target_cols, error_cols, n_splits)

    print("\nDone.")


if __name__ == "__main__":
    main()
