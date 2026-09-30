# scripts/13_build_handcrafted_embeddings.py
#
# Builds 128-point hand-crafted-feature embeddings (video-only model only --
# the sole handcrafted variant requested), in the same payload format the
# DINOv2 pipeline produces, so the existing video-only Mamba trainer runs
# on it unchanged.
#
# The real blob-measurement code (ablations/extract_frame_features.py) is
# gone from disk, but data/external/frame_feats.pkl -- the ablations'
# own dense feature cache -- has the ACTUAL already-computed per-frame
# values (from that same now-missing blob_features()) at native-resolution
# dense sampling (every 5th frame, ~220-318 frames/video, starting at 2.0s
# -- verified against SKIP_SECONDS). All 106 videos are covered. We
# resample each video's dense sequence to a fixed 128 points via linspace
# (matching the DINOv2-128 frame count for direct comparison) -- no
# re-measurement, no reconstruction, same underlying numbers as the
# original hand-crafted results.
#
# Feature transform (NaN-fill cx/cy with crop centre, inf/nan -> 0, mirror
# flips cx only, log1p on the area-like features) and per-fold
# standardization (train-split stats only) match
# xy_early_prediction/scripts/03b_extract_handcrafted_embeddings.py exactly.
#
# Writes: results/handcrafted_embeddings_{xy,z}[_loo_batch]_with_power/
#         fold_N/{train,val}_embeddings.pt

import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_PATH = os.path.join(ROOT, "data", "external", "frame_feats.pkl")
NUM_FRAMES = 128
CROP_BOX = (1049, 337, 1831, 915)
LOG_FEATURES = {"area25", "area60", "area120", "sat_area", "int_diff"}

with open(CACHE_PATH, "rb") as f:
    _CACHE = pickle.load(f)
FEATURES = list(_CACHE["features"])
CX, CY = FEATURES.index("cx"), FEATURES.index("cy")


def resampled_raw(video_id: str) -> np.ndarray:
    """[NUM_FRAMES, len(FEATURES)] -- fixed 128-point linspace resample of
    the dense native-resolution feature sequence."""
    seq = _CACHE["data"][video_id]["feats"]["native"].astype(np.float64)
    idx = np.round(np.linspace(0, len(seq) - 1, NUM_FRAMES)).astype(int)
    return seq[idx]


def transform(raw: np.ndarray, variant: str) -> np.ndarray:
    """Raw [T, F] -> model-ready [T, F], before standardization."""
    x = raw.copy()
    w, h = CROP_BOX[2] - CROP_BOX[0], CROP_BOX[3] - CROP_BOX[1]
    x[:, CX] = np.where(np.isnan(x[:, CX]), w / 2, x[:, CX])
    x[:, CY] = np.where(np.isnan(x[:, CY]), h / 2, x[:, CY])
    x[~np.isfinite(x)] = 0.0
    if variant == "mirror":
        x[:, CX] = w - x[:, CX]
    for k, name in enumerate(FEATURES):
        if name in LOG_FEATURES:
            x[:, k] = np.log1p(np.clip(x[:, k], 0, None))
    return x


def build_payload(df, variants, target_cols, error_cols, target_stats=None, feature_stats=None):
    if target_stats is None:
        target_stats = {a: (float(df[c].mean()), float(df[c].std() + 1e-8)) for a, c in target_cols.items()}

    raw_x, video_ids, variant_names, is_augmented, power = [], [], [], [], []
    phys = {a: [] for a in target_cols}
    norm = {a: [] for a in target_cols}
    errs = {a: [] for a in error_cols}

    for _, row in df.iterrows():
        vid = row["video_id"]
        raw = resampled_raw(vid)
        for variant in variants:
            raw_x.append(transform(raw, variant))
            video_ids.append(vid)
            variant_names.append(variant)
            is_augmented.append(variant != "clean")
            power.append(float(row["power_mW"]))
            for a, c in target_cols.items():
                v = float(row[c])
                phys[a].append(v)
                norm[a].append((v - target_stats[a][0]) / target_stats[a][1])
            for a, c in error_cols.items():
                v = row.get(c)
                errs[a].append(float(v) if pd.notna(v) else float("nan"))

    raw_x = np.stack(raw_x, axis=0)  # [N, T, F]
    if feature_stats is None:
        flat = raw_x.reshape(-1, len(FEATURES))
        feature_stats = (flat.mean(axis=0), flat.std(axis=0) + 1e-8)
    mean, std = feature_stats
    x = (raw_x - mean) / std

    payload = {
        "embeddings": torch.tensor(x, dtype=torch.float32),
        "video_ids": video_ids, "variant_names": variant_names, "is_augmented": is_augmented,
        "power_mW": torch.tensor(power, dtype=torch.float32),
        "feature_names": list(FEATURES),
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
    }
    for a in target_cols:
        payload[f"targets_{a}"] = torch.tensor(norm[a], dtype=torch.float32)
        payload[f"targets_{a}_phys"] = torch.tensor(phys[a], dtype=torch.float32)
        payload[f"target_{a}_mean"], payload[f"target_{a}_std"] = target_stats[a]
    for a in error_cols:
        payload[f"errors_{a}"] = torch.tensor(errs[a], dtype=torch.float32)
    return payload, feature_stats


def build_target(split_dir: str, out_dir: str, target_cols, error_cols, n_splits: int):
    for fold in range(1, n_splits + 1):
        train_csv = os.path.join(split_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(split_dir, f"fold_{fold}_val.csv")
        if not os.path.exists(train_csv):
            break
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        train_payload, feature_stats = build_payload(train_df, ["clean", "mirror"], target_cols, error_cols)
        target_stats = {a: (train_payload[f"target_{a}_mean"], train_payload[f"target_{a}_std"]) for a in target_cols}
        val_payload, _ = build_payload(val_df, ["clean"], target_cols, error_cols,
                                        target_stats=target_stats, feature_stats=feature_stats)

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
        ("kfold_splits_xy", "handcrafted_embeddings_xy_with_power", xy_target_cols, xy_error_cols),
        ("kfold_splits_z", "handcrafted_embeddings_z_with_power", z_target_cols, z_error_cols),
        ("kfold_splits_xy_loo_batch", "handcrafted_embeddings_xy_loo_batch_with_power", xy_target_cols, xy_error_cols),
        ("kfold_splits_z_loo_batch", "handcrafted_embeddings_z_loo_batch_with_power", z_target_cols, z_error_cols),
    ]
    for split_name, out_name, target_cols, error_cols in jobs:
        split_dir = os.path.join(ROOT, "data", "processed", split_name)
        out_dir = os.path.join(ROOT, "results", out_name)
        n_splits = n_folds_in(split_dir)
        print(f"\n=== {split_name} -> {out_name} ({n_splits} folds) ===")
        build_target(split_dir, out_dir, target_cols, error_cols, n_splits)

    print(f"\n{len(FEATURES)} features: {FEATURES}")
    print("Done.")


if __name__ == "__main__":
    main()
