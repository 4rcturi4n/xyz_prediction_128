# scripts/07_train_fused_hint_postmamba.py
#
# DINOv2 + Mamba + power-only-baseline-MLP hint, fused via a dedicated
# branch AFTER the Mamba trunk. Correct per-target hint: sqrt(ln(P)) for
# x/y, sqrt(sqrt(P)-1) for z (no P_th ever computed -- leak-free
# internal-holdout baseline fitting per fold). Both split schemes,
# patience=15.

import os
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from mamba_xy.core import set_seed
from mamba_xy.fused_hint_postmamba import (
    train_cached_mamba_fold_xy_postmamba_hint,
    train_cached_mamba_fold_z_postmamba_hint,
)

BASE_CFG = {
    "n_splits": 5, "n_mamba_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2,
    "hidden_dim": 128, "dropout": 0.3, "batch_size": 4, "epochs": 100, "lr": 1e-3,
    "weight_decay": 3e-4, "early_stopping_patience": 15, "min_delta": 0.0,
    "ept_threshold_um": 0.05, "max_trajectory_plots": 10, "seed": 42,
}


def run(embeddings_dir_name, out_dir_name, split_dir_name, n_splits, train_fn, mae_keys):
    cfg = deepcopy(BASE_CFG)
    cfg["n_splits"] = n_splits
    cfg["embeddings_dir"] = os.path.join(ROOT, "results", embeddings_dir_name)
    cfg["out_dir"] = os.path.join(ROOT, "results", out_dir_name)
    cfg["split_dir"] = os.path.join(ROOT, "data", "processed", split_dir_name)
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"fused-hint-postmamba | {out_dir_name} | {n_splits} folds")
    print("=" * 80)

    histories, summaries = [], []
    for fold in range(1, n_splits + 1):
        h, s = train_fn(fold=fold, cfg=deepcopy(cfg), device=device)
        histories.append(h)
        summaries.append(s)

    pd.concat(histories, ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "history_all_folds.csv"), index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(cfg["out_dir"], "mamba_cached_summary.csv"), index=False)

    final = {"n_splits": n_splits}
    for key in mae_keys:
        vals = summary_df[key].values
        final[f"mean_{key}"] = float(np.mean(vals))
        final[f"std_{key}"] = float(np.std(vals))
    final["folds"] = summaries
    with open(os.path.join(cfg["out_dir"], "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    for key in mae_keys:
        print(f"{key}: {final[f'mean_{key}']:.5f} +/- {final[f'std_{key}']:.5f}")


def main():
    run("dinov2_embeddings_xy_with_power", "xy_dinov2_mamba_fused", "kfold_splits_xy", 5,
        train_cached_mamba_fold_xy_postmamba_hint, ["best_val_mae_x_phys", "best_val_mae_y_phys"])
    run("dinov2_embeddings_xy_loo_batch_with_power", "xy_dinov2_mamba_fused_loo_batch", "kfold_splits_xy_loo_batch", 6,
        train_cached_mamba_fold_xy_postmamba_hint, ["best_val_mae_x_phys", "best_val_mae_y_phys"])
    run("dinov2_embeddings_z_with_power", "z_dinov2_mamba_fused", "kfold_splits_z", 5,
        train_cached_mamba_fold_z_postmamba_hint, ["best_val_mae_z_phys"])
    run("dinov2_embeddings_z_loo_batch_with_power", "z_dinov2_mamba_fused_loo_batch", "kfold_splits_z_loo_batch", 6,
        train_cached_mamba_fold_z_postmamba_hint, ["best_val_mae_z_phys"])
    print("\nALL_FUSED_HINT_POSTMAMBA_DONE")


if __name__ == "__main__":
    main()
