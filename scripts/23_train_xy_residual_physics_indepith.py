# scripts/23_train_xy_residual_physics_indepith.py
#
# xy residual-physics, INDEPENDENT per-axis I_th (no joint fit with z) --
# same scheme as the pre-joint-I_th z run that beat both fused and
# baseline MLP (results/z_residual_physics_no joint_fitting/). x and y
# each get their own fit_lateral call; no cross-loading of z at all.
# Base + all 4 ablations, 5-fold only.

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
from mamba_xy.residual_physics import (
    train_cached_mamba_fold_xy_residual_indepith,
    train_cached_mamba_fold_xy_residual_indepith_shrinkage,
    train_cached_mamba_fold_xy_residual_indepith_attnpool,
    train_cached_mamba_fold_xy_residual_indepith_pretrained,
)

BASE_CFG = {
    "n_splits": 5, "n_mamba_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2,
    "hidden_dim": 128, "dropout": 0.3, "batch_size": 4, "epochs": 100, "lr": 1e-3,
    "weight_decay": 3e-4, "early_stopping_patience": 15, "min_delta": 0.0,
    "ept_threshold_um": 0.05, "max_trajectory_plots": 10, "seed": 42,
}

SIMPLIFIED_CFG_OVERRIDES = {
    "n_mamba_layers": 1, "d_state": 8, "hidden_dim": 32, "dropout": 0.5,
    "weight_decay": 1e-3,
}
ATTNPOOL_CFG_OVERRIDES = {
    "attn_n_layers": 2, "attn_n_heads": 4,
}
PRETRAINED_CFG_OVERRIDES = {
    "pretrain_lr": 1e-3, "pretrain_epochs": 50,
}


def run(out_dir_name, train_fn, cfg_overrides=None):
    cfg = deepcopy(BASE_CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)
    cfg["embeddings_dir"] = os.path.join(ROOT, "results", "dinov2_embeddings_xy_with_power")
    cfg["split_dir"] = os.path.join(ROOT, "data", "processed", "kfold_splits_xy")
    cfg["out_dir"] = os.path.join(ROOT, "results", out_dir_name)
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"xy residual-physics (independent I_th) | {out_dir_name} | {cfg['n_splits']} folds")
    print("=" * 80)

    histories, summaries = [], []
    for fold in range(1, cfg["n_splits"] + 1):
        h, s = train_fn(fold=fold, cfg=deepcopy(cfg), device=device)
        histories.append(h)
        summaries.append(s)

    pd.concat(histories, ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "history_all_folds.csv"), index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(cfg["out_dir"], "mamba_cached_summary.csv"), index=False)

    mae_keys = ["best_val_mae_x_phys", "best_val_mae_y_phys"]
    final = {"n_splits": cfg["n_splits"]}
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
    run("xy_residual_physics_indepith", train_cached_mamba_fold_xy_residual_indepith)
    run("xy_residual_physics_indepith_shrinkage", train_cached_mamba_fold_xy_residual_indepith_shrinkage)
    run("xy_residual_physics_indepith_simplified", train_cached_mamba_fold_xy_residual_indepith, SIMPLIFIED_CFG_OVERRIDES)
    run("xy_residual_physics_indepith_attnpool", train_cached_mamba_fold_xy_residual_indepith_attnpool, ATTNPOOL_CFG_OVERRIDES)
    run("xy_residual_physics_indepith_pretrained", train_cached_mamba_fold_xy_residual_indepith_pretrained, PRETRAINED_CFG_OVERRIDES)
    print("\nALL_XY_RESIDUAL_PHYSICS_INDEPITH_DONE")


if __name__ == "__main__":
    main()
