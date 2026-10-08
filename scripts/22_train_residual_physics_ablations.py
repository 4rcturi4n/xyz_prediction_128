# scripts/22_train_residual_physics_ablations.py
#
# Four ideas to help the video pathway beat/match the baseline MLP,
# each built as a separate, isolated ablation on top of the
# residual-physics model, for BOTH x/y and z (5-fold only so far --
# extend to loo-batch once we know which ones are worth it):
#
#   1) learned shrinkage  -- fit scalar alpha(s) (OLS, leak-free train
#      holdout) on how much to trust the residual correction.
#   2) simplified architecture -- same residual-physics setup, smaller
#      model + more regularization (less capacity to overfit on ~85
#      train videos/fold).
#   3) causal attention pooling -- replace the Mamba trunk with a
#      2-layer causal self-attention trunk.
#   4) self-supervised pretraining -- pretrain the Mamba trunk via
#      next-embedding prediction (no labels) on that fold's train
#      embeddings before residual fine-tuning.
#
# All use the joint shared-I_th physics fit (one I_th across x/y/z),
# so every xy run also cross-loads the matched z fold and vice versa.
# Same full reporting suite (MAE, EPT, learning curves) as every other
# model in the repo.

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
    train_cached_mamba_fold_z_residual_shrinkage,
    train_cached_mamba_fold_z_residual,
    train_cached_mamba_fold_z_residual_attnpool,
    train_cached_mamba_fold_z_residual_pretrained,
    train_cached_mamba_fold_xy_residual_shrinkage,
    train_cached_mamba_fold_xy_residual,
    train_cached_mamba_fold_xy_residual_attnpool,
    train_cached_mamba_fold_xy_residual_pretrained,
)

BASE_CFG = {
    "n_splits": 5, "n_mamba_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2,
    "hidden_dim": 128, "dropout": 0.3, "batch_size": 4, "epochs": 100, "lr": 1e-3,
    "weight_decay": 3e-4, "early_stopping_patience": 15, "min_delta": 0.0,
    "ept_threshold_um": 0.05, "max_trajectory_plots": 10, "seed": 42,
}

# Ablation 2: simplified architecture -- smaller model, more regularization.
SIMPLIFIED_CFG_OVERRIDES = {
    "n_mamba_layers": 1, "d_state": 8, "hidden_dim": 32, "dropout": 0.5,
    "weight_decay": 1e-3,
}

# Ablation 3: causal attention pooling trunk hyperparameters.
ATTNPOOL_CFG_OVERRIDES = {
    "attn_n_layers": 2, "attn_n_heads": 4,
}

# Ablation 4: self-supervised pretraining hyperparameters.
PRETRAINED_CFG_OVERRIDES = {
    "pretrain_lr": 1e-3, "pretrain_epochs": 50,
}


def run(axis, out_dir_name, train_fn, mae_keys, cfg_overrides=None):
    """axis: 'z' or 'xy'. Each cross-loads the matched fold of the other
    axis for the joint shared-I_th physics fit."""
    cfg = deepcopy(BASE_CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)
    if axis == "z":
        cfg["embeddings_dir"] = os.path.join(ROOT, "results", "dinov2_embeddings_z_with_power")
        cfg["xy_embeddings_dir"] = os.path.join(ROOT, "results", "dinov2_embeddings_xy_with_power")
        cfg["split_dir"] = os.path.join(ROOT, "data", "processed", "kfold_splits_z")
    elif axis == "xy":
        cfg["embeddings_dir"] = os.path.join(ROOT, "results", "dinov2_embeddings_xy_with_power")
        cfg["z_embeddings_dir"] = os.path.join(ROOT, "results", "dinov2_embeddings_z_with_power")
        cfg["split_dir"] = os.path.join(ROOT, "data", "processed", "kfold_splits_xy")
    else:
        raise ValueError(axis)
    cfg["out_dir"] = os.path.join(ROOT, "results", out_dir_name)
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"residual-physics ablation | {out_dir_name} | {cfg['n_splits']} folds")
    print("=" * 80)

    histories, summaries = [], []
    for fold in range(1, cfg["n_splits"] + 1):
        h, s = train_fn(fold=fold, cfg=deepcopy(cfg), device=device)
        histories.append(h)
        summaries.append(s)

    pd.concat(histories, ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "history_all_folds.csv"), index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(cfg["out_dir"], "mamba_cached_summary.csv"), index=False)

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
    z_mae_keys = ["best_val_mae_z_phys"]
    xy_mae_keys = ["best_val_mae_x_phys", "best_val_mae_y_phys"]

    # _jointith suffix: these use the joint shared-I_th physics fit, distinct
    # from any earlier z_residual_physics_* run made before that fix landed.
    run("z", "z_residual_physics_shrinkage_jointith", train_cached_mamba_fold_z_residual_shrinkage, z_mae_keys)
    run("z", "z_residual_physics_simplified_jointith", train_cached_mamba_fold_z_residual, z_mae_keys, SIMPLIFIED_CFG_OVERRIDES)
    run("z", "z_residual_physics_attnpool_jointith", train_cached_mamba_fold_z_residual_attnpool, z_mae_keys, ATTNPOOL_CFG_OVERRIDES)
    run("z", "z_residual_physics_pretrained_jointith", train_cached_mamba_fold_z_residual_pretrained, z_mae_keys, PRETRAINED_CFG_OVERRIDES)

    run("xy", "xy_residual_physics_shrinkage", train_cached_mamba_fold_xy_residual_shrinkage, xy_mae_keys)
    run("xy", "xy_residual_physics_simplified", train_cached_mamba_fold_xy_residual, xy_mae_keys, SIMPLIFIED_CFG_OVERRIDES)
    run("xy", "xy_residual_physics_attnpool", train_cached_mamba_fold_xy_residual_attnpool, xy_mae_keys, ATTNPOOL_CFG_OVERRIDES)
    run("xy", "xy_residual_physics_pretrained", train_cached_mamba_fold_xy_residual_pretrained, xy_mae_keys, PRETRAINED_CFG_OVERRIDES)

    print("\nALL_RESIDUAL_PHYSICS_ABLATIONS_DONE")


if __name__ == "__main__":
    main()
