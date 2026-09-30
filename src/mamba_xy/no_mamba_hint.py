"""
ABLATION: removes the Mamba trunk entirely. Tests whether the temporal
sequence modeling (Mamba's causal state-space processing across frames)
adds anything beyond a single frame's DINOv2 embedding + the physics
hint. Each timestep's prediction uses ONLY that timestep's own frame
embedding -- no memory of earlier frames at all, no pooling, nothing
Mamba-like -- concatenated with the same baseline-MLP hint used by
fused_hint_postmamba.py, then straight through a small MLP head applied
independently per timestep.

Same data, same folds, same hint features (sqrt(ln P) for x/y,
sqrt(sqrt(P)-1) for z, no P_th, leak-free internal-holdout baseline
fitting) as the other fused-hint experiments -- only the trunk differs.

Reuses (imports only, never modifies): the post-Mamba fusion's dataset
classes (raw embeddings kept separate from the hint, exactly what this
needs), evaluate_xy/evaluate_z, compute_loss_xy/compute_loss_z, and all
plotting from mamba_xy_regression.py / mamba_z_regression.py /
extra_plots.py.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mamba_xy.mamba_xy_regression import (
    evaluate_xy, compute_loss_xy, plot_mae_curve_xy,
    plot_video_trajectories_xy, load_bandwidth_lookup_xy, load_video_power_lookup_xy,
)
from mamba_xy.mamba_z_regression import (
    evaluate_z, compute_loss_z, plot_mae_curve_z,
    plot_video_trajectories_z, load_bandwidth_lookup_z, load_video_power_lookup_z,
)
from mamba_xy.fused_hint_xy import fit_baseline_mlp_no_leak as fit_baseline_mlp_no_leak_xy
from mamba_xy.fused_hint_z import fit_baseline_mlp_no_leak as fit_baseline_mlp_no_leak_z
from mamba_xy.fused_hint_postmamba import (
    CachedMambaEmbeddingDatasetXYPostMambaHint,
    CachedMambaEmbeddingDatasetZPostMambaHint,
)
from mamba_xy.extra_plots import (
    plot_learning_curve, compute_bandwidth_ept, plot_ept_distribution_bandwidth,
)


# --------------------------------------------------
# Models: no Mamba blocks at all -- LayerNorm -> concat hint -> MLP head,
# applied independently at every timestep
# --------------------------------------------------

class NoMambaHeadXY(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, dropout=0.3, hint_feat_dim=16):
        super().__init__()
        self.pre_head_norm = nn.LayerNorm(embed_dim)
        self.hint_proj = nn.Sequential(nn.Linear(2, hint_feat_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(embed_dim + hint_feat_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D] raw per-frame embeddings, no temporal mixing; hint: [B,2]."""
        x = self.pre_head_norm(x)
        B, T, D = x.shape
        hint_feat = self.hint_proj(hint).unsqueeze(1).expand(-1, T, -1)
        return self.head(torch.cat([x, hint_feat], dim=-1))  # [B, T, 2]


class NoMambaHeadZ(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, dropout=0.3, hint_feat_dim=16):
        super().__init__()
        self.pre_head_norm = nn.LayerNorm(embed_dim)
        self.hint_proj = nn.Sequential(nn.Linear(1, hint_feat_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(embed_dim + hint_feat_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D]; hint: [B] scalar per video."""
        x = self.pre_head_norm(x)
        B, T, D = x.shape
        hint_feat = self.hint_proj(hint.unsqueeze(-1)).unsqueeze(1).expand(-1, T, -1)
        return self.head(torch.cat([x, hint_feat], dim=-1))  # [B, T, 1]


# --------------------------------------------------
# x/y training
# --------------------------------------------------

def train_cached_mamba_fold_xy_no_mamba(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    x_train = train_payload_raw["targets_x_phys"].numpy()
    y_train = train_payload_raw["targets_y_phys"].numpy()

    baseline_model_x = fit_baseline_mlp_no_leak_xy(power_train, x_train, seed=cfg["seed"] + fold)
    baseline_model_y = fit_baseline_mlp_no_leak_xy(power_train, y_train, seed=cfg["seed"] + fold + 1000)
    print(f"Fold {fold} | baseline MLPs fit (TRAIN-internal-holdout only, no val leakage)")

    train_dataset = CachedMambaEmbeddingDatasetXYPostMambaHint(train_path, baseline_model_x, baseline_model_y)
    val_dataset = CachedMambaEmbeddingDatasetXYPostMambaHint(val_path, baseline_model_x, baseline_model_y)
    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    embed_dim = train_dataset.embeddings.shape[-1]
    model = NoMambaHeadXY(embed_dim=embed_dim, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()

    target_x_mean, target_x_std = train_dataset.target_x_mean, train_dataset.target_x_std
    target_y_mean, target_y_std = train_dataset.target_y_mean, train_dataset.target_y_std

    best_val_mae, best_epoch, epochs_no_imp, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for emb, targets_norm, targets_phys, loss_weight, hint, video_ids in train_loader:
            emb, targets_norm, hint = emb.to(device), targets_norm.to(device), hint.to(device)
            optimizer.zero_grad()
            preds = model(emb, hint)
            loss = compute_loss_xy(preds, targets_norm, loss_fn_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_xy(
            model=model, loader=val_loader, device=device,
            target_x_mean=target_x_mean, target_x_std=target_x_std,
            target_y_mean=target_y_mean, target_y_std=target_y_std,
            ept_threshold_um=cfg["ept_threshold_um"],
        )
        current_mae = (val_metrics["final_mae_x_phys"] + val_metrics["final_mae_y_phys"]) / 2.0
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae, best_epoch, epochs_no_imp = current_mae, epoch, 0
            torch.save({"fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg,
                        "target_x_mean": target_x_mean, "target_x_std": target_x_std,
                        "target_y_mean": target_y_mean, "target_y_std": target_y_std,
                        "best_val_mae": best_val_mae}, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, **val_metrics,
                         "val_mae_combined": current_mae, "best_val_mae_so_far": best_val_mae,
                         "improved": improved, "epochs_without_improvement": epochs_no_imp})
        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"mae_x={val_metrics['final_mae_x_phys']:.5f} mae_y={val_metrics['final_mae_y_phys']:.5f} | "
              f"best={best_val_mae:.5f} | no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}")

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    plot_learning_curve(history_df, os.path.join(out_dir, "learning_curve.png"), fold,
                         val_cols=[("final_mae_x_phys", "val_mae_x"), ("final_mae_y_phys", "val_mae_y")],
                         best_epoch_col="val_mae_combined")

    ckpt = torch.load(os.path.join(out_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_xy(
        model=model, loader=val_loader, device=device,
        target_x_mean=target_x_mean, target_x_std=target_x_std,
        target_y_mean=target_y_mean, target_y_std=target_y_std,
        ept_threshold_um=cfg["ept_threshold_um"],
    )
    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
    ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)

    plot_mae_curve_xy(mae_per_t_df, os.path.join(out_dir, "mae_curve.png"), fold)

    x_bw_lookup, y_bw_lookup = load_bandwidth_lookup_xy()
    video_power_lookup = load_video_power_lookup_xy(fold, split_dir=cfg.get("split_dir"))

    bandwidth_ept_df = compute_bandwidth_ept(
        per_video_df, axis_names=["x", "y"], bandwidth_lookups={"x": x_bw_lookup, "y": y_bw_lookup},
        video_power_lookup=video_power_lookup, ept_threshold_um=cfg["ept_threshold_um"],
    )
    plot_ept_distribution_bandwidth(bandwidth_ept_df, os.path.join(out_dir, "ept_distribution.png"), fold,
                                     "x and y, each within its own band")

    plot_video_trajectories_xy(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, x_bandwidth_lookup=x_bw_lookup, y_bandwidth_lookup=y_bw_lookup,
    )

    fold_summary = {
        "fold": fold, "best_epoch": best_epoch,
        "best_val_mae_x_phys": float(best_val_metrics["final_mae_x_phys"]),
        "best_val_mae_y_phys": float(best_val_metrics["final_mae_y_phys"]),
        "best_val_mae_combined": float(best_val_mae),
        "best_val_mae_std_x_phys": float(best_val_metrics["final_mae_std_x_phys"]),
        "best_val_mae_std_y_phys": float(best_val_metrics["final_mae_std_y_phys"]),
        "best_val_rmse_x_phys": float(best_val_metrics["final_rmse_x_phys"]),
        "best_val_rmse_y_phys": float(best_val_metrics["final_rmse_y_phys"]),
        "best_val_bias_x_phys": float(best_val_metrics["final_bias_x_phys"]),
        "best_val_bias_y_phys": float(best_val_metrics["final_bias_y_phys"]),
        "mean_ept_pct": float(best_val_metrics["mean_ept_pct"]),
        "pct_videos_ept_found": float(best_val_metrics["pct_videos_ept_found"]),
        "num_train_rows": len(train_dataset), "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary


# --------------------------------------------------
# z training
# --------------------------------------------------

def train_cached_mamba_fold_z_no_mamba(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    z_train = train_payload_raw["targets_z_phys"].numpy()

    baseline_model_z = fit_baseline_mlp_no_leak_z(power_train, z_train, seed=cfg["seed"] + fold)
    print(f"Fold {fold} | baseline MLP fit (TRAIN-internal-holdout only, no val leakage)")

    train_dataset = CachedMambaEmbeddingDatasetZPostMambaHint(train_path, baseline_model_z)
    val_dataset = CachedMambaEmbeddingDatasetZPostMambaHint(val_path, baseline_model_z)
    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    embed_dim = train_dataset.embeddings.shape[-1]
    model = NoMambaHeadZ(embed_dim=embed_dim, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()

    target_z_mean, target_z_std = train_dataset.target_z_mean, train_dataset.target_z_std

    best_val_mae, best_epoch, epochs_no_imp, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for emb, targets_norm, targets_phys, loss_weight, hint, video_ids in train_loader:
            emb, targets_norm, hint = emb.to(device), targets_norm.to(device), hint.to(device)
            optimizer.zero_grad()
            preds = model(emb, hint)
            loss = compute_loss_z(preds, targets_norm, loss_fn_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_z(
            model=model, loader=val_loader, device=device,
            target_z_mean=target_z_mean, target_z_std=target_z_std,
            ept_threshold_um=cfg["ept_threshold_um"],
        )
        current_mae = val_metrics["final_mae_z_phys"]
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae, best_epoch, epochs_no_imp = current_mae, epoch, 0
            torch.save({"fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg,
                        "target_z_mean": target_z_mean, "target_z_std": target_z_std,
                        "best_val_mae": best_val_mae}, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, **val_metrics,
                         "val_mae": current_mae, "best_val_mae_so_far": best_val_mae,
                         "improved": improved, "epochs_without_improvement": epochs_no_imp})
        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"mae_z={val_metrics['final_mae_z_phys']:.5f} | best={best_val_mae:.5f} | "
              f"no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}")

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    plot_learning_curve(history_df, os.path.join(out_dir, "learning_curve.png"), fold,
                         val_cols=[("final_mae_z_phys", "val_mae_z")])

    ckpt = torch.load(os.path.join(out_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_z(
        model=model, loader=val_loader, device=device,
        target_z_mean=target_z_mean, target_z_std=target_z_std,
        ept_threshold_um=cfg["ept_threshold_um"],
    )
    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
    ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)

    plot_mae_curve_z(mae_per_t_df, os.path.join(out_dir, "mae_curve.png"), fold)

    z_bw_lookup = load_bandwidth_lookup_z()
    video_power_lookup = load_video_power_lookup_z(fold, split_dir=cfg.get("split_dir"))

    bandwidth_ept_df = compute_bandwidth_ept(
        per_video_df, axis_names=["z"], bandwidth_lookups={"z": z_bw_lookup},
        video_power_lookup=video_power_lookup, ept_threshold_um=cfg["ept_threshold_um"],
    )
    plot_ept_distribution_bandwidth(bandwidth_ept_df, os.path.join(out_dir, "ept_distribution.png"), fold, "z")

    plot_video_trajectories_z(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, bandwidth_lookup=z_bw_lookup,
    )

    fold_summary = {
        "fold": fold, "best_epoch": best_epoch,
        "best_val_mae_z_phys": float(best_val_metrics["final_mae_z_phys"]),
        "best_val_mae_std_z_phys": float(best_val_metrics["final_mae_std_z_phys"]),
        "best_val_rmse_z_phys": float(best_val_metrics["final_rmse_z_phys"]),
        "best_val_bias_z_phys": float(best_val_metrics["final_bias_z_phys"]),
        "mean_ept_pct": float(best_val_metrics["mean_ept_pct"]),
        "pct_videos_ept_found": float(best_val_metrics["pct_videos_ept_found"]),
        "num_train_rows": len(train_dataset), "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary
