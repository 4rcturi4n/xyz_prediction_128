"""
Post-Mamba fusion variant -- isolates whether fusing BEFORE vs AFTER the
Mamba trunk is what hurt z's results in fused_hint_z.py (concatenating
into the raw embeddings, pre-Mamba). Same baseline-MLP hint predictions
as fused_hint_xy.py / fused_hint_z.py (sqrt(ln P) for x/y, sqrt(sqrt(P)-1)
for z, no P_th, leak-free internal-holdout fitting), but fused the same
way the ORIGINAL sqrt(power) model did: a small dedicated projection
branch, concatenated to the embedding AFTER the Mamba trunk, right before
the head. Same new head size (128) / dropout (0.3) / weight_decay (3e-4)
as the pre-Mamba variant, so only the fusion point differs.

For z, this reuses MambaSequenceRegressorZWithPower / evaluate_z /
compute_loss_z unchanged (its power_proj already takes a single scalar,
which is exactly what z's one hint value is). For x/y, a new model class
is needed since the original power_proj is Linear(1, ...) but x/y's hint
is 2 values (pred_x, pred_y) -- everything else (evaluate_xy,
compute_loss_xy, plotting) is reused unchanged.

Does not touch mamba_xy_regression.py / mamba_z_regression.py /
fused_hint_xy.py / fused_hint_z.py.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mamba_xy.core import build_mamba_layer
from mamba_xy.mamba_xy_regression import (
    evaluate_xy, compute_loss_xy, plot_mae_curve_xy,
    plot_video_trajectories_xy, load_bandwidth_lookup_xy, load_video_power_lookup_xy,
)
from mamba_xy.mamba_z_regression import (
    MambaSequenceRegressorZWithPower, evaluate_z, compute_loss_z, plot_mae_curve_z,
    plot_video_trajectories_z, load_bandwidth_lookup_z, load_video_power_lookup_z,
)
from mamba_xy.fused_hint_xy import fit_baseline_mlp_no_leak as fit_baseline_mlp_no_leak_xy
from mamba_xy.fused_hint_xy import baseline_predict as baseline_predict_xy
from mamba_xy.fused_hint_xy import load_video_path_lookup_xy
from mamba_xy.fused_hint_z import fit_baseline_mlp_no_leak as fit_baseline_mlp_no_leak_z
from mamba_xy.fused_hint_z import baseline_predict as baseline_predict_z
from mamba_xy.fused_hint_z import load_video_path_lookup_z
from mamba_xy.extra_plots import (
    plot_learning_curve, save_video_frames_for_fold,
    compute_bandwidth_ept, plot_ept_distribution_bandwidth,
)


# --------------------------------------------------
# x/y: new model class (power_proj takes 2 hint values, not 1)
# --------------------------------------------------

class MambaSequenceRegressorXYWithHint(nn.Module):
    def __init__(self, embed_dim, n_mamba_layers=2, d_state=16, d_conv=4, expand=2,
                 hidden_dim=128, dropout=0.3, hint_feat_dim=16):
        super().__init__()
        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(embed_dim),
                "mamba": build_mamba_layer(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
            })
            for _ in range(n_mamba_layers)
        ])
        self.pre_head_norm = nn.LayerNorm(embed_dim)
        self.hint_proj = nn.Sequential(nn.Linear(2, hint_feat_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(embed_dim + hint_feat_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D]; hint: [B,2] (normalized baseline pred_x, pred_y)."""
        for block in self.mamba_blocks:
            x = x + block["mamba"](block["norm"](x))
        x = self.pre_head_norm(x)
        B, T, D = x.shape
        hint_feat = self.hint_proj(hint)                      # [B, hint_feat_dim]
        hint_feat = hint_feat.unsqueeze(1).expand(-1, T, -1)   # [B, T, hint_feat_dim]
        combined = torch.cat([x, hint_feat], dim=-1)
        return self.head(combined)   # [B, T, 2]


class CachedMambaEmbeddingDatasetXYPostMambaHint(Dataset):
    def __init__(self, payload_path: str, baseline_model_x, baseline_model_y, loss_weight_min_eps: float = 1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        self.embeddings = payload["embeddings"].float()      # [N, T, D] -- UNCHANGED
        self.targets_x = payload["targets_x"].float()
        self.targets_y = payload["targets_y"].float()
        self.targets_x_phys = payload["targets_x_phys"].float()
        self.targets_y_phys = payload["targets_y_phys"].float()
        self.errors_x_phys = payload["errors_x"].float()
        self.errors_y_phys = payload["errors_y"].float()
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()

        self.target_x_mean = float(payload["target_x_mean"])
        self.target_x_std = float(payload["target_x_std"])
        self.target_y_mean = float(payload["target_y_mean"])
        self.target_y_std = float(payload["target_y_std"])

        pred_x = baseline_predict_xy(baseline_model_x, self.power_mW.numpy())
        pred_y = baseline_predict_xy(baseline_model_y, self.power_mW.numpy())
        pred_x_norm = (pred_x - self.target_x_mean) / self.target_x_std
        pred_y_norm = (pred_y - self.target_y_mean) / self.target_y_std
        self.hint = torch.tensor(np.stack([pred_x_norm, pred_y_norm], axis=1), dtype=torch.float32)  # [N, 2]

        self.loss_weight_x = self._compute_loss_weights(self.errors_x_phys, self.target_x_std, loss_weight_min_eps)
        self.loss_weight_y = self._compute_loss_weights(self.errors_y_phys, self.target_y_std, loss_weight_min_eps)

    @staticmethod
    def _compute_loss_weights(errors_phys, target_std, eps):
        errors_norm = errors_phys / target_std
        raw_weights = 1.0 / errors_norm.clamp_min(eps).pow(2)
        valid = ~torch.isnan(errors_phys)
        mean_valid_weight = raw_weights[valid].mean() if valid.sum() > 0 else torch.tensor(1.0)
        return torch.where(valid, raw_weights / mean_valid_weight, torch.ones_like(raw_weights))

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        emb = self.embeddings[idx].clone()
        target_norm = torch.stack([self.targets_x[idx], self.targets_y[idx]])
        target_phys = torch.stack([self.targets_x_phys[idx], self.targets_y_phys[idx]])
        loss_weight = torch.stack([self.loss_weight_x[idx], self.loss_weight_y[idx]])
        video_id = self.video_ids[idx]
        return emb, target_norm, target_phys, loss_weight, self.hint[idx], str(video_id)


def train_cached_mamba_fold_xy_postmamba_hint(fold: int, cfg: dict, device: torch.device):
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
    model = MambaSequenceRegressorXYWithHint(
        embed_dim=embed_dim, n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

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
    plot_ept_distribution_bandwidth(bandwidth_ept_df, os.path.join(out_dir, "ept_distribution.png"), fold, "x and y, each within its own band")

    plot_video_trajectories_xy(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, x_bandwidth_lookup=x_bw_lookup, y_bandwidth_lookup=y_bw_lookup,
    )
    video_path_lookup = load_video_path_lookup_xy(fold, split_dir=cfg.get("split_dir"))
    save_video_frames_for_fold(
        per_video_df, video_path_lookup=video_path_lookup,
        out_dir=os.path.join(out_dir, "monitoring_plots_frames"),
        max_videos=cfg.get("max_trajectory_plots", 10),
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
# z: reuses MambaSequenceRegressorZWithPower / evaluate_z / compute_loss_z
# unchanged -- its power_proj already takes a single scalar
# --------------------------------------------------

class CachedMambaEmbeddingDatasetZPostMambaHint(Dataset):
    def __init__(self, payload_path: str, baseline_model_z, loss_weight_min_eps: float = 1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        self.embeddings = payload["embeddings"].float()      # [N, T, D] -- UNCHANGED
        self.targets_z = payload["targets_z"].float()
        self.targets_z_phys = payload["targets_z_phys"].float()
        self.errors_z_phys = payload["errors_z"].float()
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()

        self.target_z_mean = float(payload["target_z_mean"])
        self.target_z_std = float(payload["target_z_std"])

        pred_z = baseline_predict_z(baseline_model_z, self.power_mW.numpy())
        pred_z_norm = (pred_z - self.target_z_mean) / self.target_z_std
        self.hint = torch.tensor(pred_z_norm, dtype=torch.float32)  # [N] scalar per video

        self.loss_weight_z = self._compute_loss_weights(self.errors_z_phys, self.target_z_std, loss_weight_min_eps)

    @staticmethod
    def _compute_loss_weights(errors_phys, target_std, eps):
        errors_norm = errors_phys / target_std
        raw_weights = 1.0 / errors_norm.clamp_min(eps).pow(2)
        valid = ~torch.isnan(errors_phys)
        mean_valid_weight = raw_weights[valid].mean() if valid.sum() > 0 else torch.tensor(1.0)
        return torch.where(valid, raw_weights / mean_valid_weight, torch.ones_like(raw_weights))

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        emb = self.embeddings[idx].clone()
        target_norm = self.targets_z[idx].unsqueeze(0)
        target_phys = self.targets_z_phys[idx].unsqueeze(0)
        loss_weight = self.loss_weight_z[idx].unsqueeze(0)
        video_id = self.video_ids[idx]
        return emb, target_norm, target_phys, loss_weight, self.hint[idx], str(video_id)


def train_cached_mamba_fold_z_postmamba_hint(fold: int, cfg: dict, device: torch.device):
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
    model = MambaSequenceRegressorZWithPower(
        embed_dim=embed_dim, n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

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
    video_path_lookup = load_video_path_lookup_z(fold, split_dir=cfg.get("split_dir"))
    save_video_frames_for_fold(
        per_video_df, video_path_lookup=video_path_lookup,
        out_dir=os.path.join(out_dir, "monitoring_plots_frames"),
        max_videos=cfg.get("max_trajectory_plots", 10),
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
