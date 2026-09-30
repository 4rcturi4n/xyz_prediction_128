"""
NEW fused z architecture -- mirrors fused_hint_xy.py exactly, does NOT
touch mamba_z_regression.py. Hint feature u = sqrt(sqrt(P) - 1) (the
axial physics formula's nested-sqrt-minus-one shape, N=2 two-photon
absorption; see scripts/09_baseline_mlp_z.py) -- P_th is never computed.
Baseline MLP's early stopping uses an INTERNAL holdout of TRAIN only,
never the real validation split (no leakage into the fusion feature).

Reuses (imports only): MambaSequenceRegressorZ, evaluate_z_video_only,
compute_loss_z, plot_mae_curve_z, plot_ept_distribution_z,
plot_video_trajectories_z, load_bandwidth_lookup_z,
load_video_power_lookup_z from mamba_z_regression.py.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

from mamba_xy.mamba_z_regression import (
    MambaSequenceRegressorZ,
    evaluate_z_video_only,
    compute_loss_z,
    plot_mae_curve_z,
    plot_video_trajectories_z,
    load_bandwidth_lookup_z,
    load_video_power_lookup_z,
)
from mamba_xy.extra_plots import (
    plot_learning_curve, save_video_frames_for_fold,
    compute_bandwidth_ept, plot_ept_distribution_bandwidth,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLIT_DIR = os.path.join(ROOT, "data", "processed", "kfold_splits_z")


def hint_feature(power_mW: np.ndarray) -> np.ndarray:
    ratio = np.clip(np.sqrt(np.asarray(power_mW, dtype=np.float32)) - 1.0, 1e-6, None)
    return np.sqrt(ratio)


def make_baseline_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


def _fit_mlp(u_train, y_train, u_val, y_val, seed, epochs=3000, patience=300, lr=1e-2, weight_decay=1e-4):
    torch.manual_seed(seed)
    model = make_baseline_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    u_train_t = torch.tensor(u_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    u_val_t = torch.tensor(u_val, dtype=torch.float32).unsqueeze(1)

    best_val_mae, best_state, epochs_no_imp = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(u_train_t), y_train_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(u_val_t).numpy().ravel()
        val_mae = mean_absolute_error(y_val, pred_val)
        if val_mae < best_val_mae - 1e-6:
            best_val_mae, best_state, epochs_no_imp = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_imp += 1
        if epochs_no_imp >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def fit_baseline_mlp_no_leak(power_train: np.ndarray, y_train: np.ndarray, seed: int, holdout_frac: float = 0.2) -> nn.Module:
    rng = np.random.default_rng(seed)
    n = len(power_train)
    idx = rng.permutation(n)
    n_holdout = max(1, int(n * holdout_frac))
    holdout_idx, inner_idx = idx[:n_holdout], idx[n_holdout:]

    u = hint_feature(power_train)
    model = _fit_mlp(u[inner_idx], y_train[inner_idx].astype(np.float32),
                      u[holdout_idx], y_train[holdout_idx].astype(np.float32), seed=seed)
    return model


@torch.no_grad()
def baseline_predict(model: nn.Module, power_mW: np.ndarray) -> np.ndarray:
    u = hint_feature(power_mW)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
    return model(u_t).numpy().ravel()


class CachedMambaEmbeddingDatasetZFusedHint(Dataset):
    def __init__(self, payload_path: str, baseline_model_z: nn.Module, loss_weight_min_eps: float = 1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        embeddings = payload["embeddings"].float()          # [N, T, D]
        self.targets_z = payload["targets_z"].float()
        self.targets_z_phys = payload["targets_z_phys"].float()
        self.errors_z_phys = payload["errors_z"].float()
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()

        self.target_z_mean = float(payload["target_z_mean"])
        self.target_z_std = float(payload["target_z_std"])

        pred_z = baseline_predict(baseline_model_z, self.power_mW.numpy())  # [N]
        pred_z_norm = (pred_z - self.target_z_mean) / self.target_z_std
        hint_feat = torch.tensor(pred_z_norm, dtype=torch.float32).unsqueeze(1)  # [N, 1]

        N, T, D = embeddings.shape
        hint_feat_expanded = hint_feat.unsqueeze(1).expand(N, T, 1)   # [N, T, 1]
        self.embeddings = torch.cat([embeddings, hint_feat_expanded], dim=-1)  # [N, T, D+1]

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
        return emb, target_norm, target_phys, loss_weight, torch.tensor(0.0), str(video_id)


def load_video_path_lookup_z(fold: int, split_dir: str = None) -> dict:
    val_csv = os.path.join(split_dir or SPLIT_DIR, f"fold_{fold}_val.csv")
    df = pd.read_csv(val_csv)
    return dict(zip(df["video_id"], df["video_path"].apply(lambda p: os.path.join(ROOT, p))))


def train_cached_mamba_fold_z_fused_hint(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    z_train = train_payload_raw["targets_z_phys"].numpy()

    baseline_model_z = fit_baseline_mlp_no_leak(power_train, z_train, seed=cfg["seed"] + fold)
    print(f"Fold {fold} | baseline MLP fit (TRAIN-internal-holdout only, no val leakage)")

    train_dataset = CachedMambaEmbeddingDatasetZFusedHint(train_path, baseline_model_z)
    val_dataset = CachedMambaEmbeddingDatasetZFusedHint(val_path, baseline_model_z)

    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    embed_dim_aug = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressorZ(
        embed_dim=embed_dim_aug,
        n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()

    target_z_mean, target_z_std = train_dataset.target_z_mean, train_dataset.target_z_std

    best_val_mae, best_epoch, epochs_no_imp, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for emb, targets_norm, targets_phys, loss_weight, _, video_ids in train_loader:
            emb, targets_norm = emb.to(device), targets_norm.to(device)
            optimizer.zero_grad()
            preds = model(emb)
            loss = compute_loss_z(preds, targets_norm, loss_fn_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_z_video_only(
            model=model, loader=val_loader, device=device,
            target_z_mean=target_z_mean, target_z_std=target_z_std,
            ept_threshold_um=cfg["ept_threshold_um"],
        )
        current_mae = val_metrics["final_mae_z_phys"]
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae, best_epoch, epochs_no_imp = current_mae, epoch, 0
            torch.save({
                "fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg,
                "target_z_mean": target_z_mean, "target_z_std": target_z_std,
                "best_val_mae": best_val_mae,
            }, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({
            "fold": fold, "epoch": epoch, "train_loss": train_loss,
            **val_metrics, "val_mae": current_mae,
            "best_val_mae_so_far": best_val_mae, "improved": improved,
            "epochs_without_improvement": epochs_no_imp,
        })
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

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_z_video_only(
        model=model, loader=val_loader, device=device,
        target_z_mean=target_z_mean, target_z_std=target_z_std,
        ept_threshold_um=cfg["ept_threshold_um"],
    )
    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
    ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)

    plot_mae_curve_z(mae_per_t_df, os.path.join(out_dir, "mae_curve.png"), fold)

    z_bw_lookup = load_bandwidth_lookup_z()
    video_power_lookup = load_video_power_lookup_z(fold)

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

    video_path_lookup = load_video_path_lookup_z(fold)
    save_video_frames_for_fold(
        per_video_df, video_path_lookup=video_path_lookup,
        out_dir=os.path.join(out_dir, "monitoring_plots_frames"),
        max_videos=cfg.get("max_trajectory_plots", 10),
    )

    fold_summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae_z_phys": float(best_val_metrics["final_mae_z_phys"]),
        "best_val_mae_std_z_phys": float(best_val_metrics["final_mae_std_z_phys"]),
        "best_val_rmse_z_phys": float(best_val_metrics["final_rmse_z_phys"]),
        "best_val_bias_z_phys": float(best_val_metrics["final_bias_z_phys"]),
        "mean_ept_pct": float(best_val_metrics["mean_ept_pct"]),
        "pct_videos_ept_found": float(best_val_metrics["pct_videos_ept_found"]),
        "num_train_rows": len(train_dataset),
        "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary
