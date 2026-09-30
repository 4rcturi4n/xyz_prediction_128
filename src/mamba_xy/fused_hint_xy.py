"""
NEW fused x/y architecture -- does NOT touch mamba_xy_regression.py.

Instead of feeding a power-derived scalar feature into the fusion branch
at the head (which we found hurts optimization for the true physics
feature, and isn't "physics" at all for sqrt(power)), this predicts x
and y from power ALONE first (a small ReLU-MLP baseline, hint feature
u=sqrt(ln(P)) -- the lateral physics formula's shape, but with no P_th
committed anywhere, see scripts/09_baseline_mlp_xy.py), then concatenates
those two baseline predictions into the per-timestep video embeddings
BEFORE the Mamba layers (not at the head) -- so the video branch's whole
job is to learn a correction on top of a physics-shaped prior, with full
freedom, rather than to consume a black-box scalar.

I_th is never computed. The baseline MLP's early stopping uses an
INTERNAL holdout carved out of the fold's TRAIN split only -- never the
real validation split -- so the fusion feature going into Mamba training
is not informed by validation labels in any way (that would leak val
information into an input feature of the val-evaluated model).

Reuses (imports only, never modifies): MambaSequenceRegressorXY,
evaluate_xy_video_only, compute_loss_xy, plot_mae_curve_xy,
plot_ept_distribution_xy, plot_video_trajectories_xy,
load_bandwidth_lookup_xy, load_video_power_lookup_xy from
mamba_xy_regression.py -- since, once the baseline predictions are
concatenated into the embeddings, this is architecturally identical to
the video-only model (single embedding tensor in, no separate power
argument), just with 2 extra input channels.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

from mamba_xy.mamba_xy_regression import (
    MambaSequenceRegressorXY,
    evaluate_xy_video_only,
    compute_loss_xy,
    plot_mae_curve_xy,
    plot_video_trajectories_xy,
    load_bandwidth_lookup_xy,
    load_video_power_lookup_xy,
)
from mamba_xy.extra_plots import (
    plot_learning_curve, save_video_frames_for_fold,
    compute_bandwidth_ept, plot_ept_distribution_bandwidth,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLIT_DIR = os.path.join(ROOT, "data", "processed", "kfold_splits_xy")


def hint_feature(power_mW: np.ndarray) -> np.ndarray:
    return np.sqrt(np.log(np.asarray(power_mW, dtype=np.float32)))


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
    """
    Fits the baseline MLP with early stopping against an INTERNAL holdout
    carved out of TRAIN only -- the real validation split is never touched
    here, so the resulting model's predictions (used as a Mamba input
    feature) carry no information from validation labels.
    """
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


# --------------------------------------------------
# Dataset: baseline predictions concatenated into embeddings, pre-Mamba
# --------------------------------------------------

class CachedMambaEmbeddingDatasetXYFusedHint(Dataset):
    def __init__(self, payload_path: str, baseline_model_x: nn.Module, baseline_model_y: nn.Module,
                 loss_weight_min_eps: float = 1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        embeddings = payload["embeddings"].float()          # [N, T, D]
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

        pred_x = baseline_predict(baseline_model_x, self.power_mW.numpy())  # [N]
        pred_y = baseline_predict(baseline_model_y, self.power_mW.numpy())  # [N]
        # normalize baseline predictions to roughly unit scale for stable training,
        # using the same target mean/std as the real targets (same physical units)
        pred_x_norm = (pred_x - self.target_x_mean) / self.target_x_std
        pred_y_norm = (pred_y - self.target_y_mean) / self.target_y_std
        hint_feat = torch.tensor(np.stack([pred_x_norm, pred_y_norm], axis=1), dtype=torch.float32)  # [N, 2]

        N, T, D = embeddings.shape
        hint_feat_expanded = hint_feat.unsqueeze(1).expand(N, T, 2)   # [N, T, 2]
        self.embeddings = torch.cat([embeddings, hint_feat_expanded], dim=-1)  # [N, T, D+2]

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
        # power_norm placeholder kept only for tuple-shape compatibility with evaluate_xy_video_only
        return emb, target_norm, target_phys, loss_weight, torch.tensor(0.0), str(video_id)


def load_video_path_lookup_xy(fold: int, split_dir: str = None) -> dict:
    val_csv = os.path.join(split_dir or SPLIT_DIR, f"fold_{fold}_val.csv")
    df = pd.read_csv(val_csv)
    return dict(zip(df["video_id"], df["video_path"]))


def train_cached_mamba_fold_xy_fused_hint(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    x_train = train_payload_raw["targets_x_phys"].numpy()
    y_train = train_payload_raw["targets_y_phys"].numpy()

    baseline_model_x = fit_baseline_mlp_no_leak(power_train, x_train, seed=cfg["seed"] + fold)
    baseline_model_y = fit_baseline_mlp_no_leak(power_train, y_train, seed=cfg["seed"] + fold + 1000)
    print(f"Fold {fold} | baseline MLPs fit (TRAIN-internal-holdout only, no val leakage)")

    train_dataset = CachedMambaEmbeddingDatasetXYFusedHint(train_path, baseline_model_x, baseline_model_y)
    val_dataset = CachedMambaEmbeddingDatasetXYFusedHint(val_path, baseline_model_x, baseline_model_y)

    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    embed_dim_aug = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressorXY(
        embed_dim=embed_dim_aug,
        n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()

    target_x_mean, target_x_std = train_dataset.target_x_mean, train_dataset.target_x_std
    target_y_mean, target_y_std = train_dataset.target_y_mean, train_dataset.target_y_std

    best_val_mae, best_epoch, epochs_no_imp, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for emb, targets_norm, targets_phys, loss_weight, _, video_ids in train_loader:
            emb, targets_norm = emb.to(device), targets_norm.to(device)
            optimizer.zero_grad()
            preds = model(emb)
            loss = compute_loss_xy(preds, targets_norm, loss_fn_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_xy_video_only(
            model=model, loader=val_loader, device=device,
            target_x_mean=target_x_mean, target_x_std=target_x_std,
            target_y_mean=target_y_mean, target_y_std=target_y_std,
            ept_threshold_um=cfg["ept_threshold_um"],
        )
        current_mae = (val_metrics["final_mae_x_phys"] + val_metrics["final_mae_y_phys"]) / 2.0
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae, best_epoch, epochs_no_imp = current_mae, epoch, 0
            torch.save({
                "fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg,
                "target_x_mean": target_x_mean, "target_x_std": target_x_std,
                "target_y_mean": target_y_mean, "target_y_std": target_y_std,
                "best_val_mae": best_val_mae,
            }, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({
            "fold": fold, "epoch": epoch, "train_loss": train_loss,
            **val_metrics, "val_mae_combined": current_mae,
            "best_val_mae_so_far": best_val_mae, "improved": improved,
            "epochs_without_improvement": epochs_no_imp,
        })
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

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_xy_video_only(
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
    video_power_lookup = load_video_power_lookup_xy(fold)

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

    video_path_lookup = load_video_path_lookup_xy(fold)
    save_video_frames_for_fold(
        per_video_df, video_path_lookup=video_path_lookup,
        out_dir=os.path.join(out_dir, "monitoring_plots_frames"),
        max_videos=cfg.get("max_trajectory_plots", 10),
    )

    fold_summary = {
        "fold": fold,
        "best_epoch": best_epoch,
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
        "num_train_rows": len(train_dataset),
        "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary
