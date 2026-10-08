"""
Residual-physics ablation: the video model (same Mamba trunk as the
video-only models -- no hint/power ever passed to the model as input)
is trained to predict the RESIDUAL left over after the pure physics fit
(xy: w0*sqrt(2*ln(P/I_th)); z: zr*sqrt(sqrt(ln(P/I_th))-1)+b), not the
raw target. Physics params fit per fold, TRAIN-only, leak-free, same
convention as everywhere else in this repo.

At evaluation, final_prediction = physics_prediction + model's residual
prediction, and every reported metric (MAE, bias, EPT, per-timestep
curves) is computed against the TRUE target using that combined
prediction -- not the raw residual-prediction error.

Reuses (imports only): MambaSequenceRegressorXY/Z (no power passed to
model -- same convention as video-only), compute_loss_xy/z,
plot_mae_curve_xy/z, plot_video_trajectories_xy/z,
load_bandwidth_lookup_xy/z, load_video_power_lookup_xy/z, plotting from
extra_plots.py. Does not modify any existing file.
"""

import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset

import json

import torch.nn as nn
from torch.utils.data import DataLoader

from mamba_xy.mamba_xy_regression import (
    MambaSequenceRegressorXY, compute_loss_xy, plot_mae_curve_xy, plot_video_trajectories_xy,
    load_bandwidth_lookup_xy, load_video_power_lookup_xy,
)
from mamba_xy.mamba_z_regression import (
    MambaSequenceRegressorZ, N_ORDER, compute_loss_z, plot_mae_curve_z, plot_video_trajectories_z,
    load_bandwidth_lookup_z, load_video_power_lookup_z,
)
from mamba_xy.core import inverse_transform
from mamba_xy.extra_plots import plot_learning_curve


# --------------------------------------------------
# Physics fits (same equations as scripts/15 and scripts/20)
# --------------------------------------------------

def lateral_model(P, w0, i_th):
    ratio = np.clip(P / i_th, 1.0 + 1e-9, None)
    return w0 * np.sqrt(2.0 * np.log(ratio))


def fit_lateral(power_mW, target_phys):
    p_min = float(power_mW.min())
    w0_0 = float(target_phys.max() - target_phys.min())
    ith0 = 0.5 * p_min
    bounds = ([-np.inf, 1e-3], [np.inf, p_min * 0.999])
    popt, _ = curve_fit(lateral_model, power_mW, target_phys, p0=[w0_0, ith0], bounds=bounds, maxfev=20000)
    return popt  # w0, i_th


def axial_model_newhint(P, zr, i_th, b):
    log_ratio = np.clip(np.log(np.clip(P / i_th, 1e-9, None)), 1e-9, None)
    inner = np.clip(np.sqrt(log_ratio) - 1.0, 1e-6, None)
    return zr * np.sqrt(inner) + b


def fit_axial_newhint(power_mW, target_phys):
    p_min = float(power_mW.min())
    zr0 = float(target_phys.max() - target_phys.min())
    b0 = float(target_phys.min())
    ith0 = p_min / np.e / 2.0
    bounds = ([-np.inf, 1e-3, -np.inf], [np.inf, p_min / np.e * 0.999, np.inf])
    popt, _ = curve_fit(axial_model_newhint, power_mW, target_phys, p0=[zr0, ith0, b0], bounds=bounds, maxfev=20000)
    return popt  # zr, i_th, b


# --------------------------------------------------
# Datasets: model input is ONLY embeddings (no hint), target is the
# residual (true - physics_pred), physics_pred kept per-video for
# recombination at eval time.
# --------------------------------------------------

class CachedMambaResidualDatasetXY(Dataset):
    def __init__(self, payload_path, w0_params, residual_mean=None, residual_std=None, loss_weight_min_eps=1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        self.embeddings = payload["embeddings"].float()
        self.targets_x_phys = payload["targets_x_phys"].float()
        self.targets_y_phys = payload["targets_y_phys"].float()
        self.errors_x_phys = payload["errors_x"].float()
        self.errors_y_phys = payload["errors_y"].float()
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()

        w0_x, ith_x, w0_y, ith_y = w0_params
        P = self.power_mW.numpy()
        self.physics_pred_x = torch.tensor(lateral_model(P, w0_x, ith_x), dtype=torch.float32)
        self.physics_pred_y = torch.tensor(lateral_model(P, w0_y, ith_y), dtype=torch.float32)

        res_x = (self.targets_x_phys - self.physics_pred_x).numpy()
        res_y = (self.targets_y_phys - self.physics_pred_y).numpy()
        if residual_mean is None:
            residual_mean = {"x": float(res_x.mean()), "y": float(res_y.mean())}
            residual_std = {"x": float(res_x.std() + 1e-8), "y": float(res_y.std() + 1e-8)}
        self.residual_mean, self.residual_std = residual_mean, residual_std

        self.res_x_norm = torch.tensor((res_x - residual_mean["x"]) / residual_std["x"], dtype=torch.float32)
        self.res_y_norm = torch.tensor((res_y - residual_mean["y"]) / residual_std["y"], dtype=torch.float32)

        self.loss_weight_x = self._compute_loss_weights(self.errors_x_phys, residual_std["x"], loss_weight_min_eps)
        self.loss_weight_y = self._compute_loss_weights(self.errors_y_phys, residual_std["y"], loss_weight_min_eps)

    @staticmethod
    def _compute_loss_weights(errors_phys, scale, eps):
        errors_norm = errors_phys / scale
        raw_weights = 1.0 / errors_norm.clamp_min(eps).pow(2)
        valid = ~torch.isnan(errors_phys)
        mean_valid_weight = raw_weights[valid].mean() if valid.sum() > 0 else torch.tensor(1.0)
        return torch.where(valid, raw_weights / mean_valid_weight, torch.ones_like(raw_weights))

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        emb = self.embeddings[idx].clone()
        target_norm = torch.stack([self.res_x_norm[idx], self.res_y_norm[idx]])
        target_phys = torch.stack([self.targets_x_phys[idx], self.targets_y_phys[idx]])
        loss_weight = torch.stack([self.loss_weight_x[idx], self.loss_weight_y[idx]])
        video_id = self.video_ids[idx]
        return emb, target_norm, target_phys, loss_weight, torch.tensor(0.0), str(video_id)


class CachedMambaResidualDatasetZ(Dataset):
    def __init__(self, payload_path, z_params, residual_mean=None, residual_std=None, loss_weight_min_eps=1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        self.embeddings = payload["embeddings"].float()
        self.targets_z_phys = payload["targets_z_phys"].float()
        self.errors_z_phys = payload["errors_z"].float()
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()

        zr, ith, b = z_params
        P = self.power_mW.numpy()
        self.physics_pred_z = torch.tensor(axial_model_newhint(P, zr, ith, b), dtype=torch.float32)

        res_z = (self.targets_z_phys - self.physics_pred_z).numpy()
        if residual_mean is None:
            residual_mean = float(res_z.mean())
            residual_std = float(res_z.std() + 1e-8)
        self.residual_mean, self.residual_std = residual_mean, residual_std
        self.res_z_norm = torch.tensor((res_z - residual_mean) / residual_std, dtype=torch.float32)

        self.loss_weight_z = self._compute_loss_weights(self.errors_z_phys, residual_std, loss_weight_min_eps)

    @staticmethod
    def _compute_loss_weights(errors_phys, scale, eps):
        errors_norm = errors_phys / scale
        raw_weights = 1.0 / errors_norm.clamp_min(eps).pow(2)
        valid = ~torch.isnan(errors_phys)
        mean_valid_weight = raw_weights[valid].mean() if valid.sum() > 0 else torch.tensor(1.0)
        return torch.where(valid, raw_weights / mean_valid_weight, torch.ones_like(raw_weights))

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        emb = self.embeddings[idx].clone()
        target_norm = self.res_z_norm[idx].unsqueeze(0)
        target_phys = self.targets_z_phys[idx].unsqueeze(0)
        loss_weight = self.loss_weight_z[idx].unsqueeze(0)
        video_id = self.video_ids[idx]
        return emb, target_norm, target_phys, loss_weight, torch.tensor(0.0), str(video_id)


# --------------------------------------------------
# Custom evaluate: denormalize model output as a RESIDUAL, add back the
# per-video physics prediction, then compute every metric against the
# TRUE target using that combined (physics + residual) prediction.
# --------------------------------------------------

def evaluate_xy_residual(model, loader, device, residual_mean, residual_std,
                          physics_pred_x_lookup, physics_pred_y_lookup, ept_threshold_um):
    model.eval()
    all_preds_norm, all_targets_phys, all_video_ids = [], [], []

    with torch.no_grad():
        for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in loader:
            emb = emb.to(device)
            preds = model(emb)  # [B, T, 2], predicted RESIDUAL (normalized)
            all_preds_norm.append(preds.cpu().numpy())
            all_targets_phys.append(targets_phys.numpy())
            all_video_ids.extend(list(video_ids))

    preds_norm_arr = np.concatenate(all_preds_norm, axis=0)       # [N, T, 2]
    targets_phys_arr = np.concatenate(all_targets_phys, axis=0)   # [N, 2] -- TRUE x/y
    N, T, _ = preds_norm_arr.shape

    preds_phys_arr = np.zeros_like(preds_norm_arr)
    for t in range(T):
        preds_phys_arr[:, t, 0] = inverse_transform(preds_norm_arr[:, t, 0], residual_mean["x"], residual_std["x"])
        preds_phys_arr[:, t, 1] = inverse_transform(preds_norm_arr[:, t, 1], residual_mean["y"], residual_std["y"])
    # add back the physics prediction -- same scalar at every timestep for that video
    for i, vid in enumerate(all_video_ids):
        preds_phys_arr[i, :, 0] += physics_pred_x_lookup[vid]
        preds_phys_arr[i, :, 1] += physics_pred_y_lookup[vid]

    frame_pct = np.linspace(0.0, 100.0, T)
    mae_x_per_t = [mean_absolute_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]) for t in range(T)]
    mae_y_per_t = [mean_absolute_error(targets_phys_arr[:, 1], preds_phys_arr[:, t, 1]) for t in range(T)]
    rmse_x_per_t = [float(np.sqrt(mean_squared_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]))) for t in range(T)]
    rmse_y_per_t = [float(np.sqrt(mean_squared_error(targets_phys_arr[:, 1], preds_phys_arr[:, t, 1]))) for t in range(T)]

    mae_per_t_df = pd.DataFrame({
        "frame_idx": np.arange(T), "frame_pct": frame_pct,
        "mae_x_phys": mae_x_per_t, "rmse_x_phys": rmse_x_per_t,
        "mae_y_phys": mae_y_per_t, "rmse_y_phys": rmse_y_per_t,
    })

    final_mae_x, final_mae_y = float(mae_x_per_t[-1]), float(mae_y_per_t[-1])
    final_mae_std_x = float(np.std(np.abs(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0])))
    final_mae_std_y = float(np.std(np.abs(preds_phys_arr[:, -1, 1] - targets_phys_arr[:, 1])))
    final_rmse_x, final_rmse_y = float(rmse_x_per_t[-1]), float(rmse_y_per_t[-1])
    final_bias_x = float(np.mean(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0]))
    final_bias_y = float(np.mean(preds_phys_arr[:, -1, 1] - targets_phys_arr[:, 1]))

    ept_pcts, ept_found = [], []
    for i in range(N):
        err_x = np.abs(preds_phys_arr[i, :, 0] - targets_phys_arr[i, 0])
        err_y = np.abs(preds_phys_arr[i, :, 1] - targets_phys_arr[i, 1])
        combined_err = np.maximum(err_x, err_y)
        ept_frame = None
        for t in range(T):
            if np.all(combined_err[t:] < ept_threshold_um):
                ept_frame = t
                break
        if ept_frame is not None:
            ept_pcts.append(float(frame_pct[ept_frame])); ept_found.append(True)
        else:
            ept_pcts.append(100.0); ept_found.append(False)

    rows = []
    for i, vid_id in enumerate(all_video_ids):
        for t in range(T):
            rows.append({
                "video_id": vid_id, "frame_idx": t, "frame_pct": float(frame_pct[t]),
                "pred_x_phys": float(preds_phys_arr[i, t, 0]), "true_x_phys": float(targets_phys_arr[i, 0]),
                "pred_y_phys": float(preds_phys_arr[i, t, 1]), "true_y_phys": float(targets_phys_arr[i, 1]),
                "abs_err_x_phys": float(abs(preds_phys_arr[i, t, 0] - targets_phys_arr[i, 0])),
                "abs_err_y_phys": float(abs(preds_phys_arr[i, t, 1] - targets_phys_arr[i, 1])),
                "ept_pct": float(ept_pcts[i]), "ept_found": bool(ept_found[i]),
            })
    per_video_df = pd.DataFrame(rows)
    ept_df = pd.DataFrame({
        "video_id": all_video_ids,
        "true_x_phys": targets_phys_arr[:, 0], "final_pred_x_phys": preds_phys_arr[:, -1, 0],
        "final_abs_err_x": np.abs(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0]),
        "true_y_phys": targets_phys_arr[:, 1], "final_pred_y_phys": preds_phys_arr[:, -1, 1],
        "final_abs_err_y": np.abs(preds_phys_arr[:, -1, 1] - targets_phys_arr[:, 1]),
        "ept_pct": ept_pcts, "ept_found": ept_found,
    })
    metrics = {
        "final_mae_x_phys": final_mae_x, "final_mae_std_x_phys": final_mae_std_x,
        "final_rmse_x_phys": final_rmse_x, "final_bias_x_phys": final_bias_x,
        "final_mae_y_phys": final_mae_y, "final_mae_std_y_phys": final_mae_std_y,
        "final_rmse_y_phys": final_rmse_y, "final_bias_y_phys": final_bias_y,
        "mean_ept_pct": float(np.mean(ept_pcts)), "pct_videos_ept_found": float(np.mean(ept_found) * 100.0),
        "ept_threshold_um": float(ept_threshold_um), "n_videos": N, "n_frames": T,
    }
    return metrics, per_video_df, mae_per_t_df, ept_df


def evaluate_z_residual(model, loader, device, residual_mean, residual_std, physics_pred_lookup, ept_threshold_um):
    model.eval()
    all_preds_norm, all_targets_phys, all_video_ids = [], [], []

    with torch.no_grad():
        for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in loader:
            emb = emb.to(device)
            preds = model(emb)  # [B, T, 1]
            all_preds_norm.append(preds.cpu().numpy())
            all_targets_phys.append(targets_phys.numpy())
            all_video_ids.extend(list(video_ids))

    preds_norm_arr = np.concatenate(all_preds_norm, axis=0)
    targets_phys_arr = np.concatenate(all_targets_phys, axis=0)
    N, T, _ = preds_norm_arr.shape

    preds_phys_arr = np.zeros_like(preds_norm_arr)
    for t in range(T):
        preds_phys_arr[:, t, 0] = inverse_transform(preds_norm_arr[:, t, 0], residual_mean, residual_std)
    for i, vid in enumerate(all_video_ids):
        preds_phys_arr[i, :, 0] += physics_pred_lookup[vid]

    frame_pct = np.linspace(0.0, 100.0, T)
    mae_z_per_t = [mean_absolute_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]) for t in range(T)]
    rmse_z_per_t = [float(np.sqrt(mean_squared_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]))) for t in range(T)]
    mae_per_t_df = pd.DataFrame({
        "frame_idx": np.arange(T), "frame_pct": frame_pct, "mae_z_phys": mae_z_per_t, "rmse_z_phys": rmse_z_per_t,
    })

    final_mae_z = float(mae_z_per_t[-1])
    final_mae_std_z = float(np.std(np.abs(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0])))
    final_rmse_z = float(rmse_z_per_t[-1])
    final_bias_z = float(np.mean(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0]))

    ept_pcts, ept_found = [], []
    for i in range(N):
        err_z = np.abs(preds_phys_arr[i, :, 0] - targets_phys_arr[i, 0])
        ept_frame = None
        for t in range(T):
            if np.all(err_z[t:] < ept_threshold_um):
                ept_frame = t
                break
        if ept_frame is not None:
            ept_pcts.append(float(frame_pct[ept_frame])); ept_found.append(True)
        else:
            ept_pcts.append(100.0); ept_found.append(False)

    rows = []
    for i, vid_id in enumerate(all_video_ids):
        for t in range(T):
            rows.append({
                "video_id": vid_id, "frame_idx": t, "frame_pct": float(frame_pct[t]),
                "pred_z_phys": float(preds_phys_arr[i, t, 0]), "true_z_phys": float(targets_phys_arr[i, 0]),
                "abs_err_z_phys": float(abs(preds_phys_arr[i, t, 0] - targets_phys_arr[i, 0])),
                "ept_pct": float(ept_pcts[i]), "ept_found": bool(ept_found[i]),
            })
    per_video_df = pd.DataFrame(rows)
    ept_df = pd.DataFrame({
        "video_id": all_video_ids,
        "true_z_phys": targets_phys_arr[:, 0], "final_pred_z_phys": preds_phys_arr[:, -1, 0],
        "final_abs_err_z": np.abs(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0]),
        "ept_pct": ept_pcts, "ept_found": ept_found,
    })
    metrics = {
        "final_mae_z_phys": final_mae_z, "final_mae_std_z_phys": final_mae_std_z,
        "final_rmse_z_phys": final_rmse_z, "final_bias_z_phys": final_bias_z,
        "mean_ept_pct": float(np.mean(ept_pcts)), "pct_videos_ept_found": float(np.mean(ept_found) * 100.0),
        "ept_threshold_um": float(ept_threshold_um), "n_videos": N, "n_frames": T,
    }
    return metrics, per_video_df, mae_per_t_df, ept_df


# --------------------------------------------------
# Training (same loop structure as the video-only trainers, just a
# different dataset/evaluate pair)
# --------------------------------------------------

def train_cached_mamba_fold_xy_residual(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    w0_x, ith_x = fit_lateral(power_train, train_payload_raw["targets_x_phys"].numpy())
    w0_y, ith_y = fit_lateral(power_train, train_payload_raw["targets_y_phys"].numpy())
    print(f"Fold {fold} | physics fit (TRAIN only): w0_x={w0_x:.4f} I_th_x={ith_x:.4f}  "
          f"w0_y={w0_y:.4f} I_th_y={ith_y:.4f}")
    w0_params = (w0_x, ith_x, w0_y, ith_y)

    train_dataset = CachedMambaResidualDatasetXY(train_path, w0_params)
    val_dataset = CachedMambaResidualDatasetXY(
        val_path, w0_params, residual_mean=train_dataset.residual_mean, residual_std=train_dataset.residual_std,
    )
    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    physics_pred_x_lookup = {v: float(p) for v, p in zip(val_dataset.video_ids, val_dataset.physics_pred_x)}
    physics_pred_y_lookup = {v: float(p) for v, p in zip(val_dataset.video_ids, val_dataset.physics_pred_y)}

    embed_dim = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressorXY(
        embed_dim=embed_dim, n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()
    residual_mean, residual_std = train_dataset.residual_mean, train_dataset.residual_std

    best_val_mae, best_epoch, epochs_no_imp, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in train_loader:
            emb, targets_norm = emb.to(device), targets_norm.to(device)
            optimizer.zero_grad()
            preds = model(emb)
            loss = compute_loss_xy(preds, targets_norm, loss_fn_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_xy_residual(
            model=model, loader=val_loader, device=device,
            residual_mean=residual_mean, residual_std=residual_std,
            physics_pred_x_lookup=physics_pred_x_lookup, physics_pred_y_lookup=physics_pred_y_lookup,
            ept_threshold_um=cfg["ept_threshold_um"],
        )
        current_mae = (val_metrics["final_mae_x_phys"] + val_metrics["final_mae_y_phys"]) / 2.0
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae, best_epoch, epochs_no_imp = current_mae, epoch, 0
            torch.save({"fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg,
                        "residual_mean": residual_mean, "residual_std": residual_std,
                        "best_val_mae": best_val_mae}, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, **val_metrics,
                         "improved": improved, "epochs_without_improvement": epochs_no_imp})
        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"mae_x={val_metrics['final_mae_x_phys']:.5f} mae_y={val_metrics['final_mae_y_phys']:.5f} | "
              f"no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}")

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    plot_learning_curve(history_df, os.path.join(out_dir, "learning_curve.png"), fold,
                         val_cols=[("final_mae_x_phys", "val_mae_x"), ("final_mae_y_phys", "val_mae_y")])

    ckpt = torch.load(os.path.join(out_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_xy_residual(
        model=model, loader=val_loader, device=device,
        residual_mean=residual_mean, residual_std=residual_std,
        physics_pred_x_lookup=physics_pred_x_lookup, physics_pred_y_lookup=physics_pred_y_lookup,
        ept_threshold_um=cfg["ept_threshold_um"],
    )
    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
    plot_mae_curve_xy(mae_per_t_df, os.path.join(out_dir, "mae_curve.png"), fold)

    x_bw_lookup, y_bw_lookup = load_bandwidth_lookup_xy()
    video_power_lookup = load_video_power_lookup_xy(fold, split_dir=cfg.get("split_dir"))
    plot_video_trajectories_xy(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, x_bandwidth_lookup=x_bw_lookup, y_bandwidth_lookup=y_bw_lookup,
    )

    fold_summary = {
        "fold": fold, "best_epoch": best_epoch,
        "w0_x": float(w0_x), "i_th_x_mW": float(ith_x), "w0_y": float(w0_y), "i_th_y_mW": float(ith_y),
        "best_val_mae_x_phys": float(best_val_metrics["final_mae_x_phys"]),
        "best_val_mae_y_phys": float(best_val_metrics["final_mae_y_phys"]),
        "best_val_mae_std_x_phys": float(best_val_metrics["final_mae_std_x_phys"]),
        "best_val_mae_std_y_phys": float(best_val_metrics["final_mae_std_y_phys"]),
        "best_val_rmse_x_phys": float(best_val_metrics["final_rmse_x_phys"]),
        "best_val_rmse_y_phys": float(best_val_metrics["final_rmse_y_phys"]),
        "best_val_bias_x_phys": float(best_val_metrics["final_bias_x_phys"]),
        "best_val_bias_y_phys": float(best_val_metrics["final_bias_y_phys"]),
        "num_train_rows": len(train_dataset), "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)
    return history_df, fold_summary


def train_cached_mamba_fold_z_residual(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    zr, ith, b = fit_axial_newhint(power_train, train_payload_raw["targets_z_phys"].numpy())
    print(f"Fold {fold} | physics fit (TRAIN only): zr={zr:.4f} I_th={ith:.4f} b={b:.4f}")
    z_params = (zr, ith, b)

    train_dataset = CachedMambaResidualDatasetZ(train_path, z_params)
    val_dataset = CachedMambaResidualDatasetZ(
        val_path, z_params, residual_mean=train_dataset.residual_mean, residual_std=train_dataset.residual_std,
    )
    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    physics_pred_lookup = {v: float(p) for v, p in zip(val_dataset.video_ids, val_dataset.physics_pred_z)}

    embed_dim = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressorZ(
        embed_dim=embed_dim, n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()
    residual_mean, residual_std = train_dataset.residual_mean, train_dataset.residual_std

    best_val_mae, best_epoch, epochs_no_imp, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in train_loader:
            emb, targets_norm = emb.to(device), targets_norm.to(device)
            optimizer.zero_grad()
            preds = model(emb)
            loss = compute_loss_z(preds, targets_norm, loss_fn_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_z_residual(
            model=model, loader=val_loader, device=device,
            residual_mean=residual_mean, residual_std=residual_std,
            physics_pred_lookup=physics_pred_lookup, ept_threshold_um=cfg["ept_threshold_um"],
        )
        current_mae = val_metrics["final_mae_z_phys"]
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae, best_epoch, epochs_no_imp = current_mae, epoch, 0
            torch.save({"fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg,
                        "residual_mean": residual_mean, "residual_std": residual_std,
                        "best_val_mae": best_val_mae}, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, **val_metrics,
                         "improved": improved, "epochs_without_improvement": epochs_no_imp})
        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"mae_z={val_metrics['final_mae_z_phys']:.5f} | no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}")

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    plot_learning_curve(history_df, os.path.join(out_dir, "learning_curve.png"), fold,
                         val_cols=[("final_mae_z_phys", "val_mae_z")])

    ckpt = torch.load(os.path.join(out_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_z_residual(
        model=model, loader=val_loader, device=device,
        residual_mean=residual_mean, residual_std=residual_std,
        physics_pred_lookup=physics_pred_lookup, ept_threshold_um=cfg["ept_threshold_um"],
    )
    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
    plot_mae_curve_z(mae_per_t_df, os.path.join(out_dir, "mae_curve.png"), fold)

    z_bw_lookup = load_bandwidth_lookup_z()
    video_power_lookup = load_video_power_lookup_z(fold, split_dir=cfg.get("split_dir"))
    plot_video_trajectories_z(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, bandwidth_lookup=z_bw_lookup,
    )

    fold_summary = {
        "fold": fold, "best_epoch": best_epoch, "zr": float(zr), "i_th_mW": float(ith), "b": float(b),
        "best_val_mae_z_phys": float(best_val_metrics["final_mae_z_phys"]),
        "best_val_mae_std_z_phys": float(best_val_metrics["final_mae_std_z_phys"]),
        "best_val_rmse_z_phys": float(best_val_metrics["final_rmse_z_phys"]),
        "best_val_bias_z_phys": float(best_val_metrics["final_bias_z_phys"]),
        "num_train_rows": len(train_dataset), "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)
    return history_df, fold_summary
