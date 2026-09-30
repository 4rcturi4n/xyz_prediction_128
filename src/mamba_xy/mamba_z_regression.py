"""
z (axial resolution) regression: DINOv2 + Mamba + power fusion, single
target. Reduced from a 2-output (x, y) head to a 1-output (z) head
throughout -- same plotting/monitoring outputs as the x/y pipeline.

Settled config: flat per-timestep loss weighting, unweighted (no
precision-weighting). No more toggles -- this is the one fused z model.

Power fusion feature: sqrt((P/I_th)^(1/N) - 1), N=2 -- the actual
physics-derived AXIAL threshold-dose quantity (see
scripts/08_baseline_physics_z.py), a genuinely different functional form
from the lateral (x/y) feature, not the same shape with different
constants. I_th is NOT a fixed global constant -- it's refit per fold
from that fold's own training split only (fit_i_th_axial below, with
intercept, matching the settled z baseline method), the same way target
mean/std and every other fold-specific statistic in this pipeline are
computed, so there's no leakage from validation data into the feature.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mamba_xy.core import build_mamba_layer, inverse_transform

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BANDWIDTH_DIR = os.path.join(ROOT, "results", "ept_bandwidth")
SPLIT_DIR = os.path.join(ROOT, "data", "processed", "kfold_splits_z")


def load_bandwidth_lookup_z():
    """power_mW -> bandwidth_um (see scripts/12_compute_ept_bandwidths.py)."""
    return pd.read_csv(os.path.join(BANDWIDTH_DIR, "bandwidth_z.csv")).set_index("power_mW")["bandwidth_um"]


def load_video_power_lookup_z(fold: int, split_dir: str = None) -> dict:
    val_csv = os.path.join(split_dir or SPLIT_DIR, f"fold_{fold}_val.csv")
    df = pd.read_csv(val_csv)
    return dict(zip(df["video_id"], df["power_mW"]))


N_ORDER = 2  # two-photon absorption


def axial_power_feature(power_mW: torch.Tensor, i_th: float) -> torch.Tensor:
    """sqrt((P/I_th)^(1/N) - 1) -- the physics-derived axial threshold-dose
    feature, same functional form as the fitted z baseline. i_th must be
    fit per fold from TRAIN data only -- see fit_i_th_axial."""
    ratio = torch.clamp((power_mW / i_th) ** (1.0 / N_ORDER), min=1.0 + 1e-6)
    return torch.sqrt(ratio - 1.0)


def fit_i_th_axial(power_mW: np.ndarray, target_phys: np.ndarray) -> float:
    """Fits 2*zR*sqrt((P/I_th)^(1/N)-1) + b (with intercept, see
    08_baseline_physics_z.py) via scipy curve_fit and returns just I_th.
    Called once per fold on that fold's TRAIN split only -- never on
    validation data, to avoid leaking fold-held-out information into the
    fusion feature."""
    from scipy.optimize import curve_fit

    p_min = float(power_mW.min())
    zr0 = float(target_phys.max() - target_phys.min())
    b0 = float(target_phys.min())
    ith0 = 0.5 * p_min

    def model(P, zr, ith, b):
        ratio = np.clip((P / ith) ** (1.0 / N_ORDER), 1.0 + 1e-9, None)
        return zr * np.sqrt(ratio - 1.0) + b

    bounds = ([-np.inf, 1e-3, -np.inf], [np.inf, p_min * 0.999, np.inf])
    popt, _ = curve_fit(model, power_mW, target_phys, p0=[zr0, ith0, b0], bounds=bounds, maxfev=20000)
    return float(popt[1])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class CachedMambaEmbeddingDatasetZ(Dataset):
    """
    Loads one fold's cached embeddings .pt (from extract_dinov2_embeddings_z.py,
    enriched with power by add_power_to_embeddings_z.py).

    __getitem__ returns:
        emb          [T, D]
        target_norm  scalar (z), normalized using this payload's baked-in
                     train-fold stats
        target_phys  scalar (z), physical microns
        loss_weight  scalar precomputed 1/sigma^2 weight, normalized to
                     mean=1 across this dataset. Computed but unused
                     (settled config is unweighted); kept for reference.
        power_norm   scalar, normalized sqrt((power_mW/I_th)^(1/N)-1)
        video_id     str
    """

    def __init__(self, payload_path: str, i_th: float, power_mean: float = None, power_std: float = None,
                 loss_weight_min_eps: float = 1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        self.embeddings = payload["embeddings"].float()             # [N, T, D]
        self.targets_z = payload["targets_z"].float()               # [N] normalized
        self.targets_z_phys = payload["targets_z_phys"].float()     # [N] physical
        self.errors_z_phys = payload["errors_z"].float()            # [N] physical std dev, may be NaN
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()                 # [N]

        self.target_z_mean = float(payload["target_z_mean"])
        self.target_z_std = float(payload["target_z_std"])
        self.i_th = i_th

        power_feat_raw = axial_power_feature(self.power_mW, self.i_th)
        self.power_mean = float(power_feat_raw.mean()) if power_mean is None else power_mean
        self.power_std = float(power_feat_raw.std(unbiased=False) + 1e-8) if power_std is None else power_std

        self.loss_weight_z = self._compute_loss_weights(self.errors_z_phys, self.target_z_std, loss_weight_min_eps)

    @staticmethod
    def _compute_loss_weights(errors_phys: torch.Tensor, target_std: float, eps: float) -> torch.Tensor:
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

        power_feat_raw = float(axial_power_feature(self.power_mW[idx], self.i_th))
        power_norm = (power_feat_raw - self.power_mean) / self.power_std

        video_id = self.video_ids[idx]

        return (
            emb,
            target_norm,
            target_phys,
            loss_weight,
            torch.tensor(power_norm, dtype=torch.float32),
            str(video_id),
        )


# --------------------------------------------------
# Model: same Mamba trunk as the x/y model, 1-output head
# --------------------------------------------------

class MambaSequenceRegressorZWithPower(nn.Module):
    def __init__(
        self,
        embed_dim:      int,
        n_mamba_layers: int = 2,
        d_state:        int = 16,
        d_conv:         int = 4,
        expand:         int = 2,
        hidden_dim:     int = 256,
        dropout:        float = 0.2,
        power_feat_dim: int = 16,
    ):
        super().__init__()

        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm":  nn.LayerNorm(embed_dim),
                "mamba": build_mamba_layer(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
            })
            for _ in range(n_mamba_layers)
        ])

        self.pre_head_norm = nn.LayerNorm(embed_dim)
        self.power_proj = nn.Sequential(nn.Linear(1, power_feat_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(embed_dim + power_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # z
        )

    def forward(self, x: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, T, D] embeddings
        power: [B]       normalized sqrt((power_mW/I_th)^(1/N)-1)
        returns: [B, T, 1]  z prediction at every timestep
        """
        for block in self.mamba_blocks:
            x = x + block["mamba"](block["norm"](x))

        x = self.pre_head_norm(x)
        B, T, D = x.shape

        power_feat = self.power_proj(power.unsqueeze(-1))
        power_feat = power_feat.unsqueeze(1).expand(-1, T, -1)

        combined = torch.cat([x, power_feat], dim=-1)
        preds = self.head(combined)   # [B, T, 1]
        return preds


class MambaSequenceRegressorZ(nn.Module):
    """
    Video-only counterpart to MambaSequenceRegressorZWithPower -- same
    Mamba trunk, no power fusion, 1-output head.
    """

    def __init__(
        self,
        embed_dim:      int,
        n_mamba_layers: int = 2,
        d_state:        int = 16,
        d_conv:         int = 4,
        expand:         int = 2,
        hidden_dim:     int = 256,
        dropout:        float = 0.2,
    ):
        super().__init__()

        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm":  nn.LayerNorm(embed_dim),
                "mamba": build_mamba_layer(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
            })
            for _ in range(n_mamba_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # z
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] embeddings
        returns: [B, T, 1]  z prediction at every timestep
        """
        for block in self.mamba_blocks:
            x = x + block["mamba"](block["norm"](x))

        preds = self.head(x)   # [B, T, 1]
        return preds


# --------------------------------------------------
# Loss function -- flat per-timestep weighting, unweighted (settled config)
# --------------------------------------------------

def compute_loss_z(preds: torch.Tensor, targets: torch.Tensor, loss_fn_mean) -> torch.Tensor:
    """
    preds:   [B, T, 1]
    targets: [B, 1]
    Flat timestep weighting (every frame counts equally) and unweighted
    targets -- the settled default, matching the x/y config.
    """
    target_exp = targets.unsqueeze(1).expand_as(preds)   # [B, T, K]
    return loss_fn_mean(preds, target_exp)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

@torch.no_grad()
def evaluate_z(model, loader, device, target_z_mean, target_z_std, ept_threshold_um):
    """
    Same four-way return shape as evaluate_xy: (metrics, per_video_df,
    mae_per_t_df, ept_df) -- single target instead of doubled.
    """
    model.eval()

    all_preds_norm, all_targets_phys, all_video_ids = [], [], []

    for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in loader:
        emb = emb.to(device)
        power_norm = power_norm.to(device)
        preds = model(emb, power_norm)          # [B, T, 1]
        all_preds_norm.append(preds.cpu().numpy())
        all_targets_phys.append(targets_phys.numpy())
        all_video_ids.extend(list(video_ids))

    preds_norm_arr = np.concatenate(all_preds_norm, axis=0)       # [N, T, 1]
    targets_phys_arr = np.concatenate(all_targets_phys, axis=0)   # [N, 1]
    N, T, _ = preds_norm_arr.shape

    preds_phys_arr = np.zeros_like(preds_norm_arr)
    for t in range(T):
        preds_phys_arr[:, t, 0] = inverse_transform(preds_norm_arr[:, t, 0], target_z_mean, target_z_std)

    frame_pct = np.linspace(0.0, 100.0, T)

    mae_z_per_t = [mean_absolute_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]) for t in range(T)]
    rmse_z_per_t = [float(np.sqrt(mean_squared_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]))) for t in range(T)]

    mae_per_t_df = pd.DataFrame({
        "frame_idx": np.arange(T), "frame_pct": frame_pct,
        "mae_z_phys": mae_z_per_t, "rmse_z_phys": rmse_z_per_t,
    })

    final_mae_z = float(mae_z_per_t[-1])
    final_mae_std_z = float(np.std(np.abs(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0])))
    final_rmse_z = float(rmse_z_per_t[-1])
    final_bias_z = float(np.mean(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0]))

    # EPT: earliest frame after which z error stays below threshold
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
        "mean_ept_pct": float(np.mean(ept_pcts)),
        "pct_videos_ept_found": float(np.mean(ept_found) * 100.0),
        "ept_threshold_um": float(ept_threshold_um),
        "n_videos": N,
        "n_frames": T,
    }
    return metrics, per_video_df, mae_per_t_df, ept_df


@torch.no_grad()
def evaluate_z_video_only(model, loader, device, target_z_mean, target_z_std, ept_threshold_um):
    """
    Video-only counterpart to evaluate_z: model(emb) -> [B, T, 1], no power
    fusion. Same four-way return shape.
    """
    model.eval()

    all_preds_norm, all_targets_phys, all_video_ids = [], [], []

    for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in loader:
        emb = emb.to(device)
        preds = model(emb)                      # [B, T, 1]
        all_preds_norm.append(preds.cpu().numpy())
        all_targets_phys.append(targets_phys.numpy())
        all_video_ids.extend(list(video_ids))

    preds_norm_arr = np.concatenate(all_preds_norm, axis=0)       # [N, T, 1]
    targets_phys_arr = np.concatenate(all_targets_phys, axis=0)   # [N, 1]
    N, T, _ = preds_norm_arr.shape

    preds_phys_arr = np.zeros_like(preds_norm_arr)
    for t in range(T):
        preds_phys_arr[:, t, 0] = inverse_transform(preds_norm_arr[:, t, 0], target_z_mean, target_z_std)

    frame_pct = np.linspace(0.0, 100.0, T)

    mae_z_per_t = [mean_absolute_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]) for t in range(T)]
    rmse_z_per_t = [float(np.sqrt(mean_squared_error(targets_phys_arr[:, 0], preds_phys_arr[:, t, 0]))) for t in range(T)]

    mae_per_t_df = pd.DataFrame({
        "frame_idx": np.arange(T), "frame_pct": frame_pct,
        "mae_z_phys": mae_z_per_t, "rmse_z_phys": rmse_z_per_t,
    })

    final_mae_z = float(mae_z_per_t[-1])
    final_mae_std_z = float(np.std(np.abs(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0])))
    final_rmse_z = float(rmse_z_per_t[-1])
    final_bias_z = float(np.mean(preds_phys_arr[:, -1, 0] - targets_phys_arr[:, 0]))

    # EPT: earliest frame after which z error stays below threshold
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
        "mean_ept_pct": float(np.mean(ept_pcts)),
        "pct_videos_ept_found": float(np.mean(ept_found) * 100.0),
        "ept_threshold_um": float(ept_threshold_um),
        "n_videos": N,
        "n_frames": T,
    }
    return metrics, per_video_df, mae_per_t_df, ept_df


# --------------------------------------------------
# Plots
# --------------------------------------------------

def plot_mae_curve_z(mae_per_t_df: pd.DataFrame, out_path: str, fold: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mae_per_t_df["frame_pct"], mae_per_t_df["mae_z_phys"],
            label="MAE (μm)", color="steelblue", linewidth=2,
            marker="o", markersize=4)
    ax.plot(mae_per_t_df["frame_pct"], mae_per_t_df["rmse_z_phys"],
            label="RMSE (μm)", color="tomato", linestyle="--", linewidth=2,
            marker="o", markersize=4)
    ax.set_ylabel("z error (μm)")
    ax.set_title(f"Fold {fold} — z prediction error vs. video progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Video progress (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ept_distribution_z(ept_df: pd.DataFrame, out_path: str, fold: int, ept_threshold_um: float):
    found = ept_df[ept_df["ept_found"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(found) > 0:
        ax.hist(found["ept_pct"], bins=min(10, len(found)), color="steelblue", edgecolor="white")
    ax.set_xlabel("EPT — video progress when prediction stabilised (%)")
    ax.set_ylabel("Number of videos")
    ax.set_title(
        f"Fold {fold} — Early Prediction Time distribution (z)\n"
        f"(threshold = {ept_threshold_um} μm, "
        f"{len(found)}/{len(ept_df)} videos detected)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_video_trajectories_z(
    per_video_df: pd.DataFrame,
    out_dir: str,
    ept_threshold_um: float,
    max_videos: int = 10,
    video_power_lookup: dict = None,
    bandwidth_lookup=None,
):
    """
    video_power_lookup: video_id -> power_mW (from kfold_splits_z/fold_N_val.csv).
    bandwidth_lookup: power_mW -> bandwidth_um (see load_bandwidth_lookup_z() /
    scripts/12_compute_ept_bandwidths.py). Each video's shaded band and EPT
    marker use its own power's bandwidth instead of one flat ept_threshold_um
    shared by the whole dataset. Falls back to ept_threshold_um if the
    video's power isn't found in the lookup.
    """
    os.makedirs(out_dir, exist_ok=True)
    video_ids = per_video_df["video_id"].unique()[:max_videos]
    use_bandwidth = video_power_lookup is not None and bandwidth_lookup is not None

    for vid_id in video_ids:
        sub = per_video_df[per_video_df["video_id"] == vid_id].sort_values("frame_idx").reset_index(drop=True)

        if use_bandwidth:
            power = video_power_lookup.get(vid_id)
            bw = float(bandwidth_lookup.get(power, ept_threshold_um))
            err_z = sub["abs_err_z_phys"].values
            frame_pct_arr = sub["frame_pct"].values
            T = len(sub)
            ept_frame = None
            for t in range(T):
                if np.all(err_z[t:] < bw):
                    ept_frame = t
                    break
            # a video that only "converges" at the very last frame is treated
            # as not converged -- nothing informative to mark with a line
            # sitting right at the plot's edge
            detected = ept_frame is not None
            ept_pct = float(frame_pct_arr[ept_frame]) if detected else 100.0
            if detected and ept_pct >= 100.0:
                detected = False
        else:
            bw = ept_threshold_um
            ept_pct = float(sub["ept_pct"].iloc[0])
            detected = bool(sub["ept_found"].iloc[0]) and ept_pct < 100.0

        fig, ax = plt.subplots(figsize=(9, 5))
        true_val = float(sub["true_z_phys"].iloc[0])
        ax.plot(sub["frame_pct"], sub["pred_z_phys"],
                color="steelblue", linewidth=2, marker="o", markersize=4,
                label="Predicted z resolution")
        ax.axhline(true_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"True: {true_val:.3f} μm")
        ax.axhspan(true_val - bw, true_val + bw,
                   alpha=0.1, color="tomato", label=f"±{bw:.4f} μm bandwidth")
        if detected:
            ax.axvline(ept_pct, color="green", linestyle=":", linewidth=2,
                       label=f"EPT: {ept_pct:.1f}% of video")
        ax.set_ylabel("z resolution (μm)")
        ax.set_xlabel("Video progress (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{vid_id}")
        fig.tight_layout()

        safe_id = str(vid_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.savefig(os.path.join(out_dir, f"{safe_id}_trajectory.png"), dpi=120)
        plt.close(fig)


# --------------------------------------------------
# Train one fold
# --------------------------------------------------

def train_cached_mamba_fold_z(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    i_th = fit_i_th_axial(
        train_payload_raw["power_mW"].numpy(),
        train_payload_raw["targets_z_phys"].numpy(),
    )
    print(f"Fold {fold} | fitted I_th (TRAIN only) = {i_th:.4f} mW")

    train_dataset = CachedMambaEmbeddingDatasetZ(train_path, i_th=i_th)
    val_dataset = CachedMambaEmbeddingDatasetZ(
        val_path, i_th=i_th, power_mean=train_dataset.power_mean, power_std=train_dataset.power_std,
    )

    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    embed_dim = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressorZWithPower(
        embed_dim=embed_dim,
        n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()

    target_z_mean, target_z_std = train_dataset.target_z_mean, train_dataset.target_z_std

    best_val_mae = float("inf")
    best_epoch = None
    epochs_no_imp = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []

        for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in train_loader:
            emb = emb.to(device)
            targets_norm = targets_norm.to(device)
            power_norm = power_norm.to(device)

            optimizer.zero_grad()
            preds = model(emb, power_norm)   # [B, T, 1]
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
            best_val_mae = current_mae
            best_epoch = epoch
            epochs_no_imp = 0
            torch.save({
                "fold": fold, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "target_z_mean": target_z_mean, "target_z_std": target_z_std,
                "power_mean": train_dataset.power_mean, "power_std": train_dataset.power_std,
                "i_th": i_th,
                "best_val_mae": best_val_mae,
            }, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({
            "fold": fold, "epoch": epoch, "train_loss": train_loss,
            **val_metrics,
            "val_mae": current_mae,
            "best_val_mae_so_far": best_val_mae,
            "improved": improved,
            "epochs_without_improvement": epochs_no_imp,
        })

        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"mae_z={val_metrics['final_mae_z_phys']:.5f} | best={best_val_mae:.5f} | "
              f"no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}")

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}, MAE: {best_val_mae:.5f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

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
    plot_ept_distribution_z(ept_df, os.path.join(out_dir, "ept_distribution.png"), fold, cfg["ept_threshold_um"])
    z_bw_lookup = load_bandwidth_lookup_z()
    video_power_lookup = load_video_power_lookup_z(fold, split_dir=cfg.get("split_dir"))
    plot_video_trajectories_z(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, bandwidth_lookup=z_bw_lookup,
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
        "i_th_mW": i_th,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return pd.DataFrame(history), fold_summary


def train_cached_mamba_fold_z_video_only(fold: int, cfg: dict, device: torch.device):
    """
    Video-only counterpart to train_cached_mamba_fold_z: same dataset,
    same settled loss config, just MambaSequenceRegressorZ (no power_norm
    passed to the model) and evaluate_z_video_only.
    """
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    i_th = fit_i_th_axial(
        train_payload_raw["power_mW"].numpy(),
        train_payload_raw["targets_z_phys"].numpy(),
    )
    print(f"Fold {fold} | fitted I_th (TRAIN only) = {i_th:.4f} mW")

    train_dataset = CachedMambaEmbeddingDatasetZ(train_path, i_th=i_th)
    val_dataset = CachedMambaEmbeddingDatasetZ(
        val_path, i_th=i_th, power_mean=train_dataset.power_mean, power_std=train_dataset.power_std,
    )

    print(f"Fold {fold} | train rows: {len(train_dataset)} | val videos: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    embed_dim = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressorZ(
        embed_dim=embed_dim,
        n_mamba_layers=cfg["n_mamba_layers"], d_state=cfg["d_state"],
        d_conv=cfg["d_conv"], expand=cfg["expand"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn_mean = nn.SmoothL1Loss()

    target_z_mean, target_z_std = train_dataset.target_z_mean, train_dataset.target_z_std

    best_val_mae = float("inf")
    best_epoch = None
    epochs_no_imp = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []

        for emb, targets_norm, targets_phys, loss_weight, power_norm, video_ids in train_loader:
            emb = emb.to(device)
            targets_norm = targets_norm.to(device)

            optimizer.zero_grad()
            preds = model(emb)   # [B, T, 1]
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
            best_val_mae = current_mae
            best_epoch = epoch
            epochs_no_imp = 0
            torch.save({
                "fold": fold, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "target_z_mean": target_z_mean, "target_z_std": target_z_std,
                "i_th": i_th,
                "best_val_mae": best_val_mae,
            }, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({
            "fold": fold, "epoch": epoch, "train_loss": train_loss,
            **val_metrics,
            "val_mae": current_mae,
            "best_val_mae_so_far": best_val_mae,
            "improved": improved,
            "epochs_without_improvement": epochs_no_imp,
        })

        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"mae_z={val_metrics['final_mae_z_phys']:.5f} | best={best_val_mae:.5f} | "
              f"no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}")

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}, MAE: {best_val_mae:.5f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

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
    plot_ept_distribution_z(ept_df, os.path.join(out_dir, "ept_distribution.png"), fold, cfg["ept_threshold_um"])
    z_bw_lookup = load_bandwidth_lookup_z()
    video_power_lookup = load_video_power_lookup_z(fold, split_dir=cfg.get("split_dir"))
    plot_video_trajectories_z(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
        video_power_lookup=video_power_lookup, bandwidth_lookup=z_bw_lookup,
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
        "i_th_mW": i_th,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return pd.DataFrame(history), fold_summary
