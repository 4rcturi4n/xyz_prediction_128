"""
Copy of fused_hint_postmamba.py's z training function, using the new
hint formula (hint_z_newhint.py: sqrt(sqrt(ln(P/I_th))-1), I_th fit
per fold TRAIN-only) instead of the original sqrt(sqrt(P)-1). Everything
else (Mamba model, loss, eval, plotting) is unchanged -- only the hint
feature and the dataset class that computes it differ.

Does not touch fused_hint_postmamba.py or any other existing file.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mamba_xy.mamba_z_regression import (
    MambaSequenceRegressorZWithPower, evaluate_z, compute_loss_z, plot_mae_curve_z,
    plot_video_trajectories_z, load_bandwidth_lookup_z, load_video_power_lookup_z,
)
from mamba_xy.fused_hint_z import load_video_path_lookup_z
from mamba_xy.hint_z_newhint import fit_i_th_for_hint, fit_baseline_mlp_no_leak, baseline_predict
from mamba_xy.extra_plots import (
    plot_learning_curve, save_video_frames_for_fold,
    compute_bandwidth_ept, plot_ept_distribution_bandwidth,
)


class CachedMambaEmbeddingDatasetZPostMambaHintNew(Dataset):
    def __init__(self, payload_path: str, baseline_model_z, i_th: float, loss_weight_min_eps: float = 1e-6):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        self.embeddings = payload["embeddings"].float()
        self.targets_z = payload["targets_z"].float()
        self.targets_z_phys = payload["targets_z_phys"].float()
        self.errors_z_phys = payload["errors_z"].float()
        self.video_ids = payload["video_ids"]
        self.power_mW = payload["power_mW"].float()

        self.target_z_mean = float(payload["target_z_mean"])
        self.target_z_std = float(payload["target_z_std"])

        pred_z = baseline_predict(baseline_model_z, self.power_mW.numpy(), i_th)
        pred_z_norm = (pred_z - self.target_z_mean) / self.target_z_std
        self.hint = torch.tensor(pred_z_norm, dtype=torch.float32)

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


def train_cached_mamba_fold_z_postmamba_hint_newhint(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    train_payload_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    power_train = train_payload_raw["power_mW"].numpy()
    z_train = train_payload_raw["targets_z_phys"].numpy()

    i_th = fit_i_th_for_hint(power_train, z_train)
    print(f"Fold {fold} | fitted I_th (TRAIN only, new hint) = {i_th:.4f} mW")
    baseline_model_z = fit_baseline_mlp_no_leak(power_train, z_train, i_th, seed=cfg["seed"] + fold)
    print(f"Fold {fold} | baseline MLP fit (TRAIN-internal-holdout only, no val leakage)")

    train_dataset = CachedMambaEmbeddingDatasetZPostMambaHintNew(train_path, baseline_model_z, i_th)
    val_dataset = CachedMambaEmbeddingDatasetZPostMambaHintNew(val_path, baseline_model_z, i_th)
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
                        "best_val_mae": best_val_mae, "i_th_mW": i_th}, os.path.join(out_dir, "model_best.pth"))
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

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_z(
        model=model, loader=val_loader, device=device,
        target_z_mean=target_z_mean, target_z_std=target_z_std,
        ept_threshold_um=cfg["ept_threshold_um"],
    )
    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
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
        "fold": fold, "best_epoch": best_epoch, "i_th_mW": i_th,
        "best_val_mae_z_phys": float(best_val_metrics["final_mae_z_phys"]),
        "best_val_mae_std_z_phys": float(best_val_metrics["final_mae_std_z_phys"]),
        "best_val_rmse_z_phys": float(best_val_metrics["final_rmse_z_phys"]),
        "best_val_bias_z_phys": float(best_val_metrics["final_bias_z_phys"]),
        "num_train_rows": len(train_dataset), "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary
