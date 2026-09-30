"""
Shared plotting/export helpers used by every model type (fused, video-
only, fused-hint) -- kept separate from mamba_xy_regression.py /
mamba_z_regression.py so those settled files are never touched:

- plot_learning_curve: train_loss vs val_mae across training epochs
  (from history.csv), NOT the existing mae_curve.png (which is MAE vs
  % of video seen at the best epoch -- a different thing).
- save_video_frames_for_fold: saves the actual cropped frame the model
  saw at each plotted trajectory point as its own full-resolution PNG,
  in a SEPARATE folder from the trajectory plots (one subfolder per
  video) -- not squeezed into a filmstrip inside the plot, which made
  them too small to see anything. Frames are reconstructed from the
  source video via frame_utils.get_cropped_frame (verified against the
  real extraction config -- see frame_utils.py's docstring).
"""

import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mamba_xy.frame_utils import get_cropped_frame


def plot_learning_curve(history_df: pd.DataFrame, out_path: str, fold: int, val_cols, best_epoch_col: str = None):
    """
    val_cols: list of (column_name, display_label) pairs plotted as
        separate lines on the right axis -- e.g. for x/y models, pass
        [("final_mae_x_phys", "val_mae_x"), ("final_mae_y_phys", "val_mae_y")]
        so x and y are never averaged into one displayed number (per
        standing rule: report x/y separately, never combined). For a
        single-target model (z), pass one pair.
    best_epoch_col: column used only to pick which epoch to mark as
        "best" (matches whatever the training loop's early stopping
        actually used) -- its name is never displayed.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(history_df["epoch"], history_df["train_loss"], color="C0", label="train_loss")

    colors = ["C1", "C2", "C3"]
    for i, (col, label) in enumerate(val_cols):
        ax2.plot(history_df["epoch"], history_df[col], color=colors[i % len(colors)], label=label)

    criterion_col = best_epoch_col or val_cols[0][0]
    best_epoch = history_df.loc[history_df[criterion_col].idxmin(), "epoch"]
    ax1.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train_loss", color="C0")
    ax2.set_ylabel("val MAE (μm)", color="C1")
    ax1.set_title(f"Fold {fold} learning curve (best epoch = {int(best_epoch)})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_video_frames_for_fold(
    per_video_df: pd.DataFrame,
    video_path_lookup: dict,
    out_dir: str,
    max_videos: int = 10,
):
    """
    For each of the first max_videos videos, saves the actual cropped
    frame at every plotted timestep as its own full-resolution PNG:
        out_dir/<safe_video_id>/frame_<idx>_pct<pct>.png
    No plot is produced here -- these are meant to be viewed directly,
    alongside the existing monitoring_plots/<video_id>_trajectory.png,
    not embedded inside it.

    video_path_lookup: video_id -> source video file path (from the
        kfold_splits csv's video_path column).
    """
    os.makedirs(out_dir, exist_ok=True)
    video_ids = per_video_df["video_id"].unique()[:max_videos]

    for vid_id in video_ids:
        video_path = video_path_lookup.get(vid_id)
        if video_path is None or not os.path.exists(video_path):
            continue

        sub = per_video_df[per_video_df["video_id"] == vid_id].sort_values("frame_idx").reset_index(drop=True)
        safe_id = str(vid_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
        video_out_dir = os.path.join(out_dir, safe_id)
        os.makedirs(video_out_dir, exist_ok=True)

        for _, row in sub.iterrows():
            frame_idx = int(row["frame_idx"])
            frame_pct = float(row["frame_pct"])
            try:
                frame_rgb = get_cropped_frame(video_path, frame_idx)
            except Exception:
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            fname = f"frame_{frame_idx:02d}_pct{frame_pct:05.1f}.png"
            cv2.imwrite(os.path.join(video_out_dir, fname), frame_bgr)


def compute_bandwidth_ept(
    per_video_df: pd.DataFrame,
    axis_names,
    bandwidth_lookups: dict,
    video_power_lookup: dict,
    ept_threshold_um: float,
) -> pd.DataFrame:
    """
    Per-video EPT using each video's OWN power-specific bandwidth as its
    tolerance band (per axis_name -- for x/y both must be inside their
    own band simultaneously), instead of one flat ept_threshold_um
    shared by the whole dataset. Falls back to ept_threshold_um only
    when a video's power isn't in the bandwidth lookup.

    Returns a DataFrame with columns: video_id, ept_pct, ept_found.
    """
    rows = []
    for vid_id, sub in per_video_df.groupby("video_id", sort=False):
        sub = sub.sort_values("frame_idx").reset_index(drop=True)
        power = video_power_lookup.get(vid_id)
        T = len(sub)
        inside_band = np.ones(T, dtype=bool)
        for axis_name in axis_names:
            bw_lookup = bandwidth_lookups.get(axis_name)
            bw = float(bw_lookup.get(power, ept_threshold_um)) if bw_lookup is not None else ept_threshold_um
            err = sub[f"abs_err_{axis_name}_phys"].values
            inside_band &= (err < bw)

        ept_frame = None
        for t in range(T):
            if np.all(inside_band[t:]):
                ept_frame = t
                break
        detected = ept_frame is not None
        ept_pct = float(sub["frame_pct"].iloc[ept_frame]) if detected else 100.0
        rows.append({"video_id": vid_id, "ept_pct": ept_pct, "ept_found": detected})

    return pd.DataFrame(rows)


def plot_ept_distribution_bandwidth(ept_df: pd.DataFrame, out_path: str, fold: int, axis_label: str):
    found = ept_df[ept_df["ept_found"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(found) > 0:
        ax.hist(found["ept_pct"], bins=min(10, len(found)), color="steelblue", edgecolor="white")
    ax.set_xlabel("EPT — video progress when prediction entered its own tolerance band (%)")
    ax.set_ylabel("Number of videos")
    ax.set_title(
        f"Fold {fold} — Early Prediction Time distribution ({axis_label})\n"
        f"(per-video power-specific bandwidth tolerance, "
        f"{len(found)}/{len(ept_df)} videos detected)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
