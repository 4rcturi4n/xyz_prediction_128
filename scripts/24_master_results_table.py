# scripts/24_master_results_table.py
#
# Consolidates every model/method trained across both repos into one CSV:
# one row per (method, split scheme), columns = mean/std MAE per axis
# (x, y, z always separate, never combined), plus where the raw results
# live on disk. Handles the different kfold_summary.json key naming
# conventions used across model types (Mamba models use
# "mean_best_val_mae_*_phys", MLP/physics baselines use
# "mean_val_mae_*_phys").
#
# Writes: results/master_results_table.csv (in this repo)

import os
import json

import pandas as pd

NEW_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_REPO = os.path.abspath(os.path.join(NEW_REPO, "..", "xy_early_prediction"))

# (repo, result_dir_name, method_label, split_scheme, n_folds)
ENTRIES = [
    # --- xyz_prediction_128: 106-video, 128-frame ---
    (NEW_REPO, "xy_dinov2_mamba_video_only", "DINOv2+Mamba video-only", "5-fold", 5),
    (NEW_REPO, "xy_dinov2_mamba_video_only_loo_batch", "DINOv2+Mamba video-only", "loo-batch", 6),
    (NEW_REPO, "z_dinov2_mamba_video_only", "DINOv2+Mamba video-only", "5-fold", 5),
    (NEW_REPO, "z_dinov2_mamba_video_only_loo_batch", "DINOv2+Mamba video-only", "loo-batch", 6),

    (NEW_REPO, "xy_dinov2_mamba_fused", "DINOv2+Mamba+hint (fused, post-trunk)", "5-fold", 5),
    (NEW_REPO, "xy_dinov2_mamba_fused_loo_batch", "DINOv2+Mamba+hint (fused, post-trunk)", "loo-batch", 6),
    (NEW_REPO, "z_dinov2_mamba_fused", "DINOv2+Mamba+hint (fused, post-trunk)", "5-fold", 5),
    (NEW_REPO, "z_dinov2_mamba_fused_loo_batch", "DINOv2+Mamba+hint (fused, post-trunk)", "loo-batch", 6),

    (NEW_REPO, "xy_no_mamba", "DINOv2+MLP+hint (no Mamba)", "5-fold", 5),
    (NEW_REPO, "xy_no_mamba_loo_batch", "DINOv2+MLP+hint (no Mamba)", "loo-batch", 6),
    (NEW_REPO, "z_no_mamba", "DINOv2+MLP+hint (no Mamba)", "5-fold", 5),
    (NEW_REPO, "z_no_mamba_loo_batch", "DINOv2+MLP+hint (no Mamba)", "loo-batch", 6),

    (NEW_REPO, "xy_no_mamba_no_hint", "DINOv2+MLP (no Mamba, no hint)", "5-fold", 5),
    (NEW_REPO, "xy_no_mamba_no_hint_loo_batch", "DINOv2+MLP (no Mamba, no hint)", "loo-batch", 6),
    (NEW_REPO, "z_no_mamba_no_hint", "DINOv2+MLP (no Mamba, no hint)", "5-fold", 5),
    (NEW_REPO, "z_no_mamba_no_hint_loo_batch", "DINOv2+MLP (no Mamba, no hint)", "loo-batch", 6),

    (NEW_REPO, "xy_handcrafted_mamba", "Hand-crafted+Mamba video-only", "5-fold", 5),
    (NEW_REPO, "xy_handcrafted_mamba_loo_batch", "Hand-crafted+Mamba video-only", "loo-batch", 6),
    (NEW_REPO, "z_handcrafted_mamba", "Hand-crafted+Mamba video-only", "5-fold", 5),
    (NEW_REPO, "z_handcrafted_mamba_loo_batch", "Hand-crafted+Mamba video-only", "loo-batch", 6),

    (NEW_REPO, "xy_baseline_mlp", "Baseline MLP (hint only, no video)", "5-fold", 5),
    (NEW_REPO, "xy_baseline_mlp_loo_batch", "Baseline MLP (hint only, no video)", "loo-batch", 6),
    (NEW_REPO, "z_baseline_mlp", "Baseline MLP (hint only, no video)", "5-fold", 5),
    (NEW_REPO, "z_baseline_mlp_loo_batch", "Baseline MLP (hint only, no video)", "loo-batch", 6),

    # --- matched 106-video set, physics baselines (comparable to everything above) ---
    (NEW_REPO, "xy_baseline_physics", "Pure physics fit (independent I_th per axis)", "5-fold", 5),
    (NEW_REPO, "xy_baseline_physics_loo_batch", "Pure physics fit (independent I_th per axis)", "loo-batch", 6),
    (NEW_REPO, "z_baseline_physics", "Pure physics fit (independent I_th per axis)", "5-fold", 5),
    (NEW_REPO, "z_baseline_physics_loo_batch", "Pure physics fit (independent I_th per axis)", "loo-batch", 6),
    (NEW_REPO, "baseline_physics_joint_shared_ith", "Pure physics fit (joint, shared I_th)", "5-fold", 5),
    (NEW_REPO, "baseline_physics_joint_shared_ith_loo_batch", "Pure physics fit (joint, shared I_th)", "loo-batch", 6),

    # --- xy_early_prediction: original full (non-matched) dataset, NOT comparable 1:1 to the above ---
    (OLD_REPO, "baseline_mlp_xy", "Baseline MLP (hint only, no video) [32-frame, full dataset]", "5-fold", 5),
    (OLD_REPO, "baseline_mlp_z", "Baseline MLP (hint only, no video) [32-frame, full dataset]", "5-fold", 5),
    (OLD_REPO, "mamba_xy_fused_hint_postmamba", "DINOv2+Mamba+hint (fused, post-trunk) [32-frame, full dataset]", "5-fold", 5),
    (OLD_REPO, "mamba_z_fused_hint_postmamba", "DINOv2+Mamba+hint (fused, post-trunk) [32-frame, full dataset]", "5-fold", 5),
    (OLD_REPO, "mamba_xy_fused_handcrafted_dense", "Hand-crafted(dense)+Mamba+hint (fused) [32-frame, full dataset]", "5-fold", 5),
    (OLD_REPO, "mamba_z_fused_handcrafted_dense", "Hand-crafted(dense)+Mamba+hint (fused) [32-frame, full dataset]", "5-fold", 5),
    (OLD_REPO, "baseline_physics_xy", "Pure physics fit (independent I_th per axis) [full dataset]", "5-fold", 5),
    (OLD_REPO, "baseline_physics_z", "Pure physics fit (independent I_th per axis) [full dataset]", "5-fold", 5),
    (OLD_REPO, "baseline_physics_joint_shared_ith", "Pure physics fit (joint, shared I_th) [full dataset]", "5-fold", 5),
]

# key patterns to try, in order, for each axis's mean/std
KEY_PATTERNS = [
    ("mean_best_val_mae_{a}_phys", "std_best_val_mae_{a}_phys"),
    ("mean_val_mae_{a}_phys", "std_val_mae_{a}_phys"),
]


def extract_axis(summary, axis):
    for mean_pat, std_pat in KEY_PATTERNS:
        mean_key, std_key = mean_pat.format(a=axis), std_pat.format(a=axis)
        if mean_key in summary:
            return summary[mean_key], summary.get(std_key)
    return None, None


def main():
    rows = []
    seen = set()  # (repo, result_dir) -> avoid double-processing if an entry appears for both x/y and z calls
    for repo, result_dir, method, scheme, n_folds in ENTRIES:
        key = (repo, result_dir)
        if key in seen:
            continue
        seen.add(key)

        summary_path = os.path.join(repo, "results", result_dir, "kfold_summary.json")
        if not os.path.exists(summary_path):
            print(f"MISSING: {summary_path}")
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        mae_x_mean, mae_x_std = extract_axis(summary, "x")
        mae_y_mean, mae_y_std = extract_axis(summary, "y")
        mae_z_mean, mae_z_std = extract_axis(summary, "z")

        rows.append({
            "method": method,
            "split_scheme": scheme,
            "n_folds": n_folds,
            "repo": os.path.basename(repo),
            "result_folder": os.path.join(repo, "results", result_dir),
            "mae_x_mean": mae_x_mean, "mae_x_std": mae_x_std,
            "mae_y_mean": mae_y_mean, "mae_y_std": mae_y_std,
            "mae_z_mean": mae_z_mean, "mae_z_std": mae_z_std,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(NEW_REPO, "results", "master_results_table.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
