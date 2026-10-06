# scripts/25_wilcoxon_hint_value.py
#
# Tests the "hint does the work, not the video" pattern seen in the
# master results table: baseline MLP (hint only, no video) vs fused
# (DINOv2+Mamba+hint), and no-Mamba+hint vs fused -- on the matched
# 106-video set, both split schemes, every applicable axis.
#
# Baseline is non-temporal (one prediction/video, fold_N_predictions.csv).
# Fused/no-Mamba are temporal (val_predictions_per_timestep.csv) -- final
# timestep per video used as the comparable per-video error, same
# approach as the earlier Wilcoxon scripts in xy_early_prediction.

import os

import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def baseline_errors(result_dir, fold, cols):
    df = pd.read_csv(os.path.join(ROOT, "results", result_dir, f"fold_{fold}_predictions.csv"))
    return df.set_index("video_id")[cols]


def temporal_final_errors(result_dir, fold, cols):
    df = pd.read_csv(os.path.join(ROOT, "results", result_dir, f"fold_{fold}", "val_predictions_per_timestep.csv"))
    last = df.loc[df.groupby("video_id")["frame_pct"].idxmax()]
    return last.set_index("video_id")[cols]


def run_pair(label, a_dir, a_kind, b_dir, b_kind, cols, n_splits):
    pooled = {c: ([], []) for c in cols}
    n_total = 0
    get_a = baseline_errors if a_kind == "baseline" else temporal_final_errors
    get_b = baseline_errors if b_kind == "baseline" else temporal_final_errors

    for fold in range(1, n_splits + 1):
        a = get_a(a_dir, fold, cols)
        b = get_b(b_dir, fold, cols)
        common = a.index.intersection(b.index)
        if len(common) != len(a) or len(common) != len(b):
            print(f"{label} fold {fold}: WARNING mismatch a={len(a)} b={len(b)} common={len(common)}")
        n_total += len(common)
        for c in cols:
            pooled[c][0].extend(a.loc[common, c].tolist())
            pooled[c][1].extend(b.loc[common, c].tolist())

    print(f"\n=== {label} === (n paired videos pooled across {n_splits} folds: {n_total})")
    for c in cols:
        a_vals, b_vals = pooled[c]
        stat, p = wilcoxon(a_vals, b_vals)
        a_med, b_med = pd.Series(a_vals).median(), pd.Series(b_vals).median()
        axis = c.replace("abs_err_", "").replace("_phys", "")
        winner = "A" if a_med < b_med else "B"
        print(f"--- axis {axis} ---  A median={a_med:.5f}  B median={b_med:.5f}  "
              f"p={p:.6f}  {'significant' if p < 0.05 else 'not significant'}  (lower: {winner})")


def main():
    jobs = [
        ("5-fold", 5, ""), ("loo-batch", 6, "_loo_batch"),
    ]
    for scheme_name, n_splits, suffix in jobs:
        print(f"\n################ {scheme_name} ################")
        run_pair(f"[{scheme_name}] baseline MLP (A) vs fused (B) -- x/y",
                  f"xy_baseline_mlp{suffix}", "baseline", f"xy_dinov2_mamba_fused{suffix}", "temporal",
                  ["abs_err_x_phys", "abs_err_y_phys"], n_splits)
        run_pair(f"[{scheme_name}] baseline MLP (A) vs fused (B) -- z",
                  f"z_baseline_mlp{suffix}", "baseline", f"z_dinov2_mamba_fused{suffix}", "temporal",
                  ["abs_err_z_phys"], n_splits)
        run_pair(f"[{scheme_name}] no-Mamba+hint (A) vs fused (B) -- x/y",
                  f"xy_no_mamba{suffix}", "temporal", f"xy_dinov2_mamba_fused{suffix}", "temporal",
                  ["abs_err_x_phys", "abs_err_y_phys"], n_splits)
        run_pair(f"[{scheme_name}] no-Mamba+hint (A) vs fused (B) -- z",
                  f"z_no_mamba{suffix}", "temporal", f"z_dinov2_mamba_fused{suffix}", "temporal",
                  ["abs_err_z_phys"], n_splits)


if __name__ == "__main__":
    main()
