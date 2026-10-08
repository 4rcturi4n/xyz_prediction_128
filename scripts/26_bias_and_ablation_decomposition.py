# scripts/26_bias_and_ablation_decomposition.py
#
# Item 7: for the standout no-hint video models (systematically higher
# bias than hint-based models on z), checks whether that bias is
# consistent in SIGN across all folds (a real systematic effect) or
# mixed (just noise averaging to a nonzero mean).
#
# Item 8: formal 2x2 factorial decomposition of the Mamba x hint
# ablation grid -- main effect of Mamba, main effect of hint, and their
# interaction, on MAE, per axis, per split scheme. Uses the 4 corners:
#   M(0,0) = no_mamba_no_hint     M(0,1) = no_mamba (+hint)
#   M(1,0) = video_only           M(1,1) = fused (Mamba+hint)

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def per_fold_bias(result_dir, n_folds, axis):
    vals = []
    for fold in range(1, n_folds + 1):
        p = os.path.join(ROOT, "results", result_dir, f"fold_{fold}", "summary.json")
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        vals.append(s[f"best_val_bias_{axis}_phys"])
    return vals


def check_bias_consistency():
    print("=" * 80)
    print("ITEM 7: is the no-hint video models' positive z bias systematic (same sign")
    print("every fold) or just noise averaging to a nonzero mean?")
    print("=" * 80)
    cases = [
        ("xy_dinov2_mamba_video_only", 5, "x"), ("xy_dinov2_mamba_video_only", 5, "y"),
        ("z_dinov2_mamba_video_only", 5, "z"),
        ("xy_no_mamba_no_hint", 5, "x"), ("xy_no_mamba_no_hint", 5, "y"),
        ("z_no_mamba_no_hint", 5, "z"),
        ("xy_handcrafted_mamba", 5, "x"), ("xy_handcrafted_mamba", 5, "y"),
        ("z_handcrafted_mamba", 5, "z"),
        ("z_dinov2_mamba_fused", 5, "z"),  # hint-based control case for comparison
        ("z_no_mamba", 5, "z"),
    ]
    for result_dir, n_folds, axis in cases:
        vals = per_fold_bias(result_dir, n_folds, axis)
        signs = [v > 0 for v in vals]
        consistent = all(signs) or not any(signs)
        print(f"{result_dir:30s} axis={axis}  per-fold bias={[f'{v:+.4f}' for v in vals]}  "
              f"{'SAME SIGN every fold (systematic)' if consistent else 'mixed signs (not systematic)'}")


def decompose(mae_00, mae_01, mae_10, mae_11, label):
    main_hint = ((mae_01 - mae_00) + (mae_11 - mae_10)) / 2.0
    main_mamba = ((mae_10 - mae_00) + (mae_11 - mae_01)) / 2.0
    interaction = (mae_11 - mae_10) - (mae_01 - mae_00)
    print(f"{label:20s} no_mamba_no_hint={mae_00:.5f}  no_mamba+hint={mae_01:.5f}  "
          f"video_only={mae_10:.5f}  fused={mae_11:.5f}")
    print(f"{'':20s} main effect of HINT (avg MAE drop from adding hint):  {-main_hint:+.5f}")
    print(f"{'':20s} main effect of MAMBA (avg MAE drop from adding Mamba): {-main_mamba:+.5f}")
    print(f"{'':20s} interaction (hint's benefit changes by this much when Mamba is also present): {-interaction:+.5f}")
    print()


def run_ablation_decomposition():
    print("=" * 80)
    print("ITEM 8: 2x2 factorial decomposition (Mamba x hint), MAE, per axis/scheme")
    print("(negative 'effect' numbers = adding that factor REDUCES MAE, i.e. helps)")
    print("=" * 80)
    df = pd.read_csv(os.path.join(ROOT, "results", "master_results_table_new.csv"))
    df = df[df["repo"] == "xyz_prediction_128"]

    def get(method, scheme, col):
        row = df[(df["method"] == method) & (df["split_scheme"] == scheme)]
        return float(row[col].iloc[0])

    for scheme in ["5-fold", "loo-batch"]:
        for axis, col in [("x", "mae_x_mean"), ("y", "mae_y_mean"), ("z", "mae_z_mean")]:
            m00 = get("DINOv2+MLP (no Mamba, no hint)", scheme, col)
            m01 = get("DINOv2+MLP+hint (no Mamba)", scheme, col)
            m10 = get("DINOv2+Mamba video-only", scheme, col)
            m11 = get("DINOv2+Mamba+hint (fused, post-trunk)", scheme, col)
            decompose(m00, m01, m10, m11, f"[{scheme}, axis {axis}]")


if __name__ == "__main__":
    check_bias_consistency()
    run_ablation_decomposition()
