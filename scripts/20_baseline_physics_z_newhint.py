# scripts/20_baseline_physics_z_newhint.py
#
# Pure physics baseline for z, standalone predictor (no MLP, no video),
# using the NEW equation instead of the original:
#   OLD (15_baseline_physics.py): z = zr * sqrt((P/I_th)^(1/2) - 1) + b
#   NEW (this script):            z = zr * sqrt(sqrt(ln(P/I_th)) - 1) + b
#
# Fits zr, I_th, b jointly via curve_fit per fold, TRAIN-only, then
# predicts z directly from the equation on val -- same structure as
# 15_baseline_physics.py's run_z/fit_axial, just the new formula and
# all 3 parameters kept (not just I_th, unlike hint_z_newhint.py's
# fit_i_th_for_hint which discards zr/b since it only needs I_th for
# the MLP-hint use case).
#
# Separate output folders from xy_baseline_physics/z_baseline_physics.

import os
import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def model(P, zr, i_th, b):
    log_ratio = np.clip(np.log(np.clip(P / i_th, 1e-9, None)), 1e-9, None)
    inner = np.clip(np.sqrt(log_ratio) - 1.0, 1e-6, None)
    return zr * np.sqrt(inner) + b


def fit_z(power_mW, target_phys):
    p_min = float(power_mW.min())
    zr0 = float(target_phys.max() - target_phys.min())
    b0 = float(target_phys.min())
    ith0 = p_min / np.e / 2.0
    bounds = ([-np.inf, 1e-3, -np.inf], [np.inf, p_min / np.e * 0.999, np.inf])
    popt, _ = curve_fit(model, power_mW, target_phys, p0=[zr0, ith0, b0], bounds=bounds, maxfev=20000)
    return popt  # zr, i_th, b


def run_z(split_dir, out_dir, n_splits):
    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []
    for fold in range(1, n_splits + 1):
        train_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_train.csv"))
        val_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_val.csv"))

        zr, ith, b = fit_z(train_df["power_mW"].values.astype(np.float64),
                            train_df["z_resolution"].values.astype(np.float64))
        y_val = val_df["z_resolution"].values.astype(np.float32)
        pred_val = model(val_df["power_mW"].values.astype(np.float64), zr, ith, b).astype(np.float32)

        mae = mean_absolute_error(y_val, pred_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        pd.DataFrame({"video_id": val_df["video_id"].values, "power_mW": val_df["power_mW"].values,
                       "true_z_phys": y_val, "pred_z_phys": pred_val,
                       "abs_err_z_phys": np.abs(pred_val - y_val)}).to_csv(
            os.path.join(out_dir, f"fold_{fold}_predictions.csv"), index=False)
        fold_summaries.append({"fold": fold, "n_train": len(train_df), "n_val": len(val_df),
                                "zr": float(zr), "i_th_mW": float(ith), "b": float(b),
                                "val_mae_z_phys": float(mae), "val_rmse_z_phys": rmse})
        print(f"{out_dir} fold {fold}: z={mae:.5f} (zr={zr:.4f}, I_th={ith:.4f} mW, b={b:.4f})")

    mae_z = np.array([s["val_mae_z_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits,
             "model": "pure physics fit (new formula): z=zr*sqrt(sqrt(ln(P/I_th))-1)+b, per-fold fit",
             "mean_val_mae_z_phys": float(mae_z.mean()), "std_val_mae_z_phys": float(mae_z.std()),
             "folds": fold_summaries}
    with open(os.path.join(out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"{out_dir}: MAE_z={mae_z.mean():.5f}+/-{mae_z.std():.5f}")


def main():
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z"),
          os.path.join(ROOT, "results", "z_baseline_physics_newhint"), 5)
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z_loo_batch"),
          os.path.join(ROOT, "results", "z_baseline_physics_newhint_loo_batch"), 6)
    print("\nALL_BASELINE_PHYSICS_Z_NEWHINT_DONE")


if __name__ == "__main__":
    main()
