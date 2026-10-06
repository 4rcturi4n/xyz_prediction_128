# scripts/15_baseline_physics.py
#
# Pure physics baseline (no network at all) for the 106-video, matched
# dataset -- independent per-axis I_th fits, same equations as
# xy_early_prediction/scripts/08_baseline_physics_xy.py / _z.py, just
# repointed at this repo's 106-video kfold_splits so it's directly
# comparable to every other model in this repo's results table (the
# original physics-fit scripts ran on the OLD full-video-set splits,
# not matched to the 106-video DINOv2/Mamba models).
#
#   x/y: w = w0 * sqrt(2*ln(P/I_th))         (per axis, own w0/I_th)
#   z:   z = zr * sqrt((P/I_th)^(1/2) - 1) + b
#
# Both split schemes (5-fold, leave-one-batch-out).
# Writes: results/{xy,z}_baseline_physics(_loo_batch)/fold_N_predictions.csv, kfold_summary.json

import os
import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_ORDER = 2


def lateral_model(P, w0, ith):
    ratio = np.clip(P / ith, 1.0 + 1e-9, None)
    return w0 * np.sqrt(2.0 * np.log(ratio))


def axial_model(P, zr, ith, b):
    ratio = np.clip((P / ith) ** (1.0 / N_ORDER), 1.0 + 1e-9, None)
    return zr * np.sqrt(ratio - 1.0) + b


def fit_lateral(power_mW, target_phys):
    p_min = float(power_mW.min())
    w0_0 = float(target_phys.max() - target_phys.min())
    ith0 = 0.5 * p_min
    bounds = ([-np.inf, 1e-3], [np.inf, p_min * 0.999])
    popt, _ = curve_fit(lateral_model, power_mW, target_phys, p0=[w0_0, ith0], bounds=bounds, maxfev=20000)
    return popt


def fit_axial(power_mW, target_phys):
    p_min = float(power_mW.min())
    zr0 = float(target_phys.max() - target_phys.min())
    b0 = float(target_phys.min())
    ith0 = 0.5 * p_min
    bounds = ([-np.inf, 1e-3, -np.inf], [np.inf, p_min * 0.999, np.inf])
    popt, _ = curve_fit(axial_model, power_mW, target_phys, p0=[zr0, ith0, b0], bounds=bounds, maxfev=20000)
    return popt


def run_xy(split_dir, out_dir, n_splits):
    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []
    for fold in range(1, n_splits + 1):
        train_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_train.csv"))
        val_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_val.csv"))
        summary = {"fold": fold, "n_train": len(train_df), "n_val": len(val_df)}
        pred_cols = {"video_id": val_df["video_id"].values, "power_mW": val_df["power_mW"].values}
        for axis in ("x", "y"):
            w0, ith = fit_lateral(train_df["power_mW"].values.astype(np.float64),
                                   train_df[f"{axis}_resolution"].values.astype(np.float64))
            y_val = val_df[f"{axis}_resolution"].values.astype(np.float32)
            pred_val = lateral_model(val_df["power_mW"].values.astype(np.float64), w0, ith).astype(np.float32)
            mae = mean_absolute_error(y_val, pred_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            pred_cols[f"true_{axis}_phys"] = y_val
            pred_cols[f"pred_{axis}_phys"] = pred_val
            pred_cols[f"abs_err_{axis}_phys"] = np.abs(pred_val - y_val)
            summary[f"w0_{axis}"], summary[f"i_th_{axis}_mW"] = float(w0), float(ith)
            summary[f"val_mae_{axis}_phys"], summary[f"val_rmse_{axis}_phys"] = float(mae), rmse
        fold_summaries.append(summary)
        pd.DataFrame(pred_cols).to_csv(os.path.join(out_dir, f"fold_{fold}_predictions.csv"), index=False)
        print(f"{out_dir} fold {fold}: x={summary['val_mae_x_phys']:.5f} y={summary['val_mae_y_phys']:.5f}")

    mae_x = np.array([s["val_mae_x_phys"] for s in fold_summaries])
    mae_y = np.array([s["val_mae_y_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits, "model": "pure physics fit: w=w0*sqrt(2*ln(P/I_th)), independent per-axis I_th",
             "mean_val_mae_x_phys": float(mae_x.mean()), "std_val_mae_x_phys": float(mae_x.std()),
             "mean_val_mae_y_phys": float(mae_y.mean()), "std_val_mae_y_phys": float(mae_y.std()),
             "folds": fold_summaries}
    with open(os.path.join(out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"{out_dir}: MAE_x={mae_x.mean():.5f}+/-{mae_x.std():.5f}  MAE_y={mae_y.mean():.5f}+/-{mae_y.std():.5f}")


def run_z(split_dir, out_dir, n_splits):
    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []
    for fold in range(1, n_splits + 1):
        train_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_train.csv"))
        val_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_val.csv"))
        zr, ith, b = fit_axial(train_df["power_mW"].values.astype(np.float64),
                                train_df["z_resolution"].values.astype(np.float64))
        y_val = val_df["z_resolution"].values.astype(np.float32)
        pred_val = axial_model(val_df["power_mW"].values.astype(np.float64), zr, ith, b).astype(np.float32)
        mae = mean_absolute_error(y_val, pred_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        pd.DataFrame({"video_id": val_df["video_id"].values, "power_mW": val_df["power_mW"].values,
                       "true_z_phys": y_val, "pred_z_phys": pred_val,
                       "abs_err_z_phys": np.abs(pred_val - y_val)}).to_csv(
            os.path.join(out_dir, f"fold_{fold}_predictions.csv"), index=False)
        fold_summaries.append({"fold": fold, "n_train": len(train_df), "n_val": len(val_df),
                                "zr": float(zr), "i_th_mW": float(ith), "b": float(b),
                                "val_mae_z_phys": float(mae), "val_rmse_z_phys": rmse})
        print(f"{out_dir} fold {fold}: z={mae:.5f}")

    mae_z = np.array([s["val_mae_z_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits, "model": "pure physics fit: z=zr*sqrt((P/I_th)^(1/2)-1)+b, independent I_th",
             "mean_val_mae_z_phys": float(mae_z.mean()), "std_val_mae_z_phys": float(mae_z.std()),
             "folds": fold_summaries}
    with open(os.path.join(out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"{out_dir}: MAE_z={mae_z.mean():.5f}+/-{mae_z.std():.5f}")


def main():
    run_xy(os.path.join(ROOT, "data", "processed", "kfold_splits_xy"),
           os.path.join(ROOT, "results", "xy_baseline_physics"), 5)
    run_xy(os.path.join(ROOT, "data", "processed", "kfold_splits_xy_loo_batch"),
           os.path.join(ROOT, "results", "xy_baseline_physics_loo_batch"), 6)
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z"),
          os.path.join(ROOT, "results", "z_baseline_physics"), 5)
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z_loo_batch"),
          os.path.join(ROOT, "results", "z_baseline_physics_loo_batch"), 6)
    print("\nALL_BASELINE_PHYSICS_DONE")


if __name__ == "__main__":
    main()
