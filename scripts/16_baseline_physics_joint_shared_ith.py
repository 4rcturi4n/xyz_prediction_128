# scripts/16_baseline_physics_joint_shared_ith.py
#
# Joint physics fit with a SINGLE shared I_th across x, y, z simultaneously
# (physically: one material/laser threshold power), on this repo's
# 106-video matched splits. Same approach as
# xy_early_prediction/scripts/23_joint_physics_fit_shared_ith.py, repointed
# here. Confirmed on the original dataset that forcing a shared I_th costs
# essentially nothing in accuracy vs independent per-axis fits (within 1
# std on every axis) -- this is the version that belongs in the paper's
# results table, matched to the same 106 videos as every other model.
#
#   x = w0_x * sqrt(2*ln(P/I_th))
#   y = w0_y * sqrt(2*ln(P/I_th))
#   z = zr   * sqrt((P/I_th)^(1/2) - 1) + b
#
# Both split schemes. Pairs fold N of kfold_splits_xy(_loo_batch) with
# fold N of kfold_splits_z(_loo_batch) -- same 106 videos, same fold
# membership by construction (xy folds were built to match z's canonical
# fold membership), so this pairing is exact, not approximate.

import os
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_ORDER = 2


def lateral_model(P, w0, ith):
    ratio = np.clip(P / ith, 1.0 + 1e-9, None)
    return w0 * np.sqrt(2.0 * np.log(ratio))


def axial_model(P, zr, ith, b):
    ratio = np.clip((P / ith) ** (1.0 / N_ORDER), 1.0 + 1e-9, None)
    return zr * np.sqrt(ratio - 1.0) + b


def residuals(params, Px, yx, Py, yy, Pz, yz):
    w0_x, w0_y, zr, b, ith = params
    return np.concatenate([
        lateral_model(Px, w0_x, ith) - yx,
        lateral_model(Py, w0_y, ith) - yy,
        axial_model(Pz, zr, ith, b) - yz,
    ])


def fit_joint(Px, yx, Py, yy, Pz, yz):
    p_min = float(min(Px.min(), Py.min(), Pz.min()))
    x0 = [float(yx.max() - yx.min()), float(yy.max() - yy.min()),
          float(yz.max() - yz.min()), float(yz.min()), 0.5 * p_min]
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 1e-3],
              [np.inf, np.inf, np.inf, np.inf, p_min * 0.999])
    res = least_squares(residuals, x0, args=(Px, yx, Py, yy, Pz, yz), bounds=bounds, max_nfev=20000)
    return res.x


def run(xy_split_dir, z_split_dir, out_dir, n_splits):
    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []
    for fold in range(1, n_splits + 1):
        xy_train = pd.read_csv(os.path.join(xy_split_dir, f"fold_{fold}_train.csv"))
        xy_val = pd.read_csv(os.path.join(xy_split_dir, f"fold_{fold}_val.csv"))
        z_train = pd.read_csv(os.path.join(z_split_dir, f"fold_{fold}_train.csv"))
        z_val = pd.read_csv(os.path.join(z_split_dir, f"fold_{fold}_val.csv"))

        Px = xy_train["power_mW"].values.astype(np.float64)
        yx = xy_train["x_resolution"].values.astype(np.float64)
        yy = xy_train["y_resolution"].values.astype(np.float64)
        Pz = z_train["power_mW"].values.astype(np.float64)
        yz = z_train["z_resolution"].values.astype(np.float64)

        w0_x, w0_y, zr, b, ith = fit_joint(Px, yx, Px, yy, Pz, yz)

        pred_x = lateral_model(xy_val["power_mW"].values.astype(np.float64), w0_x, ith)
        pred_y = lateral_model(xy_val["power_mW"].values.astype(np.float64), w0_y, ith)
        pred_z = axial_model(z_val["power_mW"].values.astype(np.float64), zr, ith, b)
        true_x = xy_val["x_resolution"].values.astype(np.float32)
        true_y = xy_val["y_resolution"].values.astype(np.float32)
        true_z = z_val["z_resolution"].values.astype(np.float32)

        mae_x = mean_absolute_error(true_x, pred_x)
        mae_y = mean_absolute_error(true_y, pred_y)
        mae_z = mean_absolute_error(true_z, pred_z)

        pd.DataFrame({"video_id": xy_val["video_id"].values, "power_mW": xy_val["power_mW"].values,
                       "true_x_phys": true_x, "pred_x_phys": pred_x, "abs_err_x_phys": np.abs(pred_x - true_x),
                       "true_y_phys": true_y, "pred_y_phys": pred_y, "abs_err_y_phys": np.abs(pred_y - true_y),
                       }).to_csv(os.path.join(out_dir, f"fold_{fold}_xy_predictions.csv"), index=False)
        pd.DataFrame({"video_id": z_val["video_id"].values, "power_mW": z_val["power_mW"].values,
                       "true_z_phys": true_z, "pred_z_phys": pred_z, "abs_err_z_phys": np.abs(pred_z - true_z),
                       }).to_csv(os.path.join(out_dir, f"fold_{fold}_z_predictions.csv"), index=False)

        fold_summaries.append({"fold": fold, "w0_x": float(w0_x), "w0_y": float(w0_y),
                                "zr": float(zr), "b": float(b), "shared_i_th_mW": float(ith),
                                "val_mae_x_phys": float(mae_x), "val_mae_y_phys": float(mae_y),
                                "val_mae_z_phys": float(mae_z)})
        print(f"{out_dir} fold {fold}: shared I_th={ith:.4f} mW | x={mae_x:.5f} y={mae_y:.5f} z={mae_z:.5f}")

    mae_x = np.array([s["val_mae_x_phys"] for s in fold_summaries])
    mae_y = np.array([s["val_mae_y_phys"] for s in fold_summaries])
    mae_z = np.array([s["val_mae_z_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits, "model": "joint physics fit, single shared I_th across x/y/z",
             "mean_val_mae_x_phys": float(mae_x.mean()), "std_val_mae_x_phys": float(mae_x.std()),
             "mean_val_mae_y_phys": float(mae_y.mean()), "std_val_mae_y_phys": float(mae_y.std()),
             "mean_val_mae_z_phys": float(mae_z.mean()), "std_val_mae_z_phys": float(mae_z.std()),
             "folds": fold_summaries}
    with open(os.path.join(out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"{out_dir}: MAE_x={mae_x.mean():.5f}+/-{mae_x.std():.5f}  MAE_y={mae_y.mean():.5f}+/-{mae_y.std():.5f}  "
          f"MAE_z={mae_z.mean():.5f}+/-{mae_z.std():.5f}")


def main():
    run(os.path.join(ROOT, "data", "processed", "kfold_splits_xy"),
        os.path.join(ROOT, "data", "processed", "kfold_splits_z"),
        os.path.join(ROOT, "results", "baseline_physics_joint_shared_ith"), 5)
    run(os.path.join(ROOT, "data", "processed", "kfold_splits_xy_loo_batch"),
        os.path.join(ROOT, "data", "processed", "kfold_splits_z_loo_batch"),
        os.path.join(ROOT, "results", "baseline_physics_joint_shared_ith_loo_batch"), 6)
    print("\nALL_BASELINE_PHYSICS_JOINT_DONE")


if __name__ == "__main__":
    main()
