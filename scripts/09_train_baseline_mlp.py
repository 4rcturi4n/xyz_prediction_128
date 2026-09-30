# scripts/09_train_baseline_mlp.py
#
# Power-only baseline: small ReLU MLP on a physics-shaped hint feature
# (sqrt(ln(P)) for x/y, sqrt(sqrt(P)-1) for z), no video at all, no P_th
# ever computed. Both split schemes. Non-temporal (single prediction per
# video), so no learning-curve/trajectory report applies here -- just
# MAE/RMSE per fold, matching this baseline's own established format.

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 42
EPOCHS = 3000
PATIENCE = 300
LR = 1e-2
WEIGHT_DECAY = 1e-4


def hint_feature_xy(power_mW):
    return np.sqrt(np.log(np.asarray(power_mW, dtype=np.float32)))


def hint_feature_z(power_mW):
    ratio = np.clip(np.sqrt(np.asarray(power_mW, dtype=np.float32)) - 1.0, 1e-6, None)
    return np.sqrt(ratio)


def make_mlp():
    return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))


def fit_mlp(u_train, y_train, u_val, y_val, seed):
    torch.manual_seed(seed)
    model = make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    u_train_t = torch.tensor(u_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    u_val_t = torch.tensor(u_val, dtype=torch.float32).unsqueeze(1)

    best_val_mae, best_state, epochs_no_imp = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(u_train_t), y_train_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(u_val_t).numpy().ravel()
        val_mae = mean_absolute_error(y_val, pred_val)
        if val_mae < best_val_mae - 1e-6:
            best_val_mae, best_state, epochs_no_imp = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_imp += 1
        if epochs_no_imp >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val_final = model(u_val_t).numpy().ravel()
    return pred_val_final


def run_xy(split_dir, out_dir, n_splits):
    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []
    for fold in range(1, n_splits + 1):
        train_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_train.csv"))
        val_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_val.csv"))
        u_train = hint_feature_xy(train_df["power_mW"].values)
        u_val = hint_feature_xy(val_df["power_mW"].values)

        summary = {"fold": fold, "n_train": len(train_df), "n_val": len(val_df)}
        pred_cols = {"video_id": val_df["video_id"].values, "power_mW": val_df["power_mW"].values}
        for axis in ("x", "y"):
            y_train = train_df[f"{axis}_resolution"].values.astype(np.float32)
            y_val = val_df[f"{axis}_resolution"].values.astype(np.float32)
            pred_val = fit_mlp(u_train, y_train, u_val, y_val, seed=SEED + fold)
            mae = mean_absolute_error(y_val, pred_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            pred_cols[f"true_{axis}_phys"] = y_val
            pred_cols[f"pred_{axis}_phys"] = pred_val
            pred_cols[f"abs_err_{axis}_phys"] = np.abs(pred_val - y_val)
            summary[f"val_mae_{axis}_phys"] = float(mae)
            summary[f"val_rmse_{axis}_phys"] = rmse
        fold_summaries.append(summary)
        pd.DataFrame(pred_cols).to_csv(os.path.join(out_dir, f"fold_{fold}_predictions.csv"), index=False)
        print(f"{out_dir} fold {fold}: x={summary['val_mae_x_phys']:.5f} y={summary['val_mae_y_phys']:.5f}")

    mae_x = np.array([s["val_mae_x_phys"] for s in fold_summaries])
    mae_y = np.array([s["val_mae_y_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits, "model": "small ReLU MLP, hint sqrt(ln(P)), no P_th",
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
        u_train = hint_feature_z(train_df["power_mW"].values)
        u_val = hint_feature_z(val_df["power_mW"].values)
        y_train = train_df["z_resolution"].values.astype(np.float32)
        y_val = val_df["z_resolution"].values.astype(np.float32)

        pred_val = fit_mlp(u_train, y_train, u_val, y_val, seed=SEED + fold)
        mae = mean_absolute_error(y_val, pred_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))

        pred_df = pd.DataFrame({"video_id": val_df["video_id"].values, "power_mW": val_df["power_mW"].values,
                                 "true_z_phys": y_val, "pred_z_phys": pred_val, "abs_err_z_phys": np.abs(pred_val - y_val)})
        pred_df.to_csv(os.path.join(out_dir, f"fold_{fold}_predictions.csv"), index=False)
        fold_summaries.append({"fold": fold, "n_train": len(train_df), "n_val": len(val_df),
                                "val_mae_z_phys": float(mae), "val_rmse_z_phys": rmse})
        print(f"{out_dir} fold {fold}: z={mae:.5f}")

    mae_z = np.array([s["val_mae_z_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits, "model": "small ReLU MLP, hint sqrt(sqrt(P)-1), no P_th",
             "mean_val_mae_z_phys": float(mae_z.mean()), "std_val_mae_z_phys": float(mae_z.std()),
             "folds": fold_summaries}
    with open(os.path.join(out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"{out_dir}: MAE_z={mae_z.mean():.5f}+/-{mae_z.std():.5f}")


def main():
    run_xy(os.path.join(ROOT, "data", "processed", "kfold_splits_xy"),
           os.path.join(ROOT, "results", "xy_baseline_mlp"), 5)
    run_xy(os.path.join(ROOT, "data", "processed", "kfold_splits_xy_loo_batch"),
           os.path.join(ROOT, "results", "xy_baseline_mlp_loo_batch"), 6)
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z"),
          os.path.join(ROOT, "results", "z_baseline_mlp"), 5)
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z_loo_batch"),
          os.path.join(ROOT, "results", "z_baseline_mlp_loo_batch"), 6)
    print("\nALL_BASELINE_MLP_DONE")


if __name__ == "__main__":
    main()
