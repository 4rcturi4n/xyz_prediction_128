# scripts/19_train_baseline_mlp_z_newhint.py
#
# Baseline MLP for z, using the new hint formula (src/mamba_xy/hint_z_newhint.py:
# sqrt(sqrt(ln(P/I_th))-1), I_th fit per fold TRAIN-only) instead of the
# original sqrt(sqrt(P)-1) in 09_train_baseline_mlp.py. Same structure as
# that script's run_z, just the hint source differs. Separate output
# folders (xy_baseline_mlp untouched).

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from mamba_xy.hint_z_newhint import fit_i_th_for_hint, make_baseline_mlp, hint_feature
import torch
import torch.nn as nn

SEED = 42
EPOCHS = 3000
PATIENCE = 300
LR = 1e-2
WEIGHT_DECAY = 1e-4


def fit_mlp(u_train, y_train, u_val, y_val, seed):
    torch.manual_seed(seed)
    model = make_baseline_mlp()
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


def run_z(split_dir, out_dir, n_splits):
    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []
    for fold in range(1, n_splits + 1):
        train_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_train.csv"))
        val_df = pd.read_csv(os.path.join(split_dir, f"fold_{fold}_val.csv"))

        i_th = fit_i_th_for_hint(train_df["power_mW"].values.astype(np.float64),
                                  train_df["z_resolution"].values.astype(np.float64))
        u_train = hint_feature(train_df["power_mW"].values, i_th)
        u_val = hint_feature(val_df["power_mW"].values, i_th)
        y_train = train_df["z_resolution"].values.astype(np.float32)
        y_val = val_df["z_resolution"].values.astype(np.float32)

        pred_val = fit_mlp(u_train, y_train, u_val, y_val, seed=SEED + fold)
        mae = mean_absolute_error(y_val, pred_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))

        pred_df = pd.DataFrame({"video_id": val_df["video_id"].values, "power_mW": val_df["power_mW"].values,
                                 "true_z_phys": y_val, "pred_z_phys": pred_val, "abs_err_z_phys": np.abs(pred_val - y_val)})
        pred_df.to_csv(os.path.join(out_dir, f"fold_{fold}_predictions.csv"), index=False)
        fold_summaries.append({"fold": fold, "n_train": len(train_df), "n_val": len(val_df), "i_th_mW": i_th,
                                "val_mae_z_phys": float(mae), "val_rmse_z_phys": rmse})
        print(f"{out_dir} fold {fold}: z={mae:.5f} (I_th={i_th:.4f} mW)")

    mae_z = np.array([s["val_mae_z_phys"] for s in fold_summaries])
    final = {"n_splits": n_splits, "model": "small ReLU MLP, hint sqrt(sqrt(ln(P/I_th))-1), I_th fit per fold",
             "mean_val_mae_z_phys": float(mae_z.mean()), "std_val_mae_z_phys": float(mae_z.std()),
             "folds": fold_summaries}
    with open(os.path.join(out_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"{out_dir}: MAE_z={mae_z.mean():.5f}+/-{mae_z.std():.5f}")


def main():
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z"),
          os.path.join(ROOT, "results", "z_baseline_mlp_newhint"), 5)
    run_z(os.path.join(ROOT, "data", "processed", "kfold_splits_z_loo_batch"),
          os.path.join(ROOT, "results", "z_baseline_mlp_newhint_loo_batch"), 6)
    print("\nALL_BASELINE_MLP_NEWHINT_DONE")


if __name__ == "__main__":
    main()
