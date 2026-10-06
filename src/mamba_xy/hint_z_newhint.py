"""
New z hint formula: u = sqrt(sqrt(ln(P/I_th)) - 1), replacing the
original sqrt(sqrt(P)-1) (fused_hint_z.hint_feature). Unlike the
original, this needs I_th -- fit per fold, TRAIN-only, independently for
this hint (not reusing the joint shared-I_th physics fit), via
curve_fit on z = zr*u(P,I_th) + b against that fold's train z_resolution,
keeping only I_th (same convention as fit_i_th_axial elsewhere in this
codebase -- zr/b are nuisance parameters here, discarded after fitting).

Domain note: ln(P/I_th) > 0 needs P > I_th; sqrt(ln(P/I_th)) > 1 needs
P > I_th*e for the outer sqrt to be real. Both are clipped rather than
raising, same style as every other hint/physics function in this repo.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error


def hint_feature(power_mW: np.ndarray, i_th: float) -> np.ndarray:
    P = np.asarray(power_mW, dtype=np.float64)
    log_ratio = np.clip(np.log(np.clip(P / i_th, 1e-9, None)), 1e-9, None)
    inner = np.clip(np.sqrt(log_ratio) - 1.0, 1e-6, None)
    return np.sqrt(inner).astype(np.float32)


def _model(P, zr, i_th, b):
    log_ratio = np.clip(np.log(np.clip(P / i_th, 1e-9, None)), 1e-9, None)
    inner = np.clip(np.sqrt(log_ratio) - 1.0, 1e-6, None)
    return zr * np.sqrt(inner) + b


def fit_i_th_for_hint(power_mW: np.ndarray, target_phys: np.ndarray) -> float:
    """TRAIN-only, leak-free. Returns just I_th (zr/b discarded, same
    convention as fit_i_th_axial)."""
    p_min = float(power_mW.min())
    zr0 = float(target_phys.max() - target_phys.min())
    b0 = float(target_phys.min())
    ith0 = p_min / np.e / 2.0  # below p_min/e so ln(P/I_th) > 1 is reachable for most of the range

    bounds = ([-np.inf, 1e-3, -np.inf], [np.inf, p_min / np.e * 0.999, np.inf])
    popt, _ = curve_fit(_model, power_mW, target_phys, p0=[zr0, ith0, b0], bounds=bounds, maxfev=20000)
    return float(popt[1])


def make_baseline_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))


def _fit_mlp(u_train, y_train, u_val, y_val, seed, epochs=3000, patience=300, lr=1e-2, weight_decay=1e-4):
    torch.manual_seed(seed)
    model = make_baseline_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    u_train_t = torch.tensor(u_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    u_val_t = torch.tensor(u_val, dtype=torch.float32).unsqueeze(1)

    best_val_mae, best_state, epochs_no_imp = float("inf"), None, 0
    for _ in range(epochs):
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
        if epochs_no_imp >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def fit_baseline_mlp_no_leak(power_train: np.ndarray, y_train: np.ndarray, i_th: float, seed: int,
                              holdout_frac: float = 0.2) -> nn.Module:
    rng = np.random.default_rng(seed)
    n = len(power_train)
    idx = rng.permutation(n)
    n_holdout = max(1, int(n * holdout_frac))
    holdout_idx, inner_idx = idx[:n_holdout], idx[n_holdout:]

    u = hint_feature(power_train, i_th)
    return _fit_mlp(u[inner_idx], y_train[inner_idx].astype(np.float32),
                     u[holdout_idx], y_train[holdout_idx].astype(np.float32), seed=seed)


@torch.no_grad()
def baseline_predict(model: nn.Module, power_mW: np.ndarray, i_th: float) -> np.ndarray:
    u = hint_feature(power_mW, i_th)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
    return model(u_t).numpy().ravel()
