# scripts/27_single_video_overfit_test.py
#
# Pipeline sanity check, not a generalization test: train fused,
# video-only, and baseline-MLP on ONE video (train=val=that same
# video) for z and xy. If a model can't drive its error on that single
# video near zero, something in the training loop/architecture is
# broken. Run on the server (GPU) -- on a local Windows GPU the
# per-call Mamba kernel overhead across ~2000 tiny forward/backward
# calls makes this noticeably slow even though there's only one video.

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from mamba_xy.core import inverse_transform, set_seed
from mamba_xy.mamba_z_regression import (
    CachedMambaEmbeddingDatasetZ, MambaSequenceRegressorZWithPower, MambaSequenceRegressorZ,
    fit_i_th_axial, compute_loss_z,
)
from mamba_xy.hint_z_newhint import hint_feature as hint_feature_z, fit_i_th_for_hint as fit_i_th_hint_z, make_baseline_mlp
from mamba_xy.mamba_xy_regression import (
    CachedMambaEmbeddingDatasetXY, MambaSequenceRegressorXYWithPower, MambaSequenceRegressorXY,
    fit_i_th_xy, compute_loss_xy,
)

EPOCHS = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def make_simple_mlp():
    return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))


def run_z():
    set_seed(42)
    train_path = os.path.join(ROOT, "results", "dinov2_embeddings_z_with_power", "fold_1", "train_embeddings.pt")
    train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
    power_all = train_payload["power_mW"].numpy()
    target_all = train_payload["targets_z_phys"].numpy()
    ith = fit_i_th_axial(power_all, target_all)

    full_ds = CachedMambaEmbeddingDatasetZ(train_path, ith)
    vid = full_ds.video_ids[0]
    true_phys = float(full_ds.targets_z_phys[0])
    print(f"\n=== Z | single video: {vid} | true z = {true_phys:.5f} um | I_th={ith:.4f} ===")

    sub = Subset(full_ds, [0])
    loader = DataLoader(sub, batch_size=1)
    embed_dim = full_ds.embeddings.shape[-1]
    tmean, tstd = full_ds.target_z_mean, full_ds.target_z_std
    loss_fn = nn.SmoothL1Loss()

    model = MambaSequenceRegressorZWithPower(embed_dim=embed_dim, n_mamba_layers=2, d_state=16, d_conv=4,
                                              expand=2, hidden_dim=256, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            emb, tnorm, pnorm = emb.to(device), tnorm.to(device), pnorm.to(device)
            opt.zero_grad()
            preds = model(emb, pnorm)
            loss = compute_loss_z(preds, tnorm, loss_fn)
            loss.backward()
            opt.step()
    with torch.no_grad():
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            pred_norm = model(emb.to(device), pnorm.to(device))[0, -1, 0].item()
    pred_phys = inverse_transform(pred_norm, tmean, tstd)
    print(f"fused:       pred={pred_phys:.5f}  true={true_phys:.5f}  abs_err={abs(pred_phys - true_phys):.6f}")

    model = MambaSequenceRegressorZ(embed_dim=embed_dim, n_mamba_layers=2, d_state=16, d_conv=4,
                                     expand=2, hidden_dim=256, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            emb, tnorm = emb.to(device), tnorm.to(device)
            opt.zero_grad()
            preds = model(emb)
            loss = compute_loss_z(preds, tnorm, loss_fn)
            loss.backward()
            opt.step()
    with torch.no_grad():
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            pred_norm = model(emb.to(device))[0, -1, 0].item()
    pred_phys = inverse_transform(pred_norm, tmean, tstd)
    print(f"video-only:  pred={pred_phys:.5f}  true={true_phys:.5f}  abs_err={abs(pred_phys - true_phys):.6f}")

    ith_hint = fit_i_th_hint_z(power_all, target_all)
    u_single = hint_feature_z(np.array([power_all[0]]), ith_hint)
    y_single = np.array([target_all[0]], dtype=np.float32)
    model = make_baseline_mlp().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn2 = nn.SmoothL1Loss()
    u_t = torch.tensor(u_single, dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(y_single, dtype=torch.float32).unsqueeze(1).to(device)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn2(model(u_t), y_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_phys = model(u_t).item()
    print(f"baseline MLP: pred={pred_phys:.5f}  true={true_phys:.5f}  abs_err={abs(pred_phys - true_phys):.6f}")


def run_xy():
    set_seed(42)
    train_path = os.path.join(ROOT, "results", "dinov2_embeddings_xy_with_power", "fold_1", "train_embeddings.pt")
    train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
    power_all = train_payload["power_mW"].numpy()
    x_all = train_payload["targets_x_phys"].numpy()
    y_all = train_payload["targets_y_phys"].numpy()
    ith = fit_i_th_xy(power_all, x_all, y_all)

    full_ds = CachedMambaEmbeddingDatasetXY(train_path, ith)
    vid = full_ds.video_ids[0]
    true_x = float(full_ds.targets_x_phys[0])
    true_y = float(full_ds.targets_y_phys[0])
    print(f"\n=== XY | single video: {vid} | true x={true_x:.5f} y={true_y:.5f} um | I_th={ith:.4f} ===")

    sub = Subset(full_ds, [0])
    loader = DataLoader(sub, batch_size=1)
    embed_dim = full_ds.embeddings.shape[-1]
    xmean, xstd = full_ds.target_x_mean, full_ds.target_x_std
    ymean, ystd = full_ds.target_y_mean, full_ds.target_y_std
    loss_fn = nn.SmoothL1Loss()

    model = MambaSequenceRegressorXYWithPower(embed_dim=embed_dim, n_mamba_layers=2, d_state=16, d_conv=4,
                                               expand=2, hidden_dim=256, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            emb, tnorm, pnorm = emb.to(device), tnorm.to(device), pnorm.to(device)
            opt.zero_grad()
            preds = model(emb, pnorm)
            loss = compute_loss_xy(preds, tnorm, loss_fn)
            loss.backward()
            opt.step()
    with torch.no_grad():
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            preds = model(emb.to(device), pnorm.to(device))[0, -1]
    pred_x = inverse_transform(preds[0].item(), xmean, xstd)
    pred_y = inverse_transform(preds[1].item(), ymean, ystd)
    print(f"fused:       pred_x={pred_x:.5f} (true {true_x:.5f}, err {abs(pred_x - true_x):.6f})  "
          f"pred_y={pred_y:.5f} (true {true_y:.5f}, err {abs(pred_y - true_y):.6f})")

    model = MambaSequenceRegressorXY(embed_dim=embed_dim, n_mamba_layers=2, d_state=16, d_conv=4,
                                      expand=2, hidden_dim=256, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            emb, tnorm = emb.to(device), tnorm.to(device)
            opt.zero_grad()
            preds = model(emb)
            loss = compute_loss_xy(preds, tnorm, loss_fn)
            loss.backward()
            opt.step()
    with torch.no_grad():
        for emb, tnorm, tphys, lw, pnorm, vids in loader:
            preds = model(emb.to(device))[0, -1]
    pred_x = inverse_transform(preds[0].item(), xmean, xstd)
    pred_y = inverse_transform(preds[1].item(), ymean, ystd)
    print(f"video-only:  pred_x={pred_x:.5f} (true {true_x:.5f}, err {abs(pred_x - true_x):.6f})  "
          f"pred_y={pred_y:.5f} (true {true_y:.5f}, err {abs(pred_y - true_y):.6f})")

    def hint_feature_xy(power_mW):
        return np.sqrt(np.log(np.asarray(power_mW, dtype=np.float32)))

    u_single = hint_feature_xy(np.array([power_all[0]]))
    u_t = torch.tensor(u_single, dtype=torch.float32).unsqueeze(1).to(device)

    for axis_name, true_val in [("x", true_x), ("y", true_y)]:
        y_t = torch.tensor([true_val], dtype=torch.float32).unsqueeze(1).to(device)
        model = make_simple_mlp().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn2 = nn.SmoothL1Loss()
        for epoch in range(EPOCHS):
            opt.zero_grad()
            loss = loss_fn2(model(u_t), y_t)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_phys = model(u_t).item()
        print(f"baseline MLP ({axis_name}): pred={pred_phys:.5f}  true={true_val:.5f}  "
              f"abs_err={abs(pred_phys - true_val):.6f}")


def main():
    run_z()
    run_xy()
    print("\nSINGLE_VIDEO_OVERFIT_TEST_DONE")


if __name__ == "__main__":
    main()
