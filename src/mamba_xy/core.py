"""
Vendored core utilities -- self-contained copy of the small pieces of
printer_ml/src/printer_ml/mamba_early_prediction.py this project actually
needs (Mamba layer + seeding + inverse-transform), so this repo has no
dependency on printer_ml being present on whatever machine it runs on.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaSSM
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MAMBA_SSM_AVAILABLE = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _MambaLayerFallback(nn.Module):
    """Pure-PyTorch Mamba block, used when mamba_ssm (CUDA) isn't installed."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz            = self.in_proj(x)
        x_inner, z    = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.permute(0, 2, 1))[:, :, :T].permute(0, 2, 1)
        x_conv = F.silu(x_conv)
        ssm    = self.x_proj(x_conv)
        dt_raw, B_p, C_p = ssm[..., :1], ssm[..., 1:self.d_state+1], ssm[..., self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt_raw))
        A  = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_p.unsqueeze(2)
        h, ys = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype), []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            ys.append((h * C_p[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x_conv * self.D
        return self.out_proj(y * F.silu(z))


def build_mamba_layer(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    """Returns the best available Mamba layer for this environment."""
    if _MAMBA_SSM_AVAILABLE:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _MambaLayerFallback(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


def inverse_transform(y_norm, target_mean: float, target_std: float):
    """No log transform -- x/y resolution is trained in raw physical scale."""
    return np.asarray(y_norm) * target_std + target_mean
