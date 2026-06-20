"""Pure-PyTorch drop-in for the `fused_ssim` CUDA extension.

Copy this file into gsplat's `examples/` as `fused_ssim.py` so simple_trainer's
`from fused_ssim import fused_ssim` resolves to it. Needed because the upstream
fused-ssim wheel fails to build against torch 2.9.1 + MSVC (C2872 'std' ambiguous
in compiled_autograd.h). Same call signature; both A/B arms use it identically so
the comparison stays single-variable. See GSPLAT_SETUP.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channel: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w2d = g[:, None] @ g[None, :]
    return w2d.expand(channel, 1, window_size, window_size).contiguous()


def fused_ssim(img1, img2, padding: str = "same", train: bool = True):
    """img1, img2: [B, C, H, W] in [0, 1]. Returns mean SSIM (scalar tensor)."""
    channel = img1.shape[1]
    window_size = 11
    win = _gaussian_window(window_size, 1.5, channel, img1.device, img1.dtype)
    pad = window_size // 2 if padding == "same" else 0
    mu1 = F.conv2d(img1, win, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, win, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, win, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, win, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()
