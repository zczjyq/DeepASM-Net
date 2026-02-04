import math
from typing import Tuple

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    psnr_val = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr_val.mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size: int, channel: int, device, dtype):
    g = _gaussian_window(window_size, 1.5, device, dtype)
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.repeat(channel, 1, 1, 1)
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    _, c, _, _ = pred.size()
    window = _create_window(window_size, c, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target, size_average=True)


def color_stats_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Match channel-wise mean and std to reduce color shift
    pred_mean = pred.mean(dim=(2, 3))
    tgt_mean = target.mean(dim=(2, 3))
    pred_std = pred.std(dim=(2, 3))
    tgt_std = target.std(dim=(2, 3))
    return torch.mean(torch.abs(pred_mean - tgt_mean)) + torch.mean(torch.abs(pred_std - tgt_std))


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Sobel edge loss on luminance
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)

    def to_luma(x: torch.Tensor) -> torch.Tensor:
        return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])

    pred_l = to_luma(pred)
    tgt_l = to_luma(target)
    pred_gx = F.conv2d(pred_l, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_l, sobel_y, padding=1)
    tgt_gx = F.conv2d(tgt_l, sobel_x, padding=1)
    tgt_gy = F.conv2d(tgt_l, sobel_y, padding=1)
    return torch.mean(torch.abs(pred_gx - tgt_gx) + torch.abs(pred_gy - tgt_gy))
