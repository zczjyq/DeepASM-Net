"""
角谱法（Angular Spectrum Method, ASM）传播模块

在频域实现波动传播：Uz = IFFT2( Enhance( FFT2(U0) * H ) )
其中 H 为角谱传递函数，Enhance 为可选的频域增强模块。
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


# 频率网格缓存，避免重复计算
_FREQ_CACHE = {}


def get_freq_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 2D 空间频率网格 (fx, fy)，用于 FFT

    Args:
        h, w: 图像高度和宽度
        device, dtype: 目标设备和数据类型

    Returns:
        fx_grid, fy_grid: (1, 1, h, w) 形状，与 FFT 输出对齐
    """
    key = (h, w, device.type, device.index, dtype)
    if key in _FREQ_CACHE:
        return _FREQ_CACHE[key]

    # fftfreq: 返回 FFT 对应的归一化频率
    fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    fy_grid = fy_grid[None, None, :, :]
    fx_grid = fx_grid[None, None, :, :]

    _FREQ_CACHE[key] = (fx_grid, fy_grid)
    return fx_grid, fy_grid


class FreqEnhance(nn.Module):
    """
    频域增强模块：在 FFT*H 之后、IFFT 之前对复振幅进行可学习增强。
    残差形式 U_out = U + scale * delta(U)，scale 初始为 0 保证训练稳定。
    """

    def __init__(self, channels: int = 3, hidden: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.0))  # 残差缩放，初始 0 = 恒等

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """U: (B, C, H, W) complex -> enhanced (B, C, H, W) complex"""
        B, C, H, W = U.shape
        ri = torch.cat([U.real, U.imag], dim=1)
        delta = self.conv(ri)
        r_d, i_d = delta.chunk(2, dim=1)
        delta_c = torch.complex(r_d, i_d)
        return U + self.scale * delta_c


def asm_propagate(
    U0: torch.Tensor,
    z: torch.Tensor,
    wavelengths: torch.Tensor,
    output_mode: str = "abs",
    clamp_sqrt: bool = True,
    enhance_module: Optional[nn.Module] = None,
):
    """
    角谱法（ASM）频域传播

    将复振幅场 U0 沿 z 方向传播距离 z，得到 Uz。
    传递函数 H = exp(i * k * z * sqrt(1 - (λ*fx)^2 - (λ*fy)^2))

    Args:
        U0: 初始复振幅场 (B, C, H, W)，complex
        z: 传播距离 (B, C, H, W) 空间变化，或 (B, C) 通道标量
        wavelengths: 每通道波长 (C,) 或 (1, C, 1, 1)
        output_mode: "abs" 输出 |Uz|，"abs2" 输出 |Uz|^2
        clamp_sqrt: 是否将 evanescent 区（sqrt 内<0）截断为 0
        enhance_module: 可选，在 FFT*H 后、IFFT 前对频域复振幅做增强

    Returns:
        Uz: 传播后复振幅 (B, C, H, W)
        J: 传播后强度 (B, C, H, W)
    """
    if not torch.is_complex(U0):
        raise ValueError("U0 must be complex tensor")

    b, c, h, w = U0.shape
    device = U0.device
    dtype = U0.real.dtype

    fx, fy = get_freq_grid(h, w, device, dtype)
    wavelengths = wavelengths.to(device=device, dtype=dtype).view(1, c, 1, 1)
    z = z.to(device=device, dtype=dtype)
    if z.ndim == 2:
        # 通道标量 z: (B, C) -> (B, C, 1, 1)
        z = z.view(b, c, 1, 1)
    # 否则 z 已是 (B, C, H, W)，保持空间变化

    # 波数 k = 2π/λ
    k = 2.0 * math.pi / wavelengths
    # 角谱根号内：1 - (λ*fx)^2 - (λ*fy)^2，<0 时为倏逝波
    inside = 1.0 - (wavelengths * fx) ** 2 - (wavelengths * fy) ** 2

    if clamp_sqrt:
        # 截断倏逝波区，避免复数 sqrt
        inside = torch.clamp(inside, min=0.0)

    sqrt_term = torch.sqrt(inside)
    phase = k * z * sqrt_term
    H = torch.exp(1j * phase)

    # 频域相乘：U_f_h = FFT(U0) * H
    U_f_h = torch.fft.fft2(U0) * H

    # 可选：频域增强后再 IFFT
    if enhance_module is not None:
        U_f_h = enhance_module(U_f_h)

    Uz = torch.fft.ifft2(U_f_h)

    if output_mode == "abs":
        J = torch.abs(Uz)
    elif output_mode == "abs2":
        J = torch.abs(Uz) ** 2
    else:
        raise ValueError("output_mode must be 'abs' or 'abs2'")

    return Uz, J
