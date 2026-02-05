"""
PhaseModule：相位预测模块

从 RGB 输入预测复振幅场的相位 phi ∈ [-π, π]。
采用共享相位 + 通道残差，保证跨通道一致性的同时保留通道差异。
"""
import math
import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """标准残差块：Conv -> Norm -> Act -> Conv -> Norm -> (+ residual) -> Act"""

    def __init__(self, ch: int, use_gn: bool = True):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, ch) if use_gn else nn.Identity()
        self.norm2 = nn.GroupNorm(1, ch) if use_gn else nn.Identity()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class PhaseModule(nn.Module):
    """
    相位预测模块

    输出：phi = π * tanh(phi_shared + eps * delta_phi)
    - phi_shared: 1 通道，跨 RGB 共享
    - delta_phi: 3 通道，通道间残差，由 residual_scale 控制幅度
    通过 tanh 将相位约束在 [-π, π]。
    """

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 16,
        num_layers: int = 2,
        use_gn: bool = True,
        use_channel_residual: bool = True,
        residual_scale: float = 0.1,
    ):
        """
        Args:
            in_ch: 输入通道数
            hidden: 隐藏层通道数
            num_layers: ResBlock 层数
            use_gn: 是否使用 GroupNorm
            use_channel_residual: 是否使用通道残差 delta_phi
            residual_scale: delta_phi 的缩放系数，通常较小（如 0.1）
        """
        super().__init__()
        self.use_channel_residual = use_channel_residual
        self.residual_scale = residual_scale

        self.in_proj = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(hidden, use_gn=use_gn) for _ in range(max(1, num_layers))])
        self.out_shared = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)   # 共享相位
        self.out_resid = nn.Conv2d(hidden, in_ch, kernel_size=3, padding=1)  # 通道残差

        self.last_phi_shared = None
        self.last_delta_phi = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, in_ch, H, W)

        Returns:
            phi: 相位 (B, 3, H, W)，范围 [-π, π]
        """
        x = self.in_proj(x)
        x = self.blocks(x)
        phi_shared = self.out_shared(x)   # (B, 1, H, W)
        if self.use_channel_residual:
            delta_phi = self.out_resid(x)  # (B, 3, H, W)
        else:
            delta_phi = torch.zeros_like(x[:, :3])

        # 先相加再 tanh，保证有界
        phi_raw = phi_shared + self.residual_scale * delta_phi
        phi = math.pi * torch.tanh(phi_raw)

        self.last_phi_shared = phi_shared
        self.last_delta_phi = delta_phi
        return phi
