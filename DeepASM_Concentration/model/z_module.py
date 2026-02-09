"""
ZModule：传播距离预测模块

预测空间变化、逐通道的传播距离 z ∈ [0, z_max]。
雾浓度与场景深度相关，z 越大表示该位置"雾越厚"，ASM 传播模拟去雾效果。
"""
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


class ZModule(nn.Module):
    """
    传播距离预测模块

    输出：z = z_max * tanh(z_raw)，范围 [0, z_max]。
    每个像素、每个通道有独立的 z，实现空间变化、通道分离的传播距离预测，
    从而支持与深度/雾浓度相关的去雾。
    """

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 16,
        num_layers: int = 2,
        z_max: float = 0.3,
        use_gn: bool = True,
    ):
        """
        Args:
            in_ch: 输入通道数，输出 z 形状 (B, in_ch, H, W)
            hidden: 隐藏层通道数
            num_layers: ResBlock 层数
            z_max: z 的上界，控制最大传播距离
            use_gn: 是否使用 GroupNorm
        """
        super().__init__()
        self.z_max = z_max

        self.in_proj = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(hidden, use_gn=use_gn) for _ in range(max(1, num_layers))])
        self.out_proj = nn.Conv2d(hidden, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, in_ch, H, W)

        Returns:
            z: 传播距离 (B, 3, H, W)，范围 [0, z_max]
        """
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        z_raw = self.out_proj(feat)
        z = self.z_max * torch.tanh(z_raw)  # 有界到 [-z_max, z_max]，实际使用多为正
        return z
