"""
ABlock：波动传播去雾核心块

单层 ABlock 实现一次迭代式去雾，结合物理传播（ASM）与数据驱动的 Mix 头。
"""
import torch
import torch.nn as nn

from .asm import asm_propagate, FreqEnhance
from .phase_module import PhaseModule
from .z_module import ZModule


class _ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) 通道注意力模块
    对 delta 进行通道加权：学习每个通道的重要性，突出有效通道、抑制噪声通道。
    流程：Squeeze(全局池化) -> Excitation(两层 FC) -> Scale(逐通道相乘)
    """
    def __init__(self, ch: int, reduction: int = 4):
        """
        Args:
            ch: 输入通道数
            reduction: 瓶颈压缩比，mid = ch // reduction，用于降低参数量
        """
        super().__init__()
        mid = max(ch // reduction, 4)  # 至少 4 维，避免过度压缩
        self.fc = nn.Sequential(
            nn.Linear(ch, mid),        # 压缩：ch -> mid
            nn.SiLU(inplace=True),     # 非线性
            nn.Linear(mid, ch),        # 恢复：mid -> ch
            nn.Sigmoid(),              # 输出 [0,1] 权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: delta 特征 (B, C, H, W)

        Returns:
            通道加权后的特征 (B, C, H, W)
        """
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))            # Squeeze: 全局平均池化 -> (B, C)
        w = self.fc(w).view(b, c, 1, 1)   # Excitation: 得到通道权重 (B, C, 1, 1)
        return x * w                       # Scale: 逐通道相乘，广播到 (B, C, H, W)


class ABlock(nn.Module):
    """
    波动传播去雾块（框架 A）

    单层流程：
    1) PhaseModule -> phi（相位）
    2) ZModule -> z（传播距离，空间变化）
    3) U0 = A * exp(i*phi)（复振幅场）
    4) ASM 传播得到 Uz
    5) J = |Uz|（传播后强度）
    6) Mix 头：[x, J_luma, J_contrast] -> Delta，经通道注意力
    7) 残差更新：I_{t+1} = I_t + alpha * Delta
    """

    def __init__(
        self,
        in_ch: int = 3,
        use_norm: bool = True,
        phase_hidden: int = 16,
        phase_layers: int = 1,
        phase_use_channel_residual: bool = True,
        phase_residual_scale: float = 0.1,
        z_hidden: int = 16,
        z_layers: int = 1,
        z_max: float = 0.3,
        mix_hidden: int = 32,
        wavelengths=(0.65, 0.53, 0.47),
        amp_mode: str = "identity",
        output_mode: str = "abs",
    ):
        """
        Args:
            in_ch: 输入通道数（RGB=3）
            use_norm: 是否使用 GroupNorm
            phase_hidden/layers: PhaseModule 隐藏维度和 ResBlock 层数
            phase_use_channel_residual: 相位是否使用通道残差
            phase_residual_scale: 通道残差缩放系数
            z_hidden/layers: ZModule 隐藏维度和 ResBlock 层数
            z_max: 传播距离 z 的上界
            mix_hidden: Mix 头隐藏层通道数
            wavelengths: RGB 三通道波长 (R, G, B)，单位与空间归一化一致
            amp_mode: 振幅模式，"identity" 或 "sqrt"
            output_mode: ASM 输出模式，"abs" 或 "abs2"
        """
        super().__init__()
        self.use_norm = use_norm
        self.amp_mode = amp_mode
        self.output_mode = output_mode
        self.register_buffer("wavelengths", torch.tensor(wavelengths))

        # 输入归一化
        if use_norm:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=in_ch)
        else:
            self.norm = nn.Identity()

        # 相位预测模块
        self.phase = PhaseModule(
            in_ch=in_ch,
            hidden=phase_hidden,
            num_layers=phase_layers,
            use_gn=use_norm,
            use_channel_residual=phase_use_channel_residual,
            residual_scale=phase_residual_scale,
        )
        # 传播距离预测模块
        self.z_module = ZModule(in_ch=in_ch, hidden=z_hidden, num_layers=z_layers, z_max=z_max, use_gn=use_norm)

        # 频域增强：FFT*H 后、IFFT 前（hidden 8 减参）
        self.freq_enhance = FreqEnhance(channels=in_ch, hidden=8)

        # Mix 头输入：[x_rgb(3), J_luma(1), J_contrast(1)] = 5 通道
        # 仅用 J 的亮度结构信息，避免色偏
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch + 2, mix_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, mix_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(mix_hidden, mix_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, mix_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(mix_hidden, in_ch, kernel_size=1),
        )
        self.ca = _ChannelAttention(in_ch)

        # 残差步长，可学习
        self.alpha = nn.Parameter(torch.tensor(0.3))

        # BT.601 亮度权重，用于提取 J_luma
        self.register_buffer("_luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """振幅提取：identity 直接取 x，sqrt 取 sqrt(x)"""
        if self.amp_mode == "identity":
            return x
        if self.amp_mode == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0))
        raise ValueError("amp_mode must be 'identity' or 'sqrt'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            去雾更新后的图像 (B, 3, H, W)
        """
        # 1. 归一化并预测相位、传播距离
        x_norm = self.norm(x)
        phi = self.phase(x_norm)           # (B, 3, H, W)，相位 [-pi, pi]
        z = self.z_module(x_norm)          # (B, 3, H, W)，传播距离 [0, z_max]
        self.last_z = z
        self.last_phi_shared = self.phase.last_phi_shared

        # 2. 构建复振幅场并 ASM 传播（强制 float32，避免 AMP 下的 ComplexHalf 警告）
        amp = self._amplitude(x).float()
        phi = phi.float()
        z = z.float()
        U0 = amp * torch.exp(1j * phi)     # 复振幅 U0 = A * exp(i*phi)

        _, J = asm_propagate(U0, z, self.wavelengths, output_mode=self.output_mode, clamp_sqrt=True,
                             enhance_module=self.freq_enhance)
        J = J.to(dtype=x.dtype)

        # 3. 从 J 提取颜色无关的结构线索
        #    J_luma: 传播后亮度（去雾强度图）
        #    J_contrast: x 到 J 的亮度差（去雾区域）
        x_luma = (x * self._luma_w).sum(dim=1, keepdim=True)
        J_luma = (J * self._luma_w).sum(dim=1, keepdim=True)
        J_contrast = J_luma - x_luma

        # 4. Mix 头预测残差 Delta，经通道注意力后做残差更新
        mix_in = torch.cat([x, J_luma, J_contrast], dim=1)  # (B, 5, H, W)
        delta = self.mix(mix_in)
        delta = self.ca(delta)
        out = x + self.alpha * delta
        return out
