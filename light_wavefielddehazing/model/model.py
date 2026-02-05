"""
Wavefield Dehaze 网络主模块

本模块包含：
- ColorCorrector: 后处理颜色校正模块
- SKFusion: 跳跃连接融合模块（参考 DehazeFormer）
- AStack: 主网络，支持串行或 U-Net 式多尺度+跳跃结构
"""
import torch
import torch.nn as nn

from .a_block import ABlock


class SKFusion(nn.Module):
    """
    Selective Kernel 融合：将当前特征与跳跃特征融合。
    参考 DehazeFormer 的 SK Fusion，用于解码器与编码器跳跃连接结合。
    """

    def __init__(self, channels: int = 3, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, curr: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        curr: 当前层输出 (B, C, H, W)
        skip: 跳跃连接 (B, C, H, W)，需与 curr 同分辨率
        """
        if curr.shape != skip.shape:
            skip = nn.functional.interpolate(skip, size=curr.shape[2:], mode="bilinear", align_corners=False)
        concat = torch.cat([curr, skip], dim=1)
        w = self.fc(concat).view(-1, 2, 1, 1)
        return w[:, 0:1] * curr + w[:, 1:2] * skip


class ColorCorrector(nn.Module):
    """
    后处理颜色校正模块（可选）

    在 ASM 去雾管道之后应用，接收去雾结果和原始有雾输入，
    预测逐像素的仿射变换 scale 和 bias：
        output = scale * dehazed + bias

    初始化时 scale≈1、bias≈0（恒等变换），可安全加载到已训练的 AStack 上
    而不破坏原有效果。
    """

    def __init__(self, in_ch: int = 3, hidden: int = 16):
        """
        Args:
            in_ch: 输入通道数，RGB 为 3
            hidden: 隐藏层通道数
        """
        super().__init__()
        # 特征提取：concat(dehazed, original) -> 6 通道
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 2, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        # 输出头：预测 scale 和 bias
        self.to_scale = nn.Conv2d(hidden, in_ch, kernel_size=1)
        self.to_bias = nn.Conv2d(hidden, in_ch, kernel_size=1)

        # 恒等初始化：scale→1, bias→0
        nn.init.zeros_(self.to_scale.weight)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, dehazed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dehazed: 去雾后的图像 (B, 3, H, W)
            original: 原始有雾图像 (B, 3, H, W)

        Returns:
            颜色校正后的图像 (B, 3, H, W)
        """
        feat = self.net(torch.cat([dehazed, original], dim=1))
        scale = self.to_scale(feat)   # (B, 3, H, W)，初始接近 1
        bias = self.to_bias(feat)     # (B, 3, H, W)，初始接近 0
        return dehazed * scale + bias


class AStack(nn.Module):
    """
    主去雾网络：固定 3 层 ABlock + 层间残差

    结构：x -> [ABlock1 -> +res0*x] -> x1 -> [ABlock2 -> +res1*x1] -> x2 -> [ABlock3 -> +res2*x2] -> out
    三层之间用可学习权重的残差连接，稳定训练并保留多尺度信息。
    """

    NUM_LAYERS = 3

    def __init__(self, num_layers: int = 3, share_weights: bool = False,
                 color_corrector_hidden: int = 0, use_skip_fusion: bool = True, **ablock_kwargs):
        super().__init__()
        # 写死 3 层 ABlock
        self.blocks = nn.ModuleList([ABlock(**ablock_kwargs) for _ in range(self.NUM_LAYERS)])
        # 层间残差：每层输出 + scale * 该层输入，可学习权重初始小一点
        self.res_scale_0 = nn.Parameter(torch.tensor(0.1))  # x1 = block0(x) + res_scale_0 * x
        self.res_scale_1 = nn.Parameter(torch.tensor(0.1))  # x2 = block1(x1) + res_scale_1 * x1
        self.res_scale_2 = nn.Parameter(torch.tensor(0.1))  # x3 = block2(x2) + res_scale_2 * x2

        if color_corrector_hidden > 0:
            self.color_corrector = ColorCorrector(in_ch=3, hidden=color_corrector_hidden)
        else:
            self.color_corrector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        self.last_zs = []
        self.last_phi_shareds = []

        def _run(blk, inp):
            out = blk(inp)
            self.last_zs.append(blk.last_z)
            self.last_phi_shareds.append(blk.last_phi_shared)
            return out

        x1 = _run(self.blocks[0], x) + self.res_scale_0 * x
        x2 = _run(self.blocks[1], x1) + self.res_scale_1 * x1
        x = _run(self.blocks[2], x2) + self.res_scale_2 * x2

        x = torch.clamp(x, 0.0, 1.0)
        if self.color_corrector is not None:
            x = self.color_corrector(x, original)
            x = torch.clamp(x, 0.0, 1.0)
        return x


def build_model_from_config(cfg: dict) -> AStack:
    """
    从 YAML 配置字典构建 AStack 模型

    Args:
        cfg: 完整配置，需包含 cfg["model"] 及其子项

    Returns:
        配置好的 AStack 模型实例
    """
    mcfg = cfg["model"]
    phase_cfg = mcfg["phase_module"]
    z_cfg = mcfg["z_module"]

    cc_cfg = mcfg.get("color_corrector", {})
    cc_hidden = cc_cfg.get("hidden", 0) if cc_cfg.get("enabled", False) else 0

    model = AStack(
        num_layers=mcfg["num_layers"],
        share_weights=mcfg["share_weights"],
        color_corrector_hidden=cc_hidden,
        use_skip_fusion=mcfg.get("use_skip_fusion", True),
        use_norm=mcfg["use_norm"],
        phase_hidden=phase_cfg["hidden"],
        phase_layers=phase_cfg["num_layers"],
        z_hidden=z_cfg["hidden"],
        z_layers=z_cfg["num_layers"],
        phase_use_channel_residual=phase_cfg.get("use_channel_residual", True),
        phase_residual_scale=phase_cfg.get("phase_residual_scale", 0.1),
        z_max=mcfg["z_max"],
        mix_hidden=mcfg.get("mix_hidden", 64),
        wavelengths=mcfg["wavelengths"],
        amp_mode=mcfg["amp_mode"],
        output_mode=mcfg["output_mode"],
    )
    return model
