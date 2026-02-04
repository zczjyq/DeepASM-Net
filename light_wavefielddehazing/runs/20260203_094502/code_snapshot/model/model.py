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

    def __init__(self, in_ch: int = 3, hidden: int = 32):
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
    主去雾网络：堆叠多个 ABlock 的波动传播框架

    支持两种布局：
    - 串行：Input -> ABlock_1 -> ... -> ABlock_N -> Output
    - U-Net 式（use_skip_fusion=True）：L1->L2->L3，再 L4 与 L2 融合、L5 与 L1 融合
    """

    def __init__(self, num_layers: int = 4, share_weights: bool = False,
                 color_corrector_hidden: int = 0, use_skip_fusion: bool = True, **ablock_kwargs):
        """
        Args:
            num_layers: ABlock 堆叠层数
            share_weights: 若为 True，所有层共享同一 ABlock 权重
            color_corrector_hidden: >0 时启用颜色校正
            use_skip_fusion: True 时采用 U-Net 式结构，带 SK Fusion 跳跃连接
            **ablock_kwargs: 传递给 ABlock 的其余参数
        """
        super().__init__()
        self.num_layers = num_layers
        self.share_weights = share_weights
        self.use_skip_fusion = use_skip_fusion

        n = 5 if use_skip_fusion else num_layers
        if share_weights and not use_skip_fusion:
            self.block = ABlock(**ablock_kwargs)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([ABlock(**ablock_kwargs) for _ in range(n)])
            self.block = None

        if use_skip_fusion:
            self.sk_fusion_1 = SKFusion(channels=3)
            self.sk_fusion_2 = SKFusion(channels=3)

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

        if self.use_skip_fusion:
            # U-Net 式：L1 -> L2 -> L3，再 L4(+skip L2)、L5(+skip L1)
            x1 = _run(self.blocks[0], x)
            x2 = _run(self.blocks[1], x1)
            x3 = _run(self.blocks[2], x2)
            x4_in = self.sk_fusion_1(x3, x2)
            x4 = _run(self.blocks[3], x4_in)
            x5_in = self.sk_fusion_2(x4, x1)
            x = _run(self.blocks[4], x5_in)
        elif self.share_weights:
            for _ in range(self.num_layers):
                x = _run(self.block, x)
        else:
            for blk in self.blocks:
                x = _run(blk, x)

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
