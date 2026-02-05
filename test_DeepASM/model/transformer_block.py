"""
轻量级空间 Transformer 模块，用于在去雾模型中引入全局/长程依赖。

将 (B, C, H, W) 下采样为序列 (B, N, D)，做多头自注意力 + FFN，再恢复空间形状并以残差形式加回。
通过 scale 控制下采样倍数，在显存与效果之间折中（如 scale=8 时序列长度为 (H/8)*(W/8)）。
"""
import torch
import torch.nn as nn


class SpatialTransformerBlock(nn.Module):
    """
    空间 Transformer 块：对 2D 特征图做自注意力 + FFN，残差连接，输入输出形状一致 (B, C, H, W)。
    内部先下采样以控制序列长度，再做 MHA + FFN，最后上采样回原分辨率。
    """

    def __init__(
        self,
        in_ch: int = 3,
        embed_dim: int = 48,
        num_heads: int = 4,
        scale: int = 8,
        dropout: float = 0.0,
        ffn_ratio: float = 4.0,
    ):
        """
        Args:
            in_ch: 输入通道数（与图像通道一致，如 3）
            embed_dim: 注意力隐藏维度，需能被 num_heads 整除
            num_heads: 多头注意力头数
            scale: 空间下采样倍数，序列长度 = (H/scale)*(W/scale)，越大越省显存
            dropout: 注意力与 FFN 的 dropout
            ffn_ratio: FFN 隐藏层维度 = embed_dim * ffn_ratio
        """
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.scale = max(1, int(scale))

        self.proj_in = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.proj_out = nn.Conv2d(embed_dim, in_ch, kernel_size=1)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        ffn_hidden = max(embed_dim, int(embed_dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W)，与输入同形状，残差形式：out = x + block(x)
        """
        B, C, H, W = x.shape
        residual = x

        x = self.proj_in(x)   # (B, embed_dim, H, W)
        # 下采样以缩短序列
        if self.scale > 1:
            h, w = max(1, H // self.scale), max(1, W // self.scale)
            x = nn.functional.adaptive_avg_pool2d(x, (h, w))  # (B, embed_dim, h, w)
        else:
            h, w = H, W

        # (B, embed_dim, h, w) -> (B, h*w, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Pre-LN + MHA + residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)

        # Pre-LN + FFN + residual
        x = x + self.ffn(self.norm2(x))

        # (B, h*w, embed_dim) -> (B, embed_dim, h, w)
        x = x.transpose(1, 2).view(B, self.embed_dim, h, w)

        # 上采样回原分辨率
        if self.scale > 1:
            x = nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.proj_out(x)  # (B, in_ch, H, W)
        return residual + x
