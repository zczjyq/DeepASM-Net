import math
import torch
import torch.nn as nn

from .tiny_vit import TinyViTFusion


class _ResBlock(nn.Module):
    """Conv residual block kept for non-ViT ablation."""

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
    Predict phase map phi in [-pi, pi].

    Output formulation:
      phi = pi * tanh(phi_shared + residual_scale * delta_phi)
    where phi_shared is shared across RGB (1 channel) and delta_phi is channel-wise (3 channels).
    """

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 16,
        num_layers: int = 2,
        use_gn: bool = True,
        use_channel_residual: bool = True,
        residual_scale: float = 0.1,
        use_vit: bool = False,
        vit_patch_size: int = 3,
        vit_embed_dim: int = 24,
        vit_depth: int = 1,
        vit_num_heads: int = 2,
        vit_window_size: int = 8,
        vit_mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.use_channel_residual = use_channel_residual
        self.residual_scale = residual_scale
        self.use_vit = use_vit

        if self.use_vit:
            self.vit_feat = TinyViTFusion(
                in_ch=in_ch,
                out_ch=hidden,
                patch_size=vit_patch_size,
                embed_dim=vit_embed_dim,
                depth=vit_depth,
                num_heads=vit_num_heads,
                window_size=vit_window_size,
                mlp_ratio=vit_mlp_ratio,
            )
            self.in_proj = None
            self.blocks = None
        else:
            self.in_proj = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
            self.blocks = nn.Sequential(*[_ResBlock(hidden, use_gn=use_gn) for _ in range(max(1, num_layers))])
            self.vit_feat = None

        self.out_shared = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        self.out_resid = nn.Conv2d(hidden, in_ch, kernel_size=3, padding=1)

        self.last_phi_shared = None
        self.last_delta_phi = None

    def _extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_vit:
            return self.vit_feat(x)
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._extract_feat(x)
        phi_shared = self.out_shared(feat)
        if self.use_channel_residual:
            delta_phi = self.out_resid(feat)
        else:
            delta_phi = torch.zeros_like(x)

        phi_raw = phi_shared + self.residual_scale * delta_phi
        phi = math.pi * torch.tanh(phi_raw)

        self.last_phi_shared = phi_shared
        self.last_delta_phi = delta_phi
        return phi
