"""
Core ABlock for wavefield dehazing.

Pipeline:
1) Predict phase phi and propagation distance z from input image.
2) Build complex field U0 = A * exp(i * phi).
3) Propagate U0 with differentiable ASM to get J.  [ASM 核心]
4) Fuse [x, J_luma, J_contrast] in mix head to predict K, B (DehazeFormer style).
5) Output K*x - B + x (atmospheric scattering form).

Mix head optionally adds TinyViT branch with shifted window.
"""
import torch
import torch.nn as nn

from .asm import FreqEnhance, asm_propagate
from .phase_module import PhaseModule
from .tiny_vit import TinyViTFusion
from .z_module import ZModule


class _ChannelAttention(nn.Module):
    """SE-like channel attention for K,B refinement."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# Mix head 输出通道: K(1) + B(3) = 4
MIX_OUT_CH = 4


class ABlock(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        use_norm: bool = True,
        phase_hidden: int = 16,
        phase_layers: int = 1,
        phase_use_channel_residual: bool = True,
        phase_residual_scale: float = 0.1,
        phase_use_vit: bool = False,
        phase_vit_patch_size: int = 3,
        phase_vit_embed_dim: int = 24,
        phase_vit_depth: int = 1,
        phase_vit_num_heads: int = 2,
        phase_vit_window_size: int = 8,
        phase_vit_mlp_ratio: float = 2.0,
        z_hidden: int = 16,
        z_layers: int = 1,
        z_max: float = 0.3,
        z_use_vit: bool = False,
        z_vit_patch_size: int = 3,
        z_vit_embed_dim: int = 24,
        z_vit_depth: int = 1,
        z_vit_num_heads: int = 2,
        z_vit_window_size: int = 8,
        z_vit_mlp_ratio: float = 2.0,
        mix_hidden: int = 32,
        use_mix_head: bool = True,
        use_tiny_vit: bool = False,
        vit_patch_size: int = 3,
        vit_embed_dim: int = 24,
        vit_depth: int = 1,
        vit_num_heads: int = 2,
        vit_window_size: int = 8,
        vit_mlp_ratio: float = 2.0,
        wavelengths=(0.65, 0.53, 0.47),
        amp_mode: str = "identity",
        output_mode: str = "abs",
    ):
        super().__init__()
        self.amp_mode = amp_mode
        self.output_mode = output_mode
        self.use_mix_head = use_mix_head
        self.use_tiny_vit = use_mix_head and use_tiny_vit
        self.register_buffer("wavelengths", torch.tensor(wavelengths))

        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_ch) if use_norm else nn.Identity()
        self.phase = PhaseModule(
            in_ch=in_ch,
            hidden=phase_hidden,
            num_layers=phase_layers,
            use_gn=use_norm,
            use_channel_residual=phase_use_channel_residual,
            residual_scale=phase_residual_scale,
            use_vit=phase_use_vit,
            vit_patch_size=phase_vit_patch_size,
            vit_embed_dim=phase_vit_embed_dim,
            vit_depth=phase_vit_depth,
            vit_num_heads=phase_vit_num_heads,
            vit_window_size=phase_vit_window_size,
            vit_mlp_ratio=phase_vit_mlp_ratio,
        )
        self.z_module = ZModule(
            in_ch=in_ch,
            hidden=z_hidden,
            num_layers=z_layers,
            z_max=z_max,
            use_gn=use_norm,
            use_vit=z_use_vit,
            vit_patch_size=z_vit_patch_size,
            vit_embed_dim=z_vit_embed_dim,
            vit_depth=z_vit_depth,
            vit_num_heads=z_vit_num_heads,
            vit_window_size=z_vit_window_size,
            vit_mlp_ratio=z_vit_mlp_ratio,
        )
        self.freq_enhance = FreqEnhance(channels=in_ch, hidden=8)

        if self.use_mix_head:
            self.mix = nn.Sequential(
                nn.Conv2d(in_ch + 2, mix_hidden, kernel_size=3, padding=1),
                nn.GroupNorm(1, mix_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(mix_hidden, mix_hidden, kernel_size=3, padding=1),
                nn.GroupNorm(1, mix_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(mix_hidden, MIX_OUT_CH, kernel_size=1),
            )
            # Init last conv: K≈1, B≈0 for identity at start (K*x - B + x = x)
            nn.init.zeros_(self.mix[-1].weight)
            with torch.no_grad():
                self.mix[-1].bias.zero_()
                self.mix[-1].bias[0] = 1.0

            if self.use_tiny_vit:
                self.tiny_vit = TinyViTFusion(
                    in_ch=in_ch + 2,
                    out_ch=MIX_OUT_CH,
                    patch_size=vit_patch_size,
                    embed_dim=vit_embed_dim,
                    depth=vit_depth,
                    num_heads=vit_num_heads,
                    window_size=vit_window_size,
                    mlp_ratio=vit_mlp_ratio,
                    use_shifted_window=True,
                )
                # Init TinyViT output to 0 (conv branch provides identity)
                nn.init.zeros_(self.tiny_vit.out_proj[-1].weight)
                nn.init.zeros_(self.tiny_vit.out_proj[-1].bias)
                self.vit_scale = nn.Parameter(torch.tensor(0.1))
            else:
                self.tiny_vit = None
                self.vit_scale = None
            self.ca = _ChannelAttention(MIX_OUT_CH)
            self.register_buffer("_luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        else:
            self.mix = None
            self.tiny_vit = None
            self.vit_scale = None
            self.ca = None
            self._luma_w = None

        self.last_z = None
        self.last_phi_shared = None

    def _amplitude(self, x: torch.Tensor) -> torch.Tensor:
        if self.amp_mode == "identity":
            return x
        if self.amp_mode == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0))
        raise ValueError("amp_mode must be 'identity' or 'sqrt'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        phi = self.phase(x_norm)
        z = self.z_module(x_norm)
        self.last_z = z
        self.last_phi_shared = self.phase.last_phi_shared

        # Force float32 in ASM complex path to avoid AMP complex-half instability.
        amp = self._amplitude(x).float()
        phi = phi.float()
        z = z.float()
        u0 = amp * torch.exp(1j * phi)
        _, j = asm_propagate(
            u0,
            z,
            self.wavelengths,
            output_mode=self.output_mode,
            clamp_sqrt=True,
            enhance_module=self.freq_enhance,
        )
        j = j.to(dtype=x.dtype)

        if not self.use_mix_head:
            return torch.clamp(j, 0.0, 1.0)

        x_luma = (x * self._luma_w).sum(dim=1, keepdim=True)
        j_luma = (j * self._luma_w).sum(dim=1, keepdim=True)
        j_contrast = j_luma - x_luma
        mix_in = torch.cat([x, j_luma, j_contrast], dim=1)

        kb = self.mix(mix_in)
        if self.tiny_vit is not None:
            kb = kb + self.vit_scale * self.tiny_vit(mix_in)
        kb = self.ca(kb)

        K = kb[:, 0:1]
        B = kb[:, 1:4]
        out = K * x - B + x
        return out
