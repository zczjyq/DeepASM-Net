import torch
import torch.nn as nn

from .a_block import ABlock


class SKFusion(nn.Module):
    """Selective-kernel fusion between current and skip features."""

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
        if curr.shape != skip.shape:
            skip = nn.functional.interpolate(skip, size=curr.shape[2:], mode="bilinear", align_corners=False)
        concat = torch.cat([curr, skip], dim=1)
        w = self.fc(concat).view(-1, 2, 1, 1)
        return w[:, 0:1] * curr + w[:, 1:2] * skip


class ColorCorrector(nn.Module):
    """Optional post color correction head."""

    def __init__(self, in_ch: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 2, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.to_scale = nn.Conv2d(hidden, in_ch, kernel_size=1)
        self.to_bias = nn.Conv2d(hidden, in_ch, kernel_size=1)

        nn.init.zeros_(self.to_scale.weight)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, dehazed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        feat = self.net(torch.cat([dehazed, original], dim=1))
        scale = self.to_scale(feat)
        bias = self.to_bias(feat)
        return dehazed * scale + bias


class AStack(nn.Module):
    """Stacked ABlock backbone."""

    def __init__(
        self,
        num_layers: int = 3,
        share_weights: bool = False,
        color_corrector_hidden: int = 0,
        use_skip_fusion: bool = True,
        first_block_scale: float = 1.0,
        **ablock_kwargs,
    ):
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        self.use_skip_fusion = bool(use_skip_fusion and self.num_layers > 1)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block_kwargs = dict(ablock_kwargs)
            if i == 0 and first_block_scale < 1.0:
                block_kwargs = self._shrink_block_kwargs(block_kwargs, first_block_scale)
            self.blocks.append(ABlock(**block_kwargs))

        if self.use_skip_fusion:
            self.skip_fusions = nn.ModuleList([SKFusion(channels=3) for _ in range(self.num_layers - 1)])
        else:
            self.skip_fusions = nn.ModuleList()

        # Keep residual path for non-skip-fusion mode.
        self.res_scales = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_layers)])

        if color_corrector_hidden > 0:
            self.color_corrector = ColorCorrector(in_ch=3, hidden=color_corrector_hidden)
        else:
            self.color_corrector = None

        # currently unused, kept for config compatibility
        self.share_weights = bool(share_weights)

    @staticmethod
    def _scaled_channels(value: int, scale: float, minimum: int = 4) -> int:
        return max(minimum, int(round(value * scale)))

    @staticmethod
    def _scaled_embed_dim(value: int, heads: int, scale: float) -> int:
        raw = max(4, int(round(value * scale)))
        heads = max(1, int(heads))
        aligned = max(heads, (raw // heads) * heads)
        if aligned % heads != 0:
            aligned = ((aligned + heads - 1) // heads) * heads
        return max(4, aligned)

    @classmethod
    def _shrink_block_kwargs(cls, kwargs: dict, scale: float) -> dict:
        out = dict(kwargs)
        for key in ("phase_hidden", "z_hidden", "mix_hidden"):
            if key in out:
                out[key] = cls._scaled_channels(int(out[key]), scale)

        embed_specs = [
            ("phase_vit_embed_dim", "phase_vit_num_heads"),
            ("z_vit_embed_dim", "z_vit_num_heads"),
            ("vit_embed_dim", "vit_num_heads"),
        ]
        for dim_key, head_key in embed_specs:
            if dim_key in out:
                heads = int(out.get(head_key, 1))
                out[dim_key] = cls._scaled_embed_dim(int(out[dim_key]), heads, scale)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        self.last_zs = []
        self.last_phi_shareds = []

        def _run(blk: ABlock, inp: torch.Tensor) -> torch.Tensor:
            out = blk(inp)
            self.last_zs.append(blk.last_z)
            self.last_phi_shareds.append(blk.last_phi_shared)
            return out

        inp = x
        for i in range(self.num_layers):
            if i == 0:
                block_in = inp
            elif self.use_skip_fusion:
                # requested pipeline: stage 2/3 input = fusion(previous result, original image)
                block_in = self.skip_fusions[i - 1](inp, original)
            else:
                block_in = inp

            out = _run(self.blocks[i], block_in)

            if self.use_skip_fusion:
                inp = out
            else:
                inp = out + self.res_scales[i] * inp

        x = torch.clamp(inp, 0.0, 1.0)
        if self.color_corrector is not None:
            x = self.color_corrector(x, original)
            x = torch.clamp(x, 0.0, 1.0)
        return x


def build_model_from_config(cfg: dict) -> AStack:
    mcfg = cfg["model"]
    phase_cfg = mcfg["phase_module"]
    z_cfg = mcfg["z_module"]
    vit_cfg = mcfg.get("tiny_vit", {})

    cc_cfg = mcfg.get("color_corrector", {})
    cc_hidden = cc_cfg.get("hidden", 0) if cc_cfg.get("enabled", False) else 0

    model = AStack(
        num_layers=mcfg["num_layers"],
        share_weights=mcfg["share_weights"],
        color_corrector_hidden=cc_hidden,
        use_skip_fusion=mcfg.get("use_skip_fusion", True),
        first_block_scale=mcfg.get("first_block_scale", 1.0),
        use_norm=mcfg["use_norm"],
        phase_hidden=phase_cfg["hidden"],
        phase_layers=phase_cfg["num_layers"],
        phase_use_vit=phase_cfg.get("use_vit", False),
        phase_vit_patch_size=phase_cfg.get("vit_patch_size", 3),
        phase_vit_embed_dim=phase_cfg.get("vit_embed_dim", 24),
        phase_vit_depth=phase_cfg.get("vit_depth", 1),
        phase_vit_num_heads=phase_cfg.get("vit_num_heads", 2),
        phase_vit_window_size=phase_cfg.get("vit_window_size", 8),
        phase_vit_mlp_ratio=phase_cfg.get("vit_mlp_ratio", 2.0),
        z_hidden=z_cfg["hidden"],
        z_layers=z_cfg["num_layers"],
        z_use_vit=z_cfg.get("use_vit", False),
        z_vit_patch_size=z_cfg.get("vit_patch_size", 3),
        z_vit_embed_dim=z_cfg.get("vit_embed_dim", 24),
        z_vit_depth=z_cfg.get("vit_depth", 1),
        z_vit_num_heads=z_cfg.get("vit_num_heads", 2),
        z_vit_window_size=z_cfg.get("vit_window_size", 8),
        z_vit_mlp_ratio=z_cfg.get("vit_mlp_ratio", 2.0),
        phase_use_channel_residual=phase_cfg.get("use_channel_residual", True),
        phase_residual_scale=phase_cfg.get("phase_residual_scale", 0.1),
        z_max=mcfg["z_max"],
        mix_hidden=mcfg.get("mix_hidden", 64),
        use_mix_head=mcfg.get("use_mix_head", True),
        use_tiny_vit=vit_cfg.get("enabled", False),
        vit_patch_size=vit_cfg.get("patch_size", 3),
        vit_embed_dim=vit_cfg.get("embed_dim", 24),
        vit_depth=vit_cfg.get("depth", 1),
        vit_num_heads=vit_cfg.get("num_heads", 2),
        vit_window_size=vit_cfg.get("window_size", 8),
        vit_mlp_ratio=vit_cfg.get("mlp_ratio", 2.0),
        wavelengths=mcfg["wavelengths"],
        amp_mode=mcfg["amp_mode"],
        output_mode=mcfg["output_mode"],
    )
    return model
