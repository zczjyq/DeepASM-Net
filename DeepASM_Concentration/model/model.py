import torch
import torch.nn as nn

from .asm import asm_propagate
from .phase_module import PhaseModule
from .z_module import ZModule


class ASMPhaseZNet(nn.Module):
    """Minimal physics-guided model with only phase, z, and ASM."""

    def __init__(
        self,
        in_ch: int = 3,
        use_norm: bool = True,
        phase_hidden: int = 16,
        phase_layers: int = 2,
        phase_use_channel_residual: bool = True,
        phase_residual_scale: float = 0.1,
        z_hidden: int = 16,
        z_layers: int = 2,
        z_max: float = 0.3,
        wavelengths=(0.65, 0.53, 0.47),
        amp_mode: str = "identity",
        output_mode: str = "abs",
    ):
        super().__init__()
        self.amp_mode = amp_mode
        self.output_mode = output_mode
        self.register_buffer("wavelengths", torch.tensor(wavelengths))

        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_ch) if use_norm else nn.Identity()
        self.phase = PhaseModule(
            in_ch=in_ch,
            hidden=phase_hidden,
            num_layers=phase_layers,
            use_gn=use_norm,
            use_channel_residual=phase_use_channel_residual,
            residual_scale=phase_residual_scale,
        )
        self.z_module = ZModule(
            in_ch=in_ch,
            hidden=z_hidden,
            num_layers=z_layers,
            z_max=z_max,
            use_gn=use_norm,
        )

        self.last_zs = []
        self.last_phi_shareds = []

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

        self.last_zs = [z]
        self.last_phi_shareds = [self.phase.last_phi_shared]

        amp = self._amplitude(x).float()
        U0 = amp * torch.exp(1j * phi.float())

        _, J = asm_propagate(
            U0,
            z.float(),
            self.wavelengths,
            output_mode=self.output_mode,
            clamp_sqrt=True,
            enhance_module=None,
        )

        return torch.clamp(J.to(dtype=x.dtype), 0.0, 1.0)


def build_model_from_config(cfg: dict) -> ASMPhaseZNet:
    mcfg = cfg["model"]
    phase_cfg = mcfg["phase_module"]
    z_cfg = mcfg["z_module"]

    return ASMPhaseZNet(
        in_ch=3,
        use_norm=mcfg.get("use_norm", True),
        phase_hidden=phase_cfg.get("hidden", 16),
        phase_layers=phase_cfg.get("num_layers", 2),
        phase_use_channel_residual=phase_cfg.get("use_channel_residual", True),
        phase_residual_scale=phase_cfg.get("phase_residual_scale", 0.1),
        z_hidden=z_cfg.get("hidden", 16),
        z_layers=z_cfg.get("num_layers", 2),
        z_max=mcfg.get("z_max", 0.3),
        wavelengths=mcfg.get("wavelengths", [0.65, 0.53, 0.47]),
        amp_mode=mcfg.get("amp_mode", "identity"),
        output_mode=mcfg.get("output_mode", "abs"),
    )