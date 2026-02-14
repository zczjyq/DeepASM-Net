import torch
import torch.nn as nn
import torch.nn.functional as F


class _TransformerBlock(nn.Module):
    """Tiny transformer block for local window tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = max(4, int(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


def _partition_windows(x: torch.Tensor, ws: int):
    """
    x: (B, C, H, W)
    returns:
      windows: (B*nw, ws*ws, C)
      shape_info: (H_pad, W_pad, pad_h, pad_w)
    """
    b, c, h, w = x.shape
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    h_pad, w_pad = x.shape[-2:]
    x = x.view(b, c, h_pad // ws, ws, w_pad // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
    return x, (h_pad, w_pad, pad_h, pad_w)


def _reverse_windows(windows: torch.Tensor, ws: int, b: int, c: int, shape_info):
    h_pad, w_pad, pad_h, pad_w = shape_info
    x = windows.view(b, h_pad // ws, w_pad // ws, ws, ws, c)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h_pad, w_pad)
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


class TinyViTFusion(nn.Module):
    """
    Lightweight ViT branch:
    - 3x3 patch embedding (configurable)
    - local window self-attention with shifted window (DehazeFormer style)
    - bilinear upsample back to input size
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        patch_size: int = 3,
        embed_dim: int = 24,
        depth: int = 1,
        num_heads: int = 2,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        use_shifted_window: bool = True,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if embed_dim < 4:
            raise ValueError("embed_dim must be >= 4")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.patch_size = int(patch_size)
        self.window_size = max(1, int(window_size))
        self.shift_size = self.window_size // 2 if use_shifted_window else 0

        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(max(1, depth))]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        feat = self.patch_embed(x)  # (B, C, H/ps, W/ps)
        feat = feat + self.pos_conv(feat)
        _, c, hp, wp = feat.shape
        ws = min(self.window_size, hp, wp)
        ws = max(1, ws)
        shift = self.shift_size if (ws > 1 and self.shift_size > 0) else 0

        for i, blk in enumerate(self.blocks):
            # Shifted window: odd layers shift, even layers no shift (DehazeFormer style)
            if shift > 0 and i % 2 == 1:
                feat_shifted = torch.roll(feat, shifts=(-shift, -shift), dims=(2, 3))
            else:
                feat_shifted = feat

            tokens, shape_info = _partition_windows(feat_shifted, ws)
            tokens = blk(tokens)
            feat_blk = _reverse_windows(tokens, ws, b, c, shape_info)

            if shift > 0 and i % 2 == 1:
                feat_blk = torch.roll(feat_blk, shifts=(shift, shift), dims=(2, 3))

            feat = feat_blk

        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        out = self.out_proj(feat)
        return out
