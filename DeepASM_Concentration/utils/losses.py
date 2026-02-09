import torch


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB in [0,1] to YCbCr (BT.601).
    Output range is approximately Y in [0,1], Cb/Cr around 0.5 center.
    """
    x = torch.clamp(x, 0.0, 1.0)
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return torch.cat([y, cb, cr], dim=1)


def chroma_loss_ycbcr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ycbcr_p = rgb_to_ycbcr(pred)
    ycbcr_t = rgb_to_ycbcr(target)
    return torch.mean(torch.abs(ycbcr_p[:, 1:3] - ycbcr_t[:, 1:3]))


def luma_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ycbcr_p = rgb_to_ycbcr(pred)
    ycbcr_t = rgb_to_ycbcr(target)
    return torch.mean(torch.abs(ycbcr_p[:, 0:1] - ycbcr_t[:, 0:1]))


def to_y3(x: torch.Tensor) -> torch.Tensor:
    """
    Convert to luminance Y and repeat to 3 channels for VGG perceptual loss.
    """
    y = rgb_to_ycbcr(x)[:, 0:1]
    return y.repeat(1, 3, 1, 1)


def saturation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalise loss of saturation.  sat = (max_c - min_c) / (max_c + eps).
    Directly prevents the model from washing colours toward white/grey.
    """
    eps = 1e-6
    pred_max = pred.max(dim=1, keepdim=True)[0]
    pred_min = pred.min(dim=1, keepdim=True)[0]
    tgt_max = target.max(dim=1, keepdim=True)[0]
    tgt_min = target.min(dim=1, keepdim=True)[0]

    pred_sat = (pred_max - pred_min) / (pred_max + eps)
    tgt_sat = (tgt_max - tgt_min) / (tgt_max + eps)

    return torch.mean(torch.abs(pred_sat - tgt_sat))
