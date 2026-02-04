"""
评估模型：PSNR、SSIM、LPIPS、FID

LPIPS: pip install lpips
FID: 需要 scipy，Inception 特征由 torchvision 提供
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from datasets import get_dataloaders
from model.model import build_model_from_config
from utils.metrics import psnr, ssim

# 尝试导入 LPIPS
_lpips_fn = None
try:
    import lpips
    _lpips_fn = lpips.LPIPS(net="alex").eval()
except Exception:
    pass


def _compute_lpips(pred: torch.Tensor, target: torch.Tensor, fn) -> float:
    """LPIPS 输入需为 [-1, 1]，返回 per-image 平均"""
    if fn is None:
        return float("nan")
    x = pred * 2 - 1
    y = target * 2 - 1
    with torch.no_grad():
        d = fn(x, y)
    return d.mean().item()


def _compute_fid(clean_list: list, pred_list: list, device: torch.device) -> float:
    """FID: 基于 Inception v3 特征的 Fréchet 距离"""
    try:
        import scipy
        from scipy import linalg
    except ImportError:
        return float("nan")
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        return float("nan")

    def get_inception(x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) * 2
        return x

    def load_inception():
        try:
            inv = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        except TypeError:
            inv = inception_v3(pretrained=True, transform_input=False)
        inv.fc = torch.nn.Identity()
        inv.aux_logits = False
        inv.eval()
        return inv.to(device)

    inv = load_inception()
    all_clean = torch.cat(clean_list, dim=0)
    all_pred = torch.cat(pred_list, dim=0)

    def features_from_tensor(t):
        t = get_inception(t)
        with torch.no_grad():
            f = inv(t)
        return f.cpu().numpy()

    feats_real = features_from_tensor(all_clean)
    feats_fake = features_from_tensor(all_pred)

    def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    mu_r = np.mean(feats_real, axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    mu_f = np.mean(feats_fake, axis=0)
    sigma_f = np.cov(feats_fake, rowvar=False)
    fid = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    return float(fid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--no_lpips", action="store_true", help="跳过 LPIPS")
    parser.add_argument("--no_fid", action="store_true", help="跳过 FID")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt.get("config")
        if cfg is None:
            raise ValueError("Config not found in checkpoint. Pass --config.")

    _, val_loader, _, _ = get_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    lpips_model = _lpips_fn.to(device) if _lpips_fn is not None else None
    if _lpips_fn is None:
        print("LPIPS 未安装，跳过。pip install lpips")

    val_psnr = 0.0
    val_ssim = 0.0
    val_lpips = 0.0
    clean_list = []
    pred_list = []
    n_batches = 0

    with torch.no_grad():
        for hazy, clean, _, _ in val_loader:
            hazy = hazy.to(device)
            clean = clean.to(device)
            pred = model(hazy)
            pred = torch.clamp(pred, 0.0, 1.0)

            val_psnr += psnr(pred, clean).item()
            val_ssim += ssim(pred, clean).item()
            if not args.no_lpips and lpips_model is not None:
                val_lpips += _compute_lpips(pred, clean, lpips_model)
            if not args.no_fid:
                clean_list.append(clean)
                pred_list.append(pred)
            n_batches += 1

    val_psnr /= max(1, n_batches)
    val_ssim /= max(1, n_batches)
    if not args.no_lpips and lpips_model is not None:
        val_lpips /= max(1, n_batches)

    print(f"PSNR:  {val_psnr:.4f}")
    print(f"SSIM:  {val_ssim:.4f}")
    if not args.no_lpips and lpips_model is not None:
        print(f"LPIPS: {val_lpips:.4f}")
    if not args.no_fid and clean_list:
        fid_val = _compute_fid(clean_list, pred_list, device)
        print(f"FID:   {fid_val:.2f}")


if __name__ == "__main__":
    main()
