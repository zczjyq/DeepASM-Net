import argparse
import torch
import yaml

from datasets import get_dataloaders
from model.model import build_model_from_config
from utils.metrics import psnr, ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
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

    val_psnr = 0.0
    val_ssim = 0.0
    with torch.no_grad():
        for hazy, clean, _, _ in val_loader:
            hazy = hazy.to(device)
            clean = clean.to(device)
            pred = model(hazy)
            val_psnr += psnr(pred, clean).item()
            val_ssim += ssim(pred, clean).item()

    val_psnr /= max(1, len(val_loader))
    val_ssim /= max(1, len(val_loader))

    print(f"Val PSNR: {val_psnr:.4f}")
    print(f"Val SSIM: {val_ssim:.4f}")


if __name__ == "__main__":
    main()
