import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import yaml

from datasets import get_dataloaders
from model.model import build_model_from_config
from utils.metrics import psnr, ssim, ssim_loss, edge_loss
from utils.losses import chroma_loss_ycbcr, luma_loss, saturation_loss
from utils.io import ensure_dir, timestamp, copy_code_snapshot
from utils.vis import save_comparison_grid


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_perceptual_loss(weight: float, layers):
    if weight <= 0.0:
        return None
    try:
        from torchvision.models import vgg16, VGG16_Weights
    except Exception:
        return None

    try:
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
    except Exception:
        return None

    max_layer = max(layers)
    vgg = vgg[: max_layer + 1].eval()
    for p in vgg.parameters():
        p.requires_grad = False

    class VGGPerceptual(nn.Module):
        def __init__(self, vgg, layers):
            super().__init__()
            self.vgg = vgg
            self.layers = set(layers)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x, y):
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
            loss = 0.0
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                y = layer(y)
                if i in self.layers:
                    loss = loss + torch.mean(torch.abs(x - y))
            return loss

    return VGGPerceptual(vgg, layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="使用数据的前 x%% (0-1)，如 0.1 表示 10%%")
    parser.add_argument("--root", type=str, default=None,
                        help="覆盖 config 中的 paths.root，指定数据根目录")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size，显存不足可减小")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 继续训练，如 runs/xxx/checkpoints/last.pth")
    parser.add_argument("--resume_new_run", action="store_true",
                        help="与 --resume 同用：加载模型但保存到新的 run 目录，不覆盖原 run")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.root is not None:
        cfg["paths"]["root"] = args.root
        print(f"==> 使用数据根目录: {args.root}")
    if args.batch_size is not None:
        cfg["loader"]["batch_size"] = args.batch_size
        cfg["loader"]["val_batch_size"] = max(1, args.batch_size)
        print(f"==> 使用 batch_size: {args.batch_size}")

    set_seed(cfg.get("seed", 123))

    # 固定输入尺寸时开启，可加速卷积
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    resume_path = args.resume
    resume_new_run = args.resume_new_run
    start_epoch = 1
    if resume_path:
        resume_path = os.path.normpath(os.path.abspath(resume_path))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume 文件不存在: {resume_path}")
        # 推断 run_dir：path 形如 .../runs/xxx/checkpoints/last.pth
        ckpt_dir = os.path.dirname(resume_path)
        if os.path.basename(ckpt_dir) == "checkpoints":
            inferred_run_dir = os.path.dirname(ckpt_dir)
        else:
            inferred_run_dir = None
        use_inferred_run = inferred_run_dir and os.path.isdir(inferred_run_dir) and not resume_new_run

    if resume_path and use_inferred_run:
        run_dir = inferred_run_dir
        print(f"==> 从 checkpoint 继续训练，输出目录: {run_dir}")
    else:
        run_id = timestamp()
        runs_dir = cfg["output"]["runs_dir"]
        run_dir = os.path.join(runs_dir, run_id)
        if resume_path:
            print(f"==> 从 checkpoint 加载，保存到新 run: {run_dir}")

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "samples"))

    # 非 resume 或新 run 时保存 config 和 code 快照
    if not resume_path or not use_inferred_run:
        config_snapshot_path = os.path.join(run_dir, "config.yaml")
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        copy_code_snapshot(os.getcwd(), os.path.join(run_dir, "code_snapshot"))
    else:
        # resume 到原 run：可用 checkpoint 内 config 覆盖（可选），此处优先用当前 config
        pass

    train_loader, val_loader, train_set, val_set = get_dataloaders(cfg)

    # 可选：只使用前 x% 的数据
    if args.data_ratio < 1.0:
        n_train = max(1, int(len(train_set) * args.data_ratio))
        n_val = max(1, int(len(val_set) * args.data_ratio))
        train_set = Subset(train_set, range(n_train))
        val_set = Subset(val_set, range(n_val))
        bs = cfg["loader"]["batch_size"]
        drop_last = n_train > bs  # 样本太少时 drop_last=False 避免 0 batch
        lc = cfg["loader"]
        nw = lc["num_workers"]
        sub_kw = dict(num_workers=nw, pin_memory=lc["pin_memory"])
        if nw > 0:
            sub_kw["prefetch_factor"] = lc.get("prefetch_factor", 2)
            sub_kw["persistent_workers"] = lc.get("persistent_workers", False)
        train_loader = DataLoader(
            train_set,
            batch_size=min(bs, n_train),
            shuffle=True,
            drop_last=drop_last,
            **sub_kw,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=min(lc["val_batch_size"], n_val),
            shuffle=False,
            drop_last=False,
            **sub_kw,
        )
        print(f"==> 使用数据前 {args.data_ratio*100:.0f}%: train={n_train}, val={n_val}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = cfg["loader"]["pin_memory"]
    model = build_model_from_config(cfg).to(device)
    if cfg.get("train", {}).get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    base_lr = cfg["train"]["lr"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 0)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg["train"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["cosine_tmax"])

    scaler = GradScaler(enabled=cfg["train"]["amp"])
    best_psnr = -1.0

    # 从 checkpoint 恢复
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"  [Warn] 无法加载 optimizer 状态: {e}，使用当前 optimizer")
        if "scaler_state" in ckpt and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"==> 已加载 {resume_path}, 从 epoch {start_epoch} 继续, best_psnr={best_psnr:.4f}")

    l1_weight = cfg["loss"]["l1"]
    ssim_weight = cfg["loss"]["ssim"]
    perc_weight = cfg["loss"]["perceptual"]
    color_weight = cfg["loss"].get("color", 0.0)
    luma_weight = cfg["loss"].get("luma", 0.0)
    edge_weight = cfg["loss"].get("edge", 0.0)
    sat_weight = cfg["loss"].get("saturation", 0.0)
    z_reg_weight = cfg["loss"].get("z_reg", 0.0)
    phase_tv_weight = cfg["loss"].get("phase_tv", 0.0)
    perc_layers = cfg["loss"]["perceptual_layers"]

    perceptual = build_perceptual_loss(perc_weight, perc_layers)
    if perceptual is not None:
        perceptual = perceptual.to(device)
    elif perc_weight > 0.0:
        print("Perceptual loss requested but VGG weights not available. Disabling perceptual loss.")
        perc_weight = 0.0

    # Fixed validation samples for qualitative grids
    fixed_n = cfg["train"]["fixed_val_samples"]
    fixed_indices = list(range(min(fixed_n, len(val_set))))
    fixed_loader = DataLoader(Subset(val_set, fixed_indices), batch_size=len(fixed_indices), shuffle=False)

    log_path = os.path.join(run_dir, "logs.csv")
    log_exists = os.path.isfile(log_path)
    if not log_exists:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0
        for step, (hazy, clean, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            hazy = hazy.to(device, non_blocking=pin)
            clean = clean.to(device, non_blocking=pin)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                pred = model(hazy)
                pred = torch.clamp(pred, 0.0, 1.0)
                loss = l1_weight * torch.mean(torch.abs(pred - clean))
                if ssim_weight > 0.0:
                    loss = loss + ssim_weight * ssim_loss(pred, clean)
                if perc_weight > 0.0 and perceptual is not None:
                    loss = loss + perc_weight * perceptual(pred, clean)
                if color_weight > 0.0:
                    loss = loss + color_weight * chroma_loss_ycbcr(pred, clean)
                if luma_weight > 0.0:
                    loss = loss + luma_weight * luma_loss(pred, clean)
                if edge_weight > 0.0:
                    loss = loss + edge_weight * edge_loss(pred, clean)
                if sat_weight > 0.0:
                    loss = loss + sat_weight * saturation_loss(pred, clean)
                if z_reg_weight > 0.0 and hasattr(model, "last_zs"):
                    z_reg = 0.0
                    for z in model.last_zs:
                        z_reg = z_reg + torch.mean(torch.abs(z[:, 0] - z[:, 1]) + torch.abs(z[:, 1] - z[:, 2]))
                    z_reg = z_reg / max(1, len(model.last_zs))
                    loss = loss + z_reg_weight * z_reg
                if phase_tv_weight > 0.0 and hasattr(model, "last_phi_shareds"):
                    tv = 0.0
                    for phi_s in model.last_phi_shareds:
                        if phi_s is None:
                            continue
                        dx = torch.abs(phi_s[:, :, :, 1:] - phi_s[:, :, :, :-1]).mean()
                        dy = torch.abs(phi_s[:, :, 1:, :] - phi_s[:, :, :-1, :]).mean()
                        tv = tv + (dx + dy)
                    tv = tv / max(1, len(model.last_phi_shareds))
                    loss = loss + phase_tv_weight * tv

            if not torch.isfinite(loss):
                print(f"  [Warn] Epoch {epoch} step {step}: loss is {loss.item()}, skipping backward")
                continue

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_steps += 1

        train_loss = running_loss / max(1, n_steps)

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = base_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

        # Validation
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for hazy, clean, _, _ in val_loader:
                hazy = hazy.to(device, non_blocking=pin)
                clean = clean.to(device, non_blocking=pin)
                pred = model(hazy)
                val_psnr += psnr(pred, clean).item()
                val_ssim += ssim(pred, clean).item()
        val_psnr /= max(1, len(val_loader))
        val_ssim /= max(1, len(val_loader))

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_psnr:.4f}", f"{val_ssim:.4f}"])

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_psnr={val_psnr:.4f}, val_ssim={val_ssim:.4f}")

        # Save sample grid
        for hazy, clean, _, _ in fixed_loader:
            hazy = hazy.to(device, non_blocking=pin)
            clean = clean.to(device, non_blocking=pin)
            with torch.no_grad():
                pred = model(hazy)
            grid_path = os.path.join(run_dir, "samples", f"epoch_{epoch:03d}.png")
            save_comparison_grid(hazy.cpu(), pred.cpu(), clean.cpu(), grid_path)

        # Checkpointing
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "config": cfg,
        }
        if hasattr(scaler, "state_dict"):
            ckpt["scaler_state"] = scaler.state_dict()
        last_path = os.path.join(run_dir, "checkpoints", "last.pth")
        torch.save(ckpt, last_path)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt["best_psnr"] = best_psnr
            best_path = os.path.join(run_dir, "checkpoints", "best.pth")
            torch.save(ckpt, best_path)


if __name__ == "__main__":
    main()
