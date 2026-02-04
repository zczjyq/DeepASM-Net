import argparse
import os
import shutil

from PIL import Image
import torch
from torchvision.transforms import functional as TF
import yaml

from model.model import build_model_from_config
from utils.io import ensure_dir, timestamp
from utils.vis import save_html_summary, save_infer_grid


def list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    run_id = timestamp()
    out_dir = os.path.join(args.output_root, run_id)
    ensure_dir(out_dir)

    pairs = []
    with torch.no_grad():
        for img_path in list_images(args.input):
            img = Image.open(img_path).convert("RGB")
            inp = TF.to_tensor(img).unsqueeze(0).to(device)
            pred = model(inp).squeeze(0).cpu()

            filename = os.path.basename(img_path)
            out_path = os.path.join(out_dir, filename)
            TF.to_pil_image(pred).save(out_path)

            input_copy = os.path.join(out_dir, "input_" + filename)
            shutil.copy(img_path, input_copy)
            pairs.append((input_copy, out_path))

    html_path = os.path.join(out_dir, "summary.html")
    save_html_summary(pairs, html_path)

    grid_path = os.path.join(out_dir, "grid.png")
    save_infer_grid(pairs, grid_path, max_items=8)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
