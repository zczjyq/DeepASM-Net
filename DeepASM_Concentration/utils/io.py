import os
import shutil
from datetime import datetime
from typing import Iterable

from PIL import Image
import torch
from torchvision.transforms import functional as TF


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)


def write_image(tensor: torch.Tensor, path: str):
    img = TF.to_pil_image(torch.clamp(tensor, 0.0, 1.0).cpu())
    img.save(path)


def copy_code_snapshot(src_root: str, dst_root: str):
    ignore = shutil.ignore_patterns(
        "images",
        "runs",
        "outputs",
        "__pycache__",
        ".git",
        ".venv",
    )

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root, ignore=ignore)
