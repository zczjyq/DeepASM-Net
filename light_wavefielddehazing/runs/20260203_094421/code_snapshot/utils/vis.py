import os
from typing import List, Tuple

from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image


def save_grid(images: torch.Tensor, path: str, nrow: int = 3):
    grid = make_grid(images, nrow=nrow, padding=2)
    save_image(grid, path)


def save_comparison_grid(hazy: torch.Tensor, output: torch.Tensor, clean: torch.Tensor, path: str):
    # Stack as [hazy, output, clean] per sample
    b = hazy.size(0)
    rows = []
    for i in range(b):
        rows.append(torch.stack([hazy[i], output[i], clean[i]], dim=0))
    grid = torch.cat(rows, dim=0)
    save_grid(grid, path, nrow=3)


def save_infer_grid(pairs: List[Tuple[str, str]], path: str, max_items: int = 8):
    images = []
    for inp, out in pairs[:max_items]:
        inp_img = Image.open(inp).convert("RGB")
        out_img = Image.open(out).convert("RGB")
        images.extend([TF.to_tensor(inp_img), TF.to_tensor(out_img)])
    if not images:
        return
    grid = make_grid(torch.stack(images, dim=0), nrow=2, padding=2)
    save_image(grid, path)


def save_html_summary(pairs: List[Tuple[str, str]], path: str):
    lines = [
        "<html>",
        "<head><meta charset='utf-8'><title>Dehazing Outputs</title></head>",
        "<body>",
        "<h2>Dehazing Outputs</h2>",
        "<table border='1' cellspacing='0' cellpadding='5'>",
        "<tr><th>Input</th><th>Output</th></tr>",
    ]
    for inp, out in pairs:
        lines.append("<tr>")
        lines.append(f"<td><img src='{os.path.basename(inp)}' width='256'></td>")
        lines.append(f"<td><img src='{os.path.basename(out)}' width='256'></td>")
        lines.append("</tr>")
    lines += ["</table>", "</body>", "</html>"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
