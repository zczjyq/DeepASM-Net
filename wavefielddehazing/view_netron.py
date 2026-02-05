"""
用 Netron 可视化网络结构

用法:
  python view_netron.py                    # 默认配置，导出 ONNX 并用浏览器打开 Netron
  python view_netron.py --config configs/default.yaml
  python view_netron.py --format pt         # 导出 PyTorch .pt（若 ONNX 不支持某些算子）
  python view_netron.py --no-browser        # 只导出文件，不自动打开浏览器（可手动用 Netron 打开）
"""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import build_model_from_config


def get_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="导出模型并用 Netron 可视化")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--batch", type=int, default=1, help="导出时的 batch size")
    parser.add_argument("--size", type=int, default=256, help="输入空间尺寸 (H=W)")
    parser.add_argument("--format", choices=["onnx", "pt"], default="onnx",
                        help="导出格式: onnx（推荐）或 pt")
    parser.add_argument("--out", default="model_netron", help="输出文件名（不含扩展名）")
    parser.add_argument("--no-browser", action="store_true", help="只导出文件，不启动 Netron 浏览器")
    parser.add_argument("--port", type=int, default=8080, help="Netron 本地服务端口")
    args = parser.parse_args()

    cfg = get_config(args.config)
    model = build_model_from_config(cfg)
    model.eval()

    dummy = torch.randn(args.batch, 3, args.size, args.size)
    out_path = os.path.abspath(args.out + (".onnx" if args.format == "onnx" else ".pt"))

    if args.format == "onnx":
        try:
            torch.onnx.export(
                model,
                dummy,
                out_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch", 2: "H", 3: "W"}, "output": {0: "batch", 2: "H", 3: "W"}},
                opset_version=17,
            )
            print("已导出 ONNX:", out_path)
        except Exception as e:
            print("ONNX 导出失败（可能含复数/FFT 等算子）:", e)
            print("改用 PyTorch .pt 格式导出...")
            out_path = os.path.abspath(args.out + ".pt")
            torch.save(model, out_path)
            print("已导出 PyTorch:", out_path)
    else:
        torch.save(model, out_path)
        print("已导出 PyTorch:", out_path)

    if not args.no_browser:
        try:
            import netron
            netron.start(out_path, port=args.port)
            print("Netron 已在浏览器中打开，端口:", args.port)
        except ImportError:
            print("未安装 netron，请执行: pip install netron")
            print("然后手动用 Netron 打开文件:", out_path)
        except Exception as e:
            print("启动 Netron 失败:", e)
            print("请手动用 Netron 打开:", out_path)


if __name__ == "__main__":
    main()
