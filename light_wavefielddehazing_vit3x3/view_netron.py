"""
Minimal Netron viewer:
1) Build model from config
2) Save as .pt
3) Open with Netron
"""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import build_model_from_config


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Export .pt and open Netron")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--out", default="model_netron.pt", help="Output .pt path")
    parser.add_argument("--port", type=int, default=8080, help="Netron port")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model_from_config(cfg).eval()

    out_path = os.path.abspath(args.out)
    if not out_path.lower().endswith(".pt"):
        out_path += ".pt"

    torch.save(model, out_path)
    print("Saved model:", out_path)

    try:
        import netron

        try:
            netron.start(out_path, port=args.port)
            print(f"Netron opened on port {args.port}")
        except TypeError:
            # Compatibility with older netron versions that do not accept `port`.
            netron.start(out_path)
            print("Netron opened (default port/version behavior).")
    except ImportError:
        print("netron is not installed. Run: pip install netron")
    except Exception as e:
        print("Failed to start Netron:", e)


if __name__ == "__main__":
    main()
