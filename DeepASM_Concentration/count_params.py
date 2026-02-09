import argparse
import os
import tempfile
from pathlib import Path

import torch
import yaml

from model.model import build_model_from_config


def resolve_config_path(config_arg: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_arg)
    if not cfg_path.is_absolute():
        cfg_path = script_dir / cfg_path
    return cfg_path.resolve()


def count_tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def pretty_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(script_dir / "configs" / "default.yaml"),
        help="Path to config yaml.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    param_bytes = count_tensor_bytes(model.parameters())
    buffer_bytes = count_tensor_bytes(model.buffers())
    model_mem_bytes = param_bytes + buffer_bytes

    print(f"配置文件 Config: {config_path}")
    print(f"总参数量 Total params:           {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"可训练参数 Trainable params:      {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    print(f"冻结参数 Frozen params:          {frozen_params:,} ({frozen_params / 1e6:.3f} M)")
    print(f"参数内存 Param memory:           {pretty_mib(param_bytes)}")
    print(f"缓冲区内存 Buffer memory:        {pretty_mib(buffer_bytes)}")
    print(f"模型内存合计 Model memory sum:   {pretty_mib(model_mem_bytes)} (params + buffers)")

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            tmp_file = f.name
        torch.save(model.state_dict(), tmp_file)
        ckpt_bytes = os.path.getsize(tmp_file)
        print(f"权重文件大小 state_dict size:    {pretty_mib(ckpt_bytes)} (on disk)")
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.remove(tmp_file)

    h = int(cfg["transforms"]["image_size"])
    w = int(cfg["transforms"]["image_size"])
    dummy = torch.randn(1, 3, h, w)

    thop_ok = False
    try:
        from thop import profile

        # THOP returns MACs.
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_est = macs * 2
        print(f"THOP 乘加 THOP MACs:             {macs:,.0f} ({macs / 1e9:.3f} GMac)")
        print(f"THOP 浮点(估算) THOP FLOPs(est): {flops_est:,.0f} ({flops_est / 1e9:.3f} GFLOPs)")
        thop_ok = True
    except Exception as e:
        print(f"THOP 失败 THOP failed:           {e}")

    try:
        from fvcore.nn import FlopCountAnalysis

        flops = float(FlopCountAnalysis(model, dummy).total())
        macs_est = flops / 2.0
        print(f"fvcore 浮点 fvcore FLOPs:        {flops:,.0f} ({flops / 1e9:.3f} GFLOPs)")
        print(f"fvcore 乘加(估算) fvcore MACs:   {macs_est:,.0f} ({macs_est / 1e9:.3f} GMac)")
    except Exception as e:
        if not thop_ok:
            print("无法统计 FLOPs/MACs: 请安装 `thop` 或 `fvcore`.")
            print("FLOPs/MACs unavailable: install `thop` or `fvcore`.")
        print(f"fvcore 失败 fvcore failed:       {e}")

    print("注意 Note: FFT/复数运算可能被分析工具部分计数。")


if __name__ == "__main__":
    main()
