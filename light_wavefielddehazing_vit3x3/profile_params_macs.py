"""
按模块统计参数量 (Params) 和 MACs，找出占比最大的模块

用法:
  python profile_params_macs.py
  python profile_params_macs.py --config configs/default.yaml --size 256
"""
import argparse
from collections import defaultdict

import torch
import yaml

from model.model import build_model_from_config


def get_params_by_module(model: torch.nn.Module) -> dict:
    """按模块名聚合参数量。"""
    module_params = defaultdict(int)
    for name, param in model.named_parameters():
        # 取最顶层有意义的模块名，如 blocks.0, blocks.0.phase, blocks.0.mix
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            module_params[prefix] += param.numel()
    return dict(module_params)


def get_params_by_top_module(model: torch.nn.Module) -> dict:
    """按顶层模块聚合（如 blocks.0, blocks.1, skip_fusions 等）。"""
    by_top = defaultdict(int)
    for name, param in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "blocks" and len(parts) >= 2:
            top = f"blocks.{parts[1]}"  # blocks.0, blocks.1, blocks.2
        else:
            top = parts[0]  # skip_fusions, color_corrector 等
        by_top[top] += param.numel()
    return dict(by_top)


def get_params_by_submodule(model: torch.nn.Module) -> dict:
    """按 ABlock 内子模块聚合（phase, z_module, mix, tiny_vit 等）。"""
    by_sub = defaultdict(int)
    for name, param in model.named_parameters():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks":
            # blocks.0.phase.xxx -> blocks.*.phase
            sub = f"blocks.*.{parts[2]}"
        elif len(parts) >= 2:
            sub = ".".join(parts[:2])
        else:
            sub = parts[0]
        by_sub[sub] += param.numel()
    return dict(by_sub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--size", type=int, default=256, help="输入尺寸 H=W")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(1, 3, args.size, args.size)

    # ========== 1. 参数量按子模块汇总 ==========
    by_sub = get_params_by_submodule(model)
    # 合并三个 block 的相同子模块
    merged = defaultdict(int)
    for k, v in by_sub.items():
        merged[k] += v

    print("=" * 70)
    print("参数量 (Params) 按子模块汇总")
    print("=" * 70)
    print(f"{'模块':<35} {'Params':>12} {'占比':>8}")
    print("-" * 70)
    sorted_subs = sorted(merged.items(), key=lambda x: -x[1])
    for name, cnt in sorted_subs:
        pct = 100 * cnt / total_params
        print(f"{name:<35} {cnt:>12,} {pct:>7.1f}%")
    print("-" * 70)
    print(f"{'总计':<35} {total_params:>12,} {100.0:>7.1f}%")
    print()

    # ========== 2. MACs 按模块汇总 ==========
    macs_by_module = None
    total_macs = None

    try:
        from fvcore.nn import FlopCountAnalysis

        flop_counter = FlopCountAnalysis(model, dummy)
        total_flops = flop_counter.total()
        total_macs = total_flops // 2
        macs_by_module = flop_counter.by_module()
    except Exception as e1:
        try:
            from thop import profile

            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            total_macs = flops // 2
            # thop 不提供 per-module，仅打印总量
            macs_by_module = None
        except Exception as e2:
            print("MACs 计算失败（模型含 FFT/复数，thop/fvcore 可能不支持）")
            print(f"  fvcore: {e1}")
            print(f"  thop:   {e2}")
            print("  建议: pip install fvcore 或 pip install thop")
            return

    print("=" * 70)
    print("MACs 按模块汇总")
    print("=" * 70)
    if total_macs is not None:
        print(f"总 MACs: {total_macs:,} ({total_macs/1e9:.2f} GM)")
        print()

    agg = {}
    if macs_by_module is not None and len(macs_by_module) > 0:
        macs_counter = macs_by_module if isinstance(macs_by_module, dict) else dict(macs_by_module)
        agg = defaultdict(int)
        for mod_name, flops in macs_counter.items():
            if flops <= 0:
                continue
            parts = mod_name.split(".")
            if len(parts) >= 3 and parts[0] == "blocks":
                key = f"blocks.*.{parts[2]}"
            elif len(parts) >= 2:
                key = ".".join(parts[:2])
            else:
                key = mod_name or "(root)"
            agg[key] += flops

        macs_sorted = sorted(agg.items(), key=lambda x: -x[1])
        print(f"{'模块':<35} {'MACs':>14} {'占比':>8}")
        print("-" * 70)
        for name, macs in macs_sorted:
            pct = 100 * macs / total_macs if total_macs else 0
            print(f"{name:<35} {macs:>14,} {pct:>7.1f}%")
        print("-" * 70)
    else:
        print("(thop 不支持 per-module MACs，仅显示总量)")

    # ========== 3. 结论 ==========
    print()
    print("=" * 70)
    print("结论：参数与 MACs 占比最大的模块")
    print("=" * 70)
    if sorted_subs:
        top_param = sorted_subs[0]
        print(f"参数量最大: {top_param[0]} ({top_param[1]:,} params, {100*top_param[1]/total_params:.1f}%)")
    if agg and total_macs:
        top_mac = max(agg.items(), key=lambda x: x[1])
        print(f"MACs 最大:  {top_mac[0]} ({top_mac[1]:,} MACs, {100*top_mac[1]/total_macs:.1f}%)")


if __name__ == "__main__":
    main()
