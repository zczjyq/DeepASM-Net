"""
使用 PyTorch Profiler 对模型做性能分析

用法:
  python profile_model.py                    # 默认：前向 + 导出 trace
  python profile_model.py --no-trace          # 只打印统计，不导出 trace
  python profile_model.py --train-step        # 多跑几步「前向+反向」再统计
  python profile_model.py --cpu              # 在 CPU 上 profile（无 CUDA 统计）
"""
import argparse
import os
import sys

import torch
import yaml

# 保证可导入 model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import build_model_from_config


def get_config(config_path: str = "configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Profiler 测试脚本")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--batch", type=int, default=4, help="Profile 时的 batch size")
    parser.add_argument("--size", type=int, default=256, help="输入空间尺寸 (H=W)")
    parser.add_argument("--warmup", type=int, default=3, help="Profiler 前的预热步数")
    parser.add_argument("--steps", type=int, default=5, help="被 profile 的步数")
    parser.add_argument("--no-trace", action="store_true", help="不导出 chrome trace 文件")
    parser.add_argument("--train-step", action="store_true", help="Profile 前向+反向，模拟训练一步")
    parser.add_argument("--cpu", action="store_true", help="在 CPU 上运行（无 CUDA 事件）")
    parser.add_argument("--out-dir", default=".", help="trace 输出目录")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_cuda = device.type == "cuda"
    img_size = args.size

    print("=" * 60)
    print("PyTorch Profiler 测试 (Light Wavefield Dehazing)")
    print("=" * 60)
    print(f"  Device:      {device}")
    if use_cuda:
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Batch size:  {args.batch}")
    print(f"  Input size:  {args.batch} x 3 x {img_size} x {img_size}")
    print(f"  Warmup:      {args.warmup}  steps")
    print(f"  Profiled:    {args.steps}   steps")
    print(f"  Mode:        {'train (forward+backward)' if args.train_step else 'inference (forward only)'}")
    print("=" * 60)

    cfg = get_config(args.config)
    model = build_model_from_config(cfg).to(device)
    model.train(args.train_step)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)\n")

    dummy = torch.randn(args.batch, 3, img_size, img_size, device=device)

    # 预热
    print("Warmup ...")
    for _ in range(args.warmup):
        if args.train_step:
            out = model(dummy)
            loss = out.sum()
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    # Profiler 活动：CUDA 仅 GPU 时开启
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_dir = os.path.join(args.out_dir, "profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "trace.json")

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=use_cuda,
        with_stack=False,
        with_flops=True,
    ) as prof:
        for step in range(args.steps):
            if args.train_step:
                model.zero_grad(set_to_none=True)
                out = model(dummy)
                loss = out.sum()
                loss.backward()
            else:
                with torch.no_grad():
                    _ = model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            prof.step()

    # 控制台汇总
    print("\n" + "=" * 60)
    print("Profiler 汇总 (按 CPU 时间排序)")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    if use_cuda:
        print("\n" + "=" * 60)
        print("按 CUDA 时间排序")
        print("=" * 60)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    if not args.no_trace:
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace 已保存: {trace_path}")
        print("  在 Chrome 浏览器打开 chrome://tracing/ ，Load 该文件即可查看时间线。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
