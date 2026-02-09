image.png"""
统计模型参数量 (Params) 和 乘加运算量 (MACs/FLOPs)

参数量：sum(p.numel()) 准确统计所有权重元素个数
MACs：Multiply-Accumulate，1 MAC = 1 次乘法 + 1 次加法
FLOPs：约等于 2 * MACs（一次 MAC 算 2 次浮点运算）
"""
import yaml
import torch
from model.model import build_model_from_config

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = build_model_from_config(cfg)
model.eval()

# 参数量（准确）
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total:,} ({total/1e6:.2f}M)")
print(f"可训练参数: {trainable:,} ({trainable/1e6:.2f}M)")

# MACs / FLOPs（需要前向一次，输入尺寸影响结果）
h, w = cfg["transforms"]["image_size"], cfg["transforms"]["image_size"]
dummy = torch.randn(1, 3, h, w)

try:
    from thop import profile
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    macs = flops // 2  # 1 MAC ≈ 2 FLOPs 
    print(f"FLOPs: {flops:,} ({flops/1e9:.2f}G)")
    print(f"MACs:  {macs:,} ({macs/1e9:.2f}GM)")
except Exception as e:
    try:
        from fvcore.nn import FlopCountAnalysis
        flop_counter = FlopCountAnalysis(model, dummy)
        flops = flop_counter.total()
        macs = flops // 2
        print(f"FLOPs: {flops:,} ({flops/1e9:.2f}G)")
        print(f"MACs:  {macs:,} ({macs/1e9:.2f}GM)")
    except Exception:
        print("MACs/FLOPs 计算失败 (thop 可能不支持复数/FFT): pip install thop")
        print(f"  详情: {e}")
