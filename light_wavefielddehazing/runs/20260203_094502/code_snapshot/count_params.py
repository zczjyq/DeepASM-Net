"""统计模型参数量"""
import yaml
from model.model import build_model_from_config

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
model = build_model_from_config(cfg)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total:,} ({total/1e6:.2f}M)")
print(f"可训练参数: {trainable:,} ({trainable/1e6:.2f}M)")
