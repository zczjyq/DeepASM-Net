# 网络组成说明 (Light Wavefield Dehazing)

整体为 **AStack**：固定 3 层 **ABlock** + 层间残差 + 可选 **ColorCorrector**。

---

## 1. AStack（主网络）

- **3 × ABlock**：串行，每层后加可学习残差 `out = ABlock(in) + res_scale * in`。
- **3 个标量参数**：`res_scale_0/1/2`（层间残差权重）。
- **ColorCorrector**（可选）：后处理颜色校正 `scale * dehazed + bias`。

数据流：`x → ABlock1 → +res0·x → ABlock2 → +res1·x1 → ABlock3 → +res2·x2 → clamp → [ColorCorrector] → out`。

---

## 2. 单层 ABlock 的组成

每一层 ABlock 内部由以下部分构成：

| 模块 | 作用 | 主要参数 |
|------|------|----------|
| **GroupNorm(1, 3)** | 输入归一化 | 极少 |
| **PhaseModule** | 预测复振幅相位 φ ∈ [-π, π] | in_proj, ResBlock×N, out_shared, out_resid |
| **ZModule** | 预测传播距离 z ∈ [0, z_max] | in_proj, ResBlock×N, out_proj |
| **FreqEnhance** | 频域增强（FFT 后、IFFT 前） | 小 Conv 1×1 |
| **ASM 传播** | 角谱法 Uz = IFFT( H · FFT(U0) )，无参数 | 0 |
| **Mix 头** | [x, J_luma, J_contrast] → Δ，两层 Conv + GN | 5→mix_hidden→3，占 ABlock 大部分参数 |
| **_ChannelAttention (SE)** | 对 Δ 做通道注意力 | 两个 Linear |
| **alpha** | 可学习残差步长 I_{t+1} = I_t + α·Δ | 1 标量 |

---

## 3. PhaseModule（相位预测）

- **in_proj**：Conv2d(3 → phase_hidden, 3×3)
- **ResBlock × num_layers**：每组 2 个 3×3 Conv + GroupNorm
- **out_shared**：Conv2d(phase_hidden → 1)，共享相位
- **out_resid**：Conv2d(phase_hidden → 3)，通道残差（× phase_residual_scale）

参数量主要来自 `phase_hidden` 和 `num_layers`。

---

## 4. ZModule（传播距离预测）

- **in_proj**：Conv2d(3 → z_hidden, 3×3)
- **ResBlock × num_layers**：同 PhaseModule 结构
- **out_proj**：Conv2d(z_hidden → 3, 1×1)

参数量主要来自 `z_hidden` 和 `num_layers`。

---

## 5. 其他

- **FreqEnhance**：Conv2d(6 → hidden → 6, 1×1)，参数量很小。
- **ColorCorrector**：Conv 提取 [dehazed, original]，再预测 scale/bias，参数量由 `color_corrector.hidden` 决定。

---

## 6. 减参建议（在 config 或代码默认值里调）

| 配置项 | 当前典型值 | 可改为 | 效果 |
|--------|------------|--------|------|
| phase_module.hidden | 32 | 16 | 相位子网约减半 |
| phase_module.num_layers | 2 | 1 | 少一层 ResBlock |
| z_module.hidden | 32 | 16 | z 子网约减半 |
| z_module.num_layers | 2 | 1 | 少一层 ResBlock |
| mix_hidden | 64 | 32 | Mix 头明显减小 |
| color_corrector.hidden | 32 | 16 | 颜色校正减小 |
| FreqEnhance.hidden (代码里) | 16 | 8 | 频域增强略减 |

上述都改完后，总参数量会明显下降，适合轻量化部署。
