import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_sklearn
import torch
import numpy as np

# ==========================================
# 1. 定义一个计算 PSNR 和 SSIM 的工具函数
# ==========================================
def calc_metrics(pred_tensor, gt_tensor):
    """
    输入: (1, 3, H, W) 的 Tensor, 范围 0~1
    输出: psnr, ssim (float)
    """
    # --- 1. 准备数据 (转 numpy + 调整维度) ---
    # Tensor (1,3,H,W) -> Numpy (H,W,3)
    pred_np = pred_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    gt_np   = gt_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # 确保范围在 0~1 (防止 float 误差导致略微越界)
    pred_np = np.clip(pred_np, 0, 1)
    gt_np   = np.clip(gt_np, 0, 1)

    # --- 2. 计算 PSNR ---
    # 公式: 10 * log10(MAX^2 / MSE)
    mse = np.mean((pred_np - gt_np) ** 2)
    if mse == 0:
        psnr = 100.0 # 完美匹配
    else:
        psnr = 10 * np.log10(1.0 / mse)

    # --- 3. 计算 SSIM ---
    # channel_axis=2 表示第3个维度是通道 (H, W, C)
    # data_range=1.0 表示像素值范围是 0~1
    try:
        ssim = ssim_sklearn(
            gt_np, 
            pred_np, 
            data_range=1.0, 
            channel_axis=2  # 新版 skimage 写法
        )
    except TypeError:
        # 兼容旧版 skimage (如果报错用这个)
        ssim = ssim_sklearn(
            gt_np, 
            pred_np, 
            data_range=1.0, 
            multichannel=True 
        )

    return psnr, ssim
# ==========================================
# 1. 物理层: 支持单通道 Z 和 Phi 的 ASM
# ==========================================
def asm_propagate_broadcast(U0, z, phi, wavelengths):
    """
    U0: (B, 3, H, W) - 复振幅初始场
    z:  (B, 1, H, W) - 物理距离 (2D矩阵)
    phi:(B, 1, H, W) - 相位调制 (2D矩阵)
    wavelengths: (3,) - RGB波长
    """
    b, c, h, w = U0.shape
    device = U0.device
    
    # 1. 构建复振幅 (Amplitude * exp(j * phi))
    # 注意: phi 是 (B,1,H,W), 会自动广播给 3 个通道
    U_input = U0 * torch.exp(1j * phi)

    # 2. 频率网格
    fx = torch.fft.fftfreq(w, d=1.0, device=device)
    fy = torch.fft.fftfreq(h, d=1.0, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy') # 注意 meshgrid 顺序
    FX, FY = FX.unsqueeze(0).unsqueeze(0), FY.unsqueeze(0).unsqueeze(0) # (1,1,H,W)

    # 3. 准备波长 (1, 3, 1, 1)
    lam = wavelengths.to(device).view(1, c, 1, 1)
    
    # 4. 计算传递函数 H
    # 公式: H = exp(j * 2pi/lambda * z * sqrt(1 - (lambda*fx)^2 ...))
    # z 是 (B,1,H,W), lam 是 (1,3,1,1) -> 结果自动广播为 (B,3,H,W)
    squared_term = 1 - (lam * FX)**2 - (lam * FY)**2
    squared_term = torch.clamp(squared_term, min=0) # 物理截断
    
    k = 2 * math.pi / lam
    phase_delay = k * z * torch.sqrt(squared_term)
    H = torch.exp(1j * phase_delay)

    # 5. 频域传播
    U_freq = torch.fft.fft2(U_input)
    U_z_freq = U_freq * H
    U_z = torch.fft.ifft2(U_z_freq)
    
    J = torch.abs(U_z)
    return J

# ==========================================
# 2. 网络模型 (修改为输出 2D 矩阵)
# ==========================================
class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GroupNorm(1, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GroupNorm(1, ch)
        )
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class SingleChPredictor(nn.Module):
    """通用的单通道预测器 (用于 Z 和 Phase)"""
    def __init__(self, in_ch=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1),
            _ResBlock(hidden),
            _ResBlock(hidden),
            nn.Conv2d(hidden, 1, 1) # <--- 关键：输出通道为 1
        )
    def forward(self, x): return self.net(x)

class SimpleEndoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 物理波长 (红/绿/蓝)
        self.register_buffer('wavelengths', torch.tensor([0.66, 0.53, 0.45]))
        
        # 两个预测器，都输出 (B, 1, H, W)
        self.phase_net = SingleChPredictor()
        self.z_net = SingleChPredictor()
        
        # MixHead
        self.mix = nn.Sequential(
            nn.Conv2d(3+2, 32, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(32, 3, 1)
        )
        self.luma_w = torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1)

    def forward(self, x):
        # 1. 预测物理参数 (2D矩阵)
        phi_raw = self.phase_net(x)
        z_raw = self.z_net(x)
        
        # 激活函数控制范围
        phi = math.pi * torch.tanh(phi_raw)       # -pi 到 pi
        z = 0.1 * torch.sigmoid(z_raw)            # 0 到 0.1 (单位任意，假设是微距)
        
        # 2. ASM 传播
        # 把 x 当作振幅 A (归一化), 相位由 phi 提供
        J = asm_propagate_broadcast(x, z, phi, self.wavelengths)
        
        # 3. Mix & Residual
        # 简单的结构提取
        x_luma = (x * self.luma_w.to(x.device)).sum(1, keepdim=True)
        J_luma = (J * self.luma_w.to(x.device)).sum(1, keepdim=True)
        diff = J_luma - x_luma
        
        inp = torch.cat([x, J_luma, diff], dim=1)
        delta = self.mix(inp)
        
        return x + delta, z, phi

def cv_imread(file_path):
    """
    专门用来读取带中文路径图片的函数
    """
    # 1. 先用 numpy 把文件读成二进制流
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    
    # 2. 如果读出来是 None (比如路径不对)，直接抛错
    if cv_img is None:
        raise ValueError(f"❌ 读取失败，请检查路径: {file_path}")
        
    return cv_img
# 这种组合通常效果最好：既保真度高（MSE），又细节清晰（L1）
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # 0.8 倍的 L1 (为了边缘锐利) + 0.2 倍的 MSE (为了 PSNR 跑分高)
        return 0.2 * self.l1(pred, target) + 0.8 * self.mse(pred, target)
        
loss_fn = HybridLoss()
# ==========================================
# 3. 实验设置 (单图训练 - 真实成对数据版)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

# --- A. 准备数据 (读取双图) ---

# 👇 1. 在这里填入两张图片的路径
clean_path = r"D:\Desktop\毕业论文\code\data\RESIDE-IN\train\GT\1_1_0.90179.png"   # 清晰的 Ground Truth 图片
hazy_path  = r"D:\Desktop\毕业论文\code\data\RESIDE-IN\train\hazy\1_1_0.90179.png"    # 对应的 雾图 Input

# 读取图片
img_clean_bgr = cv_imread(clean_path)
img_hazy_bgr  = cv_imread(hazy_path)

# 检查是否读取成功
if img_clean_bgr is None: raise ValueError(f"❌ 找不到清晰图: {clean_path}")
if img_hazy_bgr is None:  raise ValueError(f"❌ 找不到雾图: {hazy_path}")

# 👇 2. 强制统一尺寸
# 神经网络训练要求输入和输出必须像素对齐，尺寸完全一致
# 建议缩放到 256x256 或 512x512，过大显存会爆
H, W = 256, 256
img_clean_bgr = cv2.resize(img_clean_bgr, (H, W))
img_hazy_bgr  = cv2.resize(img_hazy_bgr,  (H, W))

# 👇 3. 预处理 (转 RGB -> 归一化 0~1)
gt_img = cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
input_img = cv2.cvtColor(img_hazy_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# 👇 4. 转 Tensor (1, 3, H, W)
t_gt = torch.from_numpy(gt_img).permute(2,0,1).unsqueeze(0).to(device)    # Ground Truth
t_in = torch.from_numpy(input_img).permute(2,0,1).unsqueeze(0).to(device) # Input (Hazy)

print(f"✅ 数据加载成功!")
print(f"   清晰图 GT   : {clean_path} {t_gt.shape}")
print(f"   雾图 Input  : {hazy_path} {t_in.shape}")

# --- B. 初始化 ---
model = SimpleEndoNet().to(device)
model.load_state_dict(torch.load("model.pth"))
optimizer = optim.Adam(model.parameters(), lr=0.01) # 单图训练学习率可以大点
loss_fn = HybridLoss()

# --- C. 训练循环 ---
epochs = 4000
pbar = tqdm(range(epochs))

loss_history = []
psnr_history = []
ssim_history = []
for i in pbar:
    optimizer.zero_grad()
    
    # 前向
    out, z_map, phi_map = model(t_in)
    
    # 计算 Loss (让输出逼近 GT)
    loss = loss_fn(out, t_gt)
    
    # 反向
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    # pbar.set_description(f"Loss: {loss.item():.6f}")
    
    if i % 100 == 0:
        # 简单的学习率衰减
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
    # --- B. 每 50 轮评估一次 ---
    if (i + 1) % 50 == 0:
        model.eval() # 🔕 关闭梯度计算和 Dropout
        with torch.no_grad():
            # 预测一次
            val_out, _, _ = model(t_in)
            
            # 计算指标
            cur_psnr, cur_ssim = calc_metrics(val_out, t_gt)
            
            # 记录
            psnr_history.append(cur_psnr)
            ssim_history.append(cur_ssim)
            
            # 在进度条上显示 (看起来很专业)
            pbar.set_description(
                f"Loss:{loss.item():.4f} | PSNR:{cur_psnr:.2f}dB | SSIM:{cur_ssim:.4f}"
            )
            
            # 可选：如果你想看到具体的打印
            # print(f"\n[Epoch {i+1}] PSNR: {cur_psnr:.2f} | SSIM: {cur_ssim:.4f}")

# ==========================================
# 4. 结果可视化
# ==========================================
model.eval()
with torch.no_grad():
    out, z_map, phi_map = model(t_in)

torch.save(model.state_dict(), "model.pth")

# 转回 numpy
res_img = out.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)
in_show = input_img # HWC
gt_show = gt_img    # HWC
z_show = z_map.squeeze().cpu().numpy()
phi_show = phi_map.squeeze().cpu().numpy()

plt.figure(figsize=(15, 8))

# 1. 输入 (模拟雾图)
plt.subplot(2, 4, 1)
plt.title("Input (Hazy)")
plt.imshow(in_show)
plt.axis('off')

# 2. 你的网络输出
plt.subplot(2, 4, 2)
plt.title(f"Output (Dehazed)\nLoss: {loss_history[-1]:.5f}")
plt.imshow(res_img)
plt.axis('off')

# 3. Ground Truth
plt.subplot(2, 4, 3)
plt.title("Ground Truth")
plt.imshow(gt_show)
plt.axis('off')

# 4. 预测的 Z 矩阵 (物理距离)
plt.subplot(2, 4, 4)
plt.title("Predicted Distance Z (2D Matrix)")
plt.imshow(z_show, cmap='inferno')
plt.colorbar()
plt.axis('off')

# 5. 预测的 Phase 矩阵 (相位)
plt.subplot(2, 4, 5)
plt.title("Predicted Phase Phi (2D Matrix)")
plt.imshow(phi_show, cmap='twilight')
plt.colorbar()
plt.axis('off')

# 6. Loss 曲线
plt.subplot(2, 4, 6)
plt.title("Training Loss")
plt.plot(loss_history)
plt.grid(True)

plt.subplot(2, 4, 7)
plt.title("PSNR")
plt.plot(psnr_history)
plt.grid(True)

plt.subplot(2, 4, 8)
plt.title("SSIM")
plt.plot(ssim_history)
plt.grid(True)

plt.tight_layout()
plt.show()