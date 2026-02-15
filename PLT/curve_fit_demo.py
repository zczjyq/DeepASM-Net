"""
曲线拟合 Demo：多项式拟合 + 自定义函数拟合
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ========== 1. 准备数据（带一点噪声，模拟真实测量）==========
np.random.seed(42)
x = np.linspace(0, 4, 20)
y_true = 0.5 * x**2 - 1.2 * x + 0.8
y = y_true + np.random.normal(0, 0.15, size=x.shape)

# ========== 2. 多项式拟合 (numpy.polyfit) ==========
deg = 2
coefs = np.polyfit(x, y, deg)
y_poly = np.polyval(coefs, x)

# ========== 3. 自定义函数拟合 (scipy.curve_fit) ==========
def model(x, a, b, c):
    return a * x**2 + b * x + c

popt, _ = curve_fit(model, x, y, p0=[0.5, -1, 1])
y_curve = model(x, *popt)

# ========== 4. 画图 ==========
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="观测点", color="gray", s=50, zorder=3)
plt.plot(x, y_true, "k--", alpha=0.6, label="真实曲线 (无噪声)")
plt.plot(x, y_poly, "b-", lw=2, label=f"多项式拟合 (阶数={deg})")
plt.plot(x, y_curve, "r-", lw=2, label="curve_fit 拟合")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("curve_fit_demo.png", dpi=150)
plt.show()

# ========== 5. 打印结果 ==========
print("多项式系数 (高次到低次):", np.round(coefs, 4))
print("curve_fit 参数 (a, b, c):", np.round(popt, 4))
