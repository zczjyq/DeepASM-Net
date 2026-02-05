"""
绘制 Wavefield Dehaze 网络架构图
风格参考 DehazeFormer，分整体结构和 ABlock 详图两部分

运行: python draw_architecture.py
输出: architecture_diagram.png (保存在脚本同目录)
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_box(ax, x, y, w, h, label, color='#E8F4F8', edge='#2E86AB'):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", 
                         facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, wrap=True)


def draw_arrow(ax, start, end, color='#333', style='->'):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color=color, lw=1.5))


def draw_architecture():
    fig = plt.figure(figsize=(14, 10))
    
    # ========== 上半部分：整体网络结构 ==========
    ax1 = fig.add_axes([0.02, 0.5, 0.96, 0.48])
    ax1.set_xlim(0, 17)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    ax1.set_title('Wavefield Dehaze Network — AStack', fontsize=14, fontweight='bold')
    
    # 输入
    draw_box(ax1, 0.5, 2.5, 1.2, 1, 'Input\nI\nHxWx3', '#FFF3E0', '#E65100')
    
    # ABlock 链
    for i in range(4):
        x = 2.2 + i * 2.6
        draw_box(ax1, x, 2.2, 2.2, 1.6, f'ABlock ×{i+1}', '#E3F2FD', '#1565C0')
    
    # 箭头：输入 -> ABlock1
    draw_arrow(ax1, (1.7, 3), (2.2, 2.8))
    # 箭头：ABlock 之间
    for i in range(3):
        draw_arrow(ax1, (4.4 + i*2.6, 3), (4.8 + i*2.6, 3))
    
    # Clamp
    draw_box(ax1, 12.8, 2.5, 0.8, 1, 'Clamp\n[0,1]', '#F3E5F5', '#7B1FA2')
    draw_arrow(ax1, (12.4, 3), (12.8, 3))
    
    # ColorCorrector (可选)
    draw_box(ax1, 13.8, 2.2, 0.9, 1.6, 'Color\nCorrector\n(可选)', '#E8F5E9', '#2E7D32')
    draw_arrow(ax1, (13.6, 3), (13.8, 3))
    
    # 输出
    draw_box(ax1, 15.2, 2.5, 1.2, 1, 'Output\nJ_hat\nHxWx3', '#E8F5E9', '#1B5E20')
    draw_arrow(ax1, (14.7, 3), (15.2, 3))
    
    # 残差连接示意（在 ABlock 内部，此处标注）
    ax1.text(7, 1.5, 'I_{t+1} = I_t + alpha*Delta  (ABlock residual)', 
             ha='center', fontsize=10, style='italic', color='#666')
    
    # ========== 下半部分：ABlock 详图 ==========
    ax2 = fig.add_axes([0.02, 0.02, 0.96, 0.46])
    ax2.set_xlim(0, 15.5)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    ax2.set_title('ABlock (波动传播去雾块) 结构详解', fontsize=14, fontweight='bold')
    
    # 输入 x
    draw_box(ax2, 0.3, 3.5, 0.9, 0.7, 'x\n(B,3,H,W)', '#FFF8E1', '#FF8F00')
    
    # Norm
    draw_box(ax2, 1.4, 3.6, 0.7, 0.5, 'GroupNorm', '#E8F5E9', '#388E3C')
    draw_arrow(ax2, (1.2, 3.85), (1.4, 3.85))
    
    # PhaseModule
    draw_box(ax2, 2.3, 3.2, 1.4, 1.2, 'PhaseModule\nConv->ResBlock x N\nphi=pi*tanh(...)', '#E3F2FD', '#1976D2')
    draw_arrow(ax2, (2.1, 3.85), (2.3, 3.85))
    
    # ZModule
    draw_box(ax2, 2.3, 1.8, 1.4, 1.0, 'ZModule\nConv->ResBlock x N\nz=z_max*tanh(z_raw)', '#FFF3E0', '#F57C00')
    draw_arrow(ax2, (2.1, 3.5), (2.3, 2.3))
    
    # 分支：A, phi -> U0
    draw_box(ax2, 4.0, 3.6, 1.2, 0.6, 'U0=A*exp(i*phi)', '#FCE4EC', '#C2185B')
    draw_arrow(ax2, (3.7, 3.85), (4.0, 3.9))
    draw_arrow(ax2, (3.7, 2.8), (4.0, 3.6))
    
    # ASM
    draw_box(ax2, 5.5, 2.8, 1.8, 1.6, 'ASM Propagation\nFFT2(U0)*H(lambda,z)\nIFFT2 -> J=|Uz|', '#E8EAF6', '#3949AB')
    draw_arrow(ax2, (5.2, 3.9), (5.5, 3.6))
    draw_arrow(ax2, (3.7, 2.3), (5.5, 2.8))
    
    # J_luma, J_contrast
    draw_box(ax2, 7.6, 3.2, 1.3, 0.9, 'J_luma\nJ_contrast', '#E0F7FA', '#00838F')
    draw_arrow(ax2, (7.3, 3.6), (7.6, 3.65))
    
    # Concat & Mix
    draw_box(ax2, 9.2, 3.0, 1.5, 1.2, 'Concat [x,J_luma,\nJ_contrast]\nMix Conv -> Delta', '#F1F8E9', '#689F38')
    draw_arrow(ax2, (9.3, 3.65), (9.2, 3.6))
    draw_arrow(ax2, (1.2, 3.5), (9.0, 3.3))
    
    # Channel Attention
    draw_box(ax2, 11.0, 3.2, 1.1, 0.8, 'Channel\nAttention', '#F3E5F5', '#7B1FA2')
    draw_arrow(ax2, (10.7, 3.6), (11.0, 3.6))
    
    # Residual
    draw_box(ax2, 12.4, 3.3, 1.3, 0.6, 'x + alpha*Delta', '#E8F5E9', '#2E7D32')
    draw_arrow(ax2, (11.1, 3.6), (12.4, 3.6))
    draw_arrow(ax2, (0.75, 3.5), (12.2, 3.4))
    
    # 输出
    draw_box(ax2, 14.0, 3.3, 0.8, 0.6, 'out', '#C8E6C9', '#1B5E20')
    draw_arrow(ax2, (13.7, 3.6), (14.0, 3.6))
    
    # PhaseModule 详图（左侧小框）
    ax2.add_patch(FancyBboxPatch((0.5, 0.3), 3.5, 1.2, boxstyle="round,pad=0.02", 
                                 facecolor='#FAFAFA', edgecolor='#9E9E9E', linewidth=1, linestyle='--'))
    ax2.text(2.25, 1.35, 'PhaseModule: in_proj -> ResBlock x N -> phi_shared(1ch) + delta_phi(3ch)', 
             ha='center', fontsize=8, color='#424242')
    ax2.text(2.25, 0.85, 'ZModule: in_proj -> ResBlock x N -> z (B,3,H,W)', 
             ha='center', fontsize=8, color='#424242')
    ax2.text(2.25, 0.5, 'ASM: H=exp(ikz*sqrt(1-(lambda*fx)^2-(lambda*fy)^2))', 
             ha='center', fontsize=8, color='#424242')
    
    out_path = os.path.join(os.path.dirname(__file__), 'architecture_diagram.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print('已保存:', out_path)
    plt.close(fig)


if __name__ == '__main__':
    draw_architecture()
