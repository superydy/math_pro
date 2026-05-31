#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带透视感的3D层叠风格模型结构图
参考：神经网络论文图风格，用 matplotlib 模拟
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os
os.chdir('/home/user/math_pro')

from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('paper_figures', exist_ok=True)

# ── 颜色工具 ─────────────────────────────────────────────────────────────────
def lighten(hex_color, factor=0.35):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    r = int(r + (255-r)*factor); g = int(g + (255-g)*factor); b = int(b + (255-b)*factor)
    return f'#{r:02X}{g:02X}{b:02X}'

def darken(hex_color, factor=0.35):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    r = int(r*(1-factor)); g = int(g*(1-factor)); b = int(b*(1-factor))
    return f'#{r:02X}{g:02X}{b:02X}'

# ── 核心绘图：3D 透视块 ───────────────────────────────────────────────────────
def block3d(ax, x, y, w, h, d=0.18, color='#4A90D9', label='', sublabel='',
            fontsize=8, zorder=3, label_above=True, n_slices=1, gap=0.06):
    """
    绘制带透视的3D层叠块（可选多层切片）
    x,y: 左下角坐标  w,h: 宽高  d: 透视深度偏移
    """
    dx, dy = d, d * 0.55   # 透视偏移

    for s in range(n_slices-1, -1, -1):
        ox = s * gap * 0.7
        oy = s * gap * 0.4
        fc  = lighten(color, 0.08*s) if s > 0 else color
        fcd = darken(fc, 0.30)
        fcl = lighten(fc, 0.30)

        # 前面
        front = plt.Polygon(
            [[x+ox, y+oy], [x+ox+w, y+oy], [x+ox+w, y+oy+h], [x+ox, y+oy+h]],
            closed=True, facecolor=fc, edgecolor='white', linewidth=1.2, zorder=zorder+s)
        ax.add_patch(front)
        # 顶面
        top = plt.Polygon(
            [[x+ox, y+oy+h], [x+ox+w, y+oy+h],
             [x+ox+w+dx, y+oy+h+dy], [x+ox+dx, y+oy+h+dy]],
            closed=True, facecolor=fcl, edgecolor='white', linewidth=1.0, zorder=zorder+s)
        ax.add_patch(top)
        # 右侧面
        right = plt.Polygon(
            [[x+ox+w, y+oy], [x+ox+w+dx, y+oy+dy],
             [x+ox+w+dx, y+oy+h+dy], [x+ox+w, y+oy+h]],
            closed=True, facecolor=fcd, edgecolor='white', linewidth=1.0, zorder=zorder+s)
        ax.add_patch(right)

    # 标签（显示在最前面切片上）
    cx = x + w/2
    cy = y + h/2
    if label:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold',
                zorder=zorder+n_slices+1)
    if sublabel:
        if label_above:
            ax.text(cx + dx*0.5, y+h+dy+0.08, sublabel,
                    ha='center', va='bottom', fontsize=fontsize-0.5,
                    color='#333', zorder=zorder+n_slices+1)
        else:
            ax.text(cx, y-0.15, sublabel,
                    ha='center', va='top', fontsize=fontsize-0.5,
                    color='#555', zorder=zorder+n_slices+1)

def arrow3d(ax, x1, y1, x2, y2, color='#555555', lw=1.5, zorder=10):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=12),
                zorder=zorder)

def label_text(ax, x, y, text, color='#CC0000', fontsize=8.5, ha='center', bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            color=color, fontweight='bold' if bold else 'normal')


# ═══════════════════════════════════════════════════════════════════════════════
# 图1：随机森林
# ═══════════════════════════════════════════════════════════════════════════════
def draw_rf():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── 输入特征块
    block3d(ax, 0.3, 1.2, 0.55, 2.6, d=0.22, color='#5B8DB8',
            label='X', sublabel='1624x84\n输入特征', n_slices=1, fontsize=9)
    label_text(ax, 0.72, 4.3, 'Input', color='#5B8DB8', fontsize=8)

    arrow3d(ax, 1.08, 2.5, 1.5, 2.5)

    # ── Bootstrap 采样（3组）
    b_colors = ['#7EB5A6', '#7EB5A6', '#7EB5A6']
    b_xs = [1.6, 1.6, 1.6]
    b_ys = [3.5, 2.1, 0.6]
    for i, (bx, by, bc) in enumerate(zip(b_xs, b_ys, b_colors)):
        block3d(ax, bx, by, 0.5, 0.75, d=0.14, color=bc,
                n_slices=3, gap=0.05,
                sublabel=f'样本{i+1}' if i < 2 else '样本N',
                label_above=False, fontsize=7.5)

    ax.text(1.85, 2.58, '...', ha='center', fontsize=13, color='#888', va='center')

    # Bootstrap 标签
    label_text(ax, 1.85, 5.1, 'Bootstrap\n采样', color='#5A9E8F', fontsize=8)
    for by in b_ys:
        arrow3d(ax, 2.26, by+0.375, 2.7, by+0.375, color='#888')

    # ── 决策树（3棵，梯形表示）
    t_xs = [2.8, 2.8, 2.8]
    t_ys = [3.2, 1.75, 0.25]
    t_colors = ['#E8834E', '#D4694B', '#C0553A']
    for i, (tx, ty, tc) in enumerate(zip(t_xs, t_ys, t_colors)):
        # 三角形树形
        tri = plt.Polygon(
            [[tx, ty], [tx+1.1, ty], [tx+0.55, ty+1.1]],
            closed=True, facecolor=tc, edgecolor='white', lw=1.4, zorder=4, alpha=0.92)
        ax.add_patch(tri)
        ax.text(tx+0.55, ty+0.38, f'树 {i+1}' if i<2 else '树 T',
                ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=5)
        # 内部节点线
        ax.plot([tx+0.55, tx+0.28], [ty+0.72, ty+0.35], '-', color='white', lw=0.8, alpha=0.7, zorder=5)
        ax.plot([tx+0.55, tx+0.82], [ty+0.72, ty+0.35], '-', color='white', lw=0.8, alpha=0.7, zorder=5)
        arrow3d(ax, tx+1.1, ty+0.38, tx+1.55, ty+0.38, color='#888')

    ax.text(3.35, 2.58, '...', ha='center', fontsize=13, color='#888', va='center')
    label_text(ax, 3.35, 5.1, '决策树 x T\n(并行)', color='#C0553A', fontsize=8)

    # ── 预测结果（3个小块）
    p_ys = [3.2, 1.75, 0.25]
    for i, py in enumerate(p_ys):
        block3d(ax, 4.45, py+0.05, 0.55, 0.65, d=0.12, color='#6BAE6E',
                label=f'y{i+1}' if i<2 else 'yT',
                sublabel='', fontsize=8, n_slices=1)

    ax.text(4.73, 2.58, '...', ha='center', fontsize=13, color='#888', va='center')
    label_text(ax, 4.73, 5.1, '单树\n预测', color='#4A8C4C', fontsize=8)

    for py in p_ys:
        arrow3d(ax, 5.0, py+0.375, 5.5, 2.5, color='#4A8C4C')

    # ── 均值聚合块
    block3d(ax, 5.5, 1.5, 0.85, 2.0, d=0.20, color='#4472C4',
            label='AVG\n均值', sublabel='mean(y1...yT)', fontsize=8.5, n_slices=1)
    label_text(ax, 5.95, 4.9, '平均\n聚合', color='#4472C4', fontsize=8)
    arrow3d(ax, 6.55, 2.5, 7.1, 2.5)

    # ── 输出
    block3d(ax, 7.1, 1.85, 0.55, 1.3, d=0.16, color='#C0504D',
            label='CO\nppm', sublabel='预测输出\n1x1', fontsize=8, n_slices=1)
    label_text(ax, 7.38, 4.9, 'Output', color='#C0504D', fontsize=8)

    ax.set_title('随机森林 (Random Forest)  结构示意图', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('paper_figures/arch2_rf.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ arch2_rf.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图2：梯度提升树 GBR
# ═══════════════════════════════════════════════════════════════════════════════
def draw_gbr():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis('off')
    fig.patch.set_facecolor('white')

    # 输入
    block3d(ax, 0.3, 1.3, 0.55, 2.2, d=0.2, color='#5B8DB8',
            label='X', sublabel='1624x84', fontsize=9, n_slices=1)
    label_text(ax, 0.65, 4.2, 'Input', color='#5B8DB8', fontsize=8)
    arrow3d(ax, 1.05, 2.4, 1.55, 2.4)

    # 初始预测 F0
    block3d(ax, 1.55, 1.65, 0.7, 1.5, d=0.18, color='#9B59B6',
            label='F0', sublabel='F0=mean(y)\n初始预测', fontsize=8, n_slices=1)
    label_text(ax, 1.9, 4.2, 'F0', color='#9B59B6', fontsize=8, bold=True)
    arrow3d(ax, 2.43, 2.4, 2.95, 2.4)

    # 串行弱学习器 t=1,2,...,T
    stage_colors = ['#E67E22', '#E74C3C', '#C0392B']
    stage_xs = [3.0, 5.5, 8.0]
    for i, (sx, sc) in enumerate(zip(stage_xs, stage_colors)):
        lbl = f't={i+1}' if i < 2 else 't=T'
        # 残差块
        block3d(ax, sx, 3.05, 0.62, 0.75, d=0.13, color='#F39C12',
                label='r', sublabel=f'残差rt\ny-Ft-1', fontsize=7.5, n_slices=2, gap=0.04)
        # 弱学习器树
        tri = plt.Polygon(
            [[sx+0.05, 1.15], [sx+0.97, 1.15], [sx+0.51, 2.1]],
            closed=True, facecolor=sc, edgecolor='white', lw=1.4, zorder=4, alpha=0.92)
        ax.add_patch(tri)
        ax.text(sx+0.51, 1.55, lbl, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=5)
        ax.plot([sx+0.51, sx+0.28], [sx*0+1.75, sx*0+1.38], '-', color='white', lw=0.8, alpha=0.7, zorder=5)
        ax.plot([sx+0.51, sx+0.74], [sx*0+1.75, sx*0+1.38], '-', color='white', lw=0.8, alpha=0.7, zorder=5)
        # 更新块
        block3d(ax, sx+0.08, 0.15, 0.65, 0.75, d=0.13, color='#2ECC71',
                label=f'Ft', sublabel=f'Ft=Ft-1\n+eta*ht', fontsize=7.5, n_slices=1)
        # 内部箭头
        arrow3d(ax, sx+0.41, 3.05, sx+0.41, 2.12, color='#F39C12')
        arrow3d(ax, sx+0.41, 1.15, sx+0.41, 0.9, color=sc)
        if i < 2:
            arrow3d(ax, sx+0.9, 0.55, sx+2.5, 0.55, color='#27AE60', lw=1.8)
            arrow3d(ax, sx+0.75, 2.4, sx+2.5, 2.4, color='#E67E22')
            label_text(ax, sx+1.7, 4.85, '', fontsize=7.5)

    ax.text(10.85, 2.4, '...', ha='center', fontsize=15, color='#888', va='center')
    ax.text(10.85, 0.55, '...', ha='center', fontsize=15, color='#888', va='center')

    label_text(ax, 6.7, 4.85, '串行弱学习器 (逐步拟合残差)', color='#C0392B', fontsize=9, bold=True)

    # 最终聚合
    block3d(ax, 11.5, 1.5, 0.85, 1.8, d=0.2, color='#4472C4',
            label='SUM\nF_T', sublabel='F0+eta*sum(ht)\n最终集成', fontsize=8, n_slices=1)
    arrow3d(ax, 10.4, 0.55, 11.7, 1.55, color='#27AE60', lw=1.4)
    label_text(ax, 12.1, 4.85, '集成', color='#4472C4', fontsize=8)
    arrow3d(ax, 12.55, 2.4, 13.1, 2.4)

    # 输出
    block3d(ax, 13.1, 1.7, 0.55, 1.35, d=0.15, color='#C0504D',
            label='CO\nppm', sublabel='输出', fontsize=8, n_slices=1)
    label_text(ax, 13.38, 4.85, 'Output', color='#C0504D', fontsize=8)

    ax.set_title('梯度提升树 (GBR)  结构示意图', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('paper_figures/arch2_gbr.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ arch2_gbr.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图3：LightGBM
# ═══════════════════════════════════════════════════════════════════════════════
def draw_lgbm():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.5); ax.axis('off')
    fig.patch.set_facecolor('white')

    # 输入
    block3d(ax, 0.2, 1.3, 0.55, 2.2, d=0.2, color='#5B8DB8',
            label='X', sublabel='1624x84', fontsize=9, n_slices=1)
    label_text(ax, 0.55, 4.2, 'Input', color='#5B8DB8', fontsize=8)
    arrow3d(ax, 0.95, 2.4, 1.45, 2.4)

    # 直方图分箱
    block3d(ax, 1.45, 0.8, 1.1, 3.2, d=0.22, color='#8E44AD',
            label='Hist\nBin', sublabel='直方图分箱\n255 bins\nO(#bins)', fontsize=8, n_slices=4, gap=0.07)
    label_text(ax, 2.05, 5.0, 'Histogram\nBinning', color='#8E44AD', fontsize=8)
    arrow3d(ax, 2.87, 2.4, 3.4, 2.4)

    # GOSS 采样块
    block3d(ax, 3.4, 1.5, 1.0, 1.8, d=0.2, color='#16A085',
            label='GOSS', sublabel='梯度单边\n采样\n大梯度全保留\n小梯度随机', fontsize=7.5, n_slices=3, gap=0.06)
    label_text(ax, 3.9, 5.0, 'GOSS', color='#16A085', fontsize=8)
    arrow3d(ax, 4.62, 2.4, 5.1, 2.4)

    # EFB 特征捆绑
    block3d(ax, 5.1, 1.5, 0.95, 1.8, d=0.2, color='#2980B9',
            label='EFB', sublabel='互斥特征\n捆绑\n稀疏特征\n合并降维', fontsize=7.5, n_slices=3, gap=0.06)
    label_text(ax, 5.6, 5.0, 'EFB', color='#2980B9', fontsize=8)
    arrow3d(ax, 6.27, 2.4, 6.8, 2.4)

    # Leaf-wise 树生长（多棵树堆叠）
    tree_colors = ['#E74C3C','#E67E22','#F1C40F','#2ECC71','#3498DB']
    for i, tc in enumerate(tree_colors):
        ox = i * 0.12; oy = i * 0.06
        tri = plt.Polygon(
            [[6.8+ox, 0.5+oy], [7.95+ox, 0.5+oy], [7.375+ox, 1.75+oy]],
            closed=True, facecolor=tc, edgecolor='white', lw=1.2, zorder=4+i, alpha=0.88)
        ax.add_patch(tri)
    # 标注叶节点
    ax.text(8.3, 2.0, 'Leaf-wise\n分裂', ha='center', va='center',
            fontsize=7.5, color='#C0392B', fontweight='bold')
    label_text(ax, 7.6, 4.2, f'T棵树\nLeaf-wise', color='#C0392B', fontsize=8)
    ax.annotate('', xy=(8.9, 2.4), xytext=(8.5, 2.4),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5, mutation_scale=12), zorder=10)

    # 集成层
    block3d(ax, 8.9, 1.3, 1.0, 2.2, d=0.22, color='#4472C4',
            label='集成\nT棵', sublabel='boosting\n迭代集成', fontsize=8, n_slices=5, gap=0.06)
    label_text(ax, 9.45, 5.0, '集成输出', color='#4472C4', fontsize=8)
    arrow3d(ax, 10.12, 2.4, 10.7, 2.4)

    # 输出
    block3d(ax, 10.7, 1.65, 0.6, 1.5, d=0.16, color='#C0504D',
            label='CO\nppm', sublabel='预测输出', fontsize=8, n_slices=1)
    label_text(ax, 11.03, 5.0, 'Output', color='#C0504D', fontsize=8)

    ax.set_title('LightGBM  结构示意图', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('paper_figures/arch2_lgbm.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ arch2_lgbm.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 图4：XGBoost + PSO
# ═══════════════════════════════════════════════════════════════════════════════
def draw_xgb():
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 5.8); ax.axis('off')
    fig.patch.set_facecolor('white')

    # 输入
    block3d(ax, 0.2, 1.3, 0.55, 2.5, d=0.22, color='#5B8DB8',
            label='X', sublabel='1624x84', fontsize=9, n_slices=1)
    label_text(ax, 0.55, 4.5, 'Input', color='#5B8DB8', fontsize=8)
    arrow3d(ax, 0.97, 2.55, 1.5, 2.55)

    # PSO 粒子群优化（上方框）
    pso_box = mpatches.FancyBboxPatch((1.5, 4.15), 7.5, 1.25,
        boxstyle="round,pad=0.12", facecolor='#FFF9C4', edgecolor='#F9A825', lw=2, zorder=2)
    ax.add_patch(pso_box)
    ax.text(5.25, 4.82, 'PSO 粒子群优化  (8 粒子 x 10 迭代)', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#E65100', zorder=5)
    ax.text(5.25, 4.35, '搜索超参数: lr / max_depth / n_est / subsample / colsample / alpha / lambda',
            ha='center', va='center', fontsize=7.8, color='#5D4037', zorder=5)
    # PSO 双向箭头
    for xp in [2.2, 3.8, 5.4, 7.0]:
        ax.annotate('', xy=(xp, 4.15), xytext=(xp, 3.7),
                    arrowprops=dict(arrowstyle='<->', color='#F9A825', lw=1.6, mutation_scale=10), zorder=6)

    # 二阶梯度块
    block3d(ax, 1.5, 0.5, 0.95, 2.9, d=0.2, color='#8E44AD',
            label='2nd\nOrder', sublabel='二阶泰勒\n展开\ngi,hi', fontsize=8, n_slices=2, gap=0.06)
    label_text(ax, 2.0, 4.4, '二阶梯度', color='#8E44AD', fontsize=7.5)
    arrow3d(ax, 2.67, 2.55, 3.2, 2.55)

    # 树1-4（Level-wise，多棵堆叠）
    tree_stage_colors = [
        ['#E74C3C','#E67E22'],
        ['#E67E22','#F39C12'],
        ['#3498DB','#2980B9'],
        ['#2ECC71','#27AE60'],
    ]
    tree_xs = [3.2, 4.7, 6.2, 7.7]
    for i, (tx, tcs) in enumerate(zip(tree_xs, tree_stage_colors)):
        for j, tc in enumerate(tcs):
            ox = j*0.1; oy = j*0.05
            tri = plt.Polygon(
                [[tx+ox+0.05, 0.6+oy], [tx+ox+1.1, 0.6+oy], [tx+ox+0.575, 1.7+oy]],
                closed=True, facecolor=tc, edgecolor='white', lw=1.3, zorder=4+j, alpha=0.9)
            ax.add_patch(tri)
        ax.text(tx+0.625, 1.05, f'Tree\n{i+1}', ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=6)
        # L1+L2 正则 小标签
        ax.text(tx+0.625, 0.38, 'L1+L2 reg.', ha='center', va='center',
                fontsize=6.5, color='#888', style='italic', zorder=5)
        if i < 3:
            arrow3d(ax, tx+1.2, 1.15, tx+1.5, 1.15, color='#888')

    label_text(ax, 5.85, 4.42, f'XGBoost  T 棵树  (Level-wise + 正则化)', color='#C0392B', fontsize=8.5, bold=True)
    arrow3d(ax, 8.95, 1.15, 9.45, 2.55)

    # 残差修正块
    block3d(ax, 9.45, 1.4, 0.9, 2.3, d=0.2, color='#16A085',
            label='Res.\nUpdate', sublabel='逐步残差\n修正\nFt=Ft-1+ht', fontsize=7.8, n_slices=3, gap=0.06)
    label_text(ax, 9.9, 4.5, '残差更新', color='#16A085', fontsize=7.5)
    arrow3d(ax, 10.57, 2.55, 11.1, 2.55)

    # 正则化块
    block3d(ax, 11.1, 1.45, 0.95, 2.2, d=0.2, color='#7F8C8D',
            label='Reg.\nOmega', sublabel='gamma*T\n+lambda*||w||^2\n防过拟合', fontsize=7.8, n_slices=2, gap=0.06)
    label_text(ax, 11.62, 4.5, '正则化', color='#7F8C8D', fontsize=7.5)
    arrow3d(ax, 12.27, 2.55, 12.8, 2.55)

    # 集成输出块
    block3d(ax, 12.8, 1.5, 0.85, 2.1, d=0.2, color='#4472C4',
            label='SUM\nF_T', sublabel='T棵树\n集成输出', fontsize=8, n_slices=4, gap=0.06)
    label_text(ax, 13.28, 4.5, '集成', color='#4472C4', fontsize=8)
    arrow3d(ax, 13.87, 2.55, 14.3, 2.55)

    # 输出
    block3d(ax, 14.3, 1.75, 0.5, 1.6, d=0.15, color='#C0504D',
            label='CO\nppm', sublabel='输出', fontsize=8, n_slices=1)
    label_text(ax, 14.55, 4.5, 'Output', color='#C0504D', fontsize=8)

    ax.set_title('XGBoost + PSO 粒子群调参  结构示意图', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('paper_figures/arch2_xgb.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ arch2_xgb.png')


if __name__ == '__main__':
    draw_rf()
    draw_gbr()
    draw_lgbm()
    draw_xgb()
    print('\n全部生成完成')
