#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os
os.chdir('/home/user/math_pro')

from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('paper_figures', exist_ok=True)

# ── 通用工具 ─────────────────────────────────────────────────────────────────
def node(ax, x, y, w, h, text, fc='#EAF4FB', ec='#2980B9', fs=8.5, radius=0.04, bold=False):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=4, wrap=True,
            multialignment='center')

def arr(ax, x1, y1, x2, y2, color='#555', lw=1.5, style='->', head=8):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=head))

def tri_node(ax, cx, cy, size, fc='#D5E8D4', ec='#82B366'):
    """小三角形代表决策树"""
    pts = np.array([[cx, cy+size], [cx-size*0.75, cy-size*0.5],
                    [cx+size*0.75, cy-size*0.5]])
    tri = plt.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(tri)

# ═══════════════════════════════════════════════════════════════════════════
# 图1：随机森林
# ═══════════════════════════════════════════════════════════════════════════
def draw_rf(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis('off')
    ax.set_title('随机森林 (Random Forest)', fontsize=12, fontweight='bold', pad=8)

    # 输入
    node(ax, 5, 8, 3.5, 0.65, '输入特征  X (n×84)', fc='#FFF3CD', ec='#D4A017', fs=9, bold=True)

    # Bootstrap采样
    for i, (xc, label) in enumerate([(1.8,1),(5,2),(8.2,3)]):
        # 虚线框表示Bootstrap样本
        rect = FancyBboxPatch((xc-1.3, 6.15), 2.6, 0.75,
                               boxstyle="round,pad=0.05",
                               facecolor='#F0F8FF', edgecolor='#7B9EC7',
                               linewidth=1.1, linestyle='--', zorder=2)
        ax.add_patch(rect)
        ax.text(xc, 6.52, f'Bootstrap\n样本 {label}', ha='center', va='center',
                fontsize=7.5, color='#2C5F8A')
        arr(ax, 5, 7.68, xc, 6.9, color='#888')

    # 三棵树
    tree_x = [1.8, 5.0, 8.2]
    tree_colors = [('#D5E8D4','#82B366'), ('#DAE8FC','#6C8EBF'), ('#F8CECC','#B85450')]
    for i, (xc, (fc, ec)) in enumerate(zip(tree_x, tree_colors)):
        tri_node(ax, xc, 5.2, 0.55, fc, ec)
        # 子节点
        tri_node(ax, xc-0.55, 4.25, 0.38, fc, ec)
        tri_node(ax, xc+0.55, 4.25, 0.38, fc, ec)
        ax.plot([xc, xc-0.55], [4.65, 4.63], '-', color=ec, lw=1.2, zorder=2)
        ax.plot([xc, xc+0.55], [4.65, 4.63], '-', color=ec, lw=1.2, zorder=2)
        ax.text(xc, 5.2, f'树 {i+1}', ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='#333', zorder=5)
        # 叶节点预测
        node(ax, xc, 3.3, 1.4, 0.52, f'y^_{i+1}',
             fc='#E8F5E9', ec='#4CAF50', fs=8.5)
        arr(ax, xc, 3.88, xc, 3.57, color='#4CAF50')

    # 随机特征子集标注
    ax.text(5, 4.85, '随机特征子集\n(max_features=√84)', ha='center', va='center',
            fontsize=7, color='#777', style='italic')

    # 平均聚合
    node(ax, 5, 2.35, 3.8, 0.62, '平均聚合  y^ = mean(y^1, y^2, y^3)',
         fc='#E3F2FD', ec='#1565C0', fs=8.5, bold=True)
    for xc in tree_x:
        arr(ax, xc, 3.04, 5, 2.66, color='#1565C0')

    # 输出
    node(ax, 5, 1.35, 2.8, 0.58, '预测输出  y^_CO',
         fc='#FCE4EC', ec='#C62828', fs=9, bold=True)
    arr(ax, 5, 2.04, 5, 1.64)

    # 特点标注
    ax.text(0.3, 0.5,
            '● Bagging：有放回Bootstrap采样\n'
            '● 各树独立并行训练\n'
            '● 随机特征子集降低相关性',
            fontsize=7.2, va='bottom', color='#444',
            bbox=dict(fc='#FAFAFA', ec='#CCC', pad=4, boxstyle='round'))


# ═══════════════════════════════════════════════════════════════════════════
# 图2：梯度提升树 GBR
# ═══════════════════════════════════════════════════════════════════════════
def draw_gbr(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis('off')
    ax.set_title('梯度提升树 (GBR)', fontsize=12, fontweight='bold', pad=8)

    # 输入
    node(ax, 5, 8, 3.5, 0.65, '输入特征  X', fc='#FFF3CD', ec='#D4A017', fs=9, bold=True)

    # 初始预测
    node(ax, 5, 7.0, 3.0, 0.58, 'F0(x) = mean(y)  初始预测',
         fc='#F3E5F5', ec='#7B1FA2', fs=8.2)
    arr(ax, 5, 7.68, 5, 7.29)

    # 三轮串行迭代
    colors = [('#E8F5E9','#2E7D32'), ('#E3F2FD','#1565C0'), ('#FFF8E1','#F57F17')]
    labels = ['t=1', 't=2', 't=T']
    res_y  = [6.1, 4.8, 3.5]
    tree_y = [5.55, 4.25, 2.95]

    for i, (yres, ytree, (fc,ec), lbl) in enumerate(zip(res_y, tree_y, colors, labels)):
        # 残差
        node(ax, 2.2, yres, 2.4, 0.5, f'残差\nrt=y−Ft₋1(x)',
             fc='#FFF3E0', ec='#E65100', fs=7.5)
        # 弱学习器
        tri_node(ax, 5.2, ytree+0.1, 0.42, fc, ec)
        ax.text(5.2, ytree+0.1, lbl, ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=5)
        # 更新
        node(ax, 8.0, yres, 2.5, 0.5, f'Ft=Ft₋1+η·ht\n更新预测',
             fc='#E8EAF6', ec='#3949AB', fs=7.5)
        # 箭头
        if i > 0:
            arr(ax, 5, res_y[i-1]-0.25, 2.2, yres+0.25, color='#E65100')
        arr(ax, 3.4, yres, 4.77, ytree+0.1, color=ec)
        arr(ax, 5.63, ytree+0.1, 6.75, yres, color=ec)
        arr(ax, 2.2, yres-0.25, 2.2, yres-0.6 if i<2 else yres, color='#E65100')

    # 省略号
    ax.text(5.2, 2.35, '...', ha='center', fontsize=14, color='#888')

    # 最终输出
    node(ax, 5, 1.55, 3.8, 0.6, 'F_T(x) = F0 + η·Σht  最终预测',
         fc='#FCE4EC', ec='#C62828', fs=8.5, bold=True)
    arr(ax, 5, 2.75, 5, 1.85, color='#C62828')

    ax.text(0.2, 0.3,
            '● Boosting：串行迭代，逐步纠错\n'
            '● 每棵树拟合前一轮残差\n'
            '● η为学习率，控制步长',
            fontsize=7.2, va='bottom', color='#444',
            bbox=dict(fc='#FAFAFA', ec='#CCC', pad=4, boxstyle='round'))


# ═══════════════════════════════════════════════════════════════════════════
# 图3：LightGBM
# ═══════════════════════════════════════════════════════════════════════════
def draw_lgbm(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis('off')
    ax.set_title('LightGBM', fontsize=12, fontweight='bold', pad=8)

    node(ax, 5, 8.1, 3.5, 0.65, '输入特征  X (84维)', fc='#FFF3CD', ec='#D4A017', fs=9, bold=True)

    # 直方图特征桶
    node(ax, 5, 7.1, 5.5, 0.62,
         '直方图分箱  (Histogram-based Binning，255箱)',
         fc='#E8EAF6', ec='#3949AB', fs=8)
    arr(ax, 5, 7.78, 5, 7.42)

    # Leaf-wise vs Level-wise 对比
    ax.text(2.5, 6.6, 'Level-wise（传统）', ha='center', fontsize=8,
            color='#888', style='italic')
    ax.text(7.5, 6.6, 'Leaf-wise（LightGBM）', ha='center', fontsize=8,
            color='#1B5E20', fontweight='bold')

    # Level-wise示意（灰色，较小）
    lw_nodes = [(2.5,6.15), (1.5,5.4),(3.5,5.4),
                (1.0,4.65),(2.0,4.65),(3.0,4.65),(4.0,4.65)]
    lw_sizes = [0.55, 0.42, 0.42, 0.3,0.3,0.3,0.3]
    for (x,y), s in zip(lw_nodes, lw_sizes):
        c = plt.Circle((x,y), s*0.5, fc='#EEEEEE', ec='#AAAAAA', lw=1, zorder=3)
        ax.add_patch(c)
    for parent, children in [(0,[1,2]),(1,[3,4]),(2,[5,6])]:
        px,py = lw_nodes[parent]; ps = lw_sizes[parent]
        for ci in children:
            cx,cy = lw_nodes[ci]
            ax.plot([px,cx],[py-ps*0.5,cy+lw_sizes[ci]*0.5],'-',color='#BBB',lw=1,zorder=2)

    # Leaf-wise示意（彩色，突出最大增益叶节点）
    rw_nodes = [(7.5,6.1),(6.3,5.35),(8.7,5.35),
                (5.6,4.6),(7.0,4.6)]
    rw_colors= ['#A5D6A7','#A5D6A7','#FF8A65','#A5D6A7','#FF8A65']
    rw_ec    = ['#2E7D32','#2E7D32','#BF360C','#2E7D32','#BF360C']
    rw_sizes = [0.55,0.42,0.5,0.35,0.45]
    for (x,y),s,fc2,ec2 in zip(rw_nodes,rw_sizes,rw_colors,rw_ec):
        c = plt.Circle((x,y), s*0.5, fc=fc2, ec=ec2, lw=1.5, zorder=3)
        ax.add_patch(c)
    for parent,children in [(0,[1,2]),(1,[3,4])]:
        px,py=rw_nodes[parent]; ps=rw_sizes[parent]
        for ci in children:
            cx,cy=rw_nodes[ci]
            ax.plot([px,cx],[py-ps*0.5,cy+rw_sizes[ci]*0.5],'-',color='#777',lw=1.2,zorder=2)
    ax.text(8.7,5.35,'最大\n增益', ha='center', va='center', fontsize=6.5,
            color='white', fontweight='bold', zorder=5)
    ax.text(7.0,4.6,'继续\n分裂', ha='center', va='center', fontsize=6.5,
            color='white', fontweight='bold', zorder=5)

    arr(ax, 5, 6.79, 2.5, 6.42, color='#AAA')
    arr(ax, 5, 6.79, 7.5, 6.42, color='#2E7D32')

    # GOSS + EFB
    node(ax, 2.5, 3.75, 3.5, 0.58,
         'GOSS：梯度采样\n大梯度全保留+小梯度随机',
         fc='#E3F2FD', ec='#1565C0', fs=7.5)
    node(ax, 7.5, 3.75, 3.5, 0.58,
         'EFB：互斥特征捆绑\n稀疏特征合并降维',
         fc='#F3E5F5', ec='#7B1FA2', fs=7.5)

    # 输出
    node(ax, 5, 2.7, 3.5, 0.58, '集成 T 棵树输出  y^_CO',
         fc='#FCE4EC', ec='#C62828', fs=9, bold=True)
    arr(ax, 2.5, 3.46, 5, 2.99, color='#1565C0')
    arr(ax, 7.5, 3.46, 5, 2.99, color='#7B1FA2')

    ax.text(0.2, 1.6,
            '● Leaf-wise：每次分裂增益最大的叶，深度更深\n'
            '● 直方图：O(#bins) 而非 O(n)，大幅提速\n'
            '● GOSS + EFB 进一步减少样本和特征计算量',
            fontsize=7.2, va='bottom', color='#444',
            bbox=dict(fc='#FAFAFA', ec='#CCC', pad=4, boxstyle='round'))


# ═══════════════════════════════════════════════════════════════════════════
# 图4：XGBoost
# ═══════════════════════════════════════════════════════════════════════════
def draw_xgb(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis('off')
    ax.set_title('XGBoost + PSO 调参', fontsize=12, fontweight='bold', pad=8)

    node(ax, 5, 8.1, 3.5, 0.65, '输入特征  X (84维)', fc='#FFF3CD', ec='#D4A017', fs=9, bold=True)

    # 目标函数
    node(ax, 5, 7.1, 6.2, 0.62,
         'Obj = Σ l(yᵢ, y^ᵢ) + Σ Ω(ft)   正则化目标',
         fc='#E8EAF6', ec='#3949AB', fs=8, bold=True)
    arr(ax, 5, 7.78, 5, 7.42)

    # 二阶泰勒展开
    node(ax, 5, 6.15, 6.5, 0.62,
         'Taylor展开：Obj ≈ Σ[gift(xᵢ) + ½hift²(xᵢ)] + Ω(ft)',
         fc='#E3F2FD', ec='#1565C0', fs=7.8)
    arr(ax, 5, 6.79, 5, 6.46)

    # Level-wise 建树
    lv_y = 5.25
    nodes_xgb = [(5,lv_y),(3.3,lv_y-0.75),(6.7,lv_y-0.75),
                 (2.3,lv_y-1.5),(4.3,lv_y-1.5),(5.7,lv_y-1.5),(7.7,lv_y-1.5)]
    sizes_xgb  = [0.48, 0.38, 0.38, 0.3, 0.3, 0.3, 0.3]
    fc_xgb     = ['#FFCCBC','#FFCCBC','#FFCCBC','#B2EBF2','#B2EBF2','#B2EBF2','#B2EBF2']
    ec_xgb     = ['#BF360C']*3 + ['#006064']*4
    for (x,y),s,fc2,ec2 in zip(nodes_xgb,sizes_xgb,fc_xgb,ec_xgb):
        c = plt.Circle((x,y), s*0.5, fc=fc2, ec=ec2, lw=1.4, zorder=3)
        ax.add_patch(c)
    for parent,children in [(0,[1,2]),(1,[3,4]),(2,[5,6])]:
        px,py=nodes_xgb[parent]; ps=sizes_xgb[parent]
        for ci in children:
            cx,cy=nodes_xgb[ci]
            ax.plot([px,cx],[py-ps*0.5,cy+sizes_xgb[ci]*0.5],'-',color='#888',lw=1.2,zorder=2)
    ax.text(5, lv_y, 'Level-\nwise', ha='center', va='center',
            fontsize=6.5, fontweight='bold', zorder=5)
    ax.text(9.5, lv_y, '逐层建树\n深度可控', ha='center', va='center',
            fontsize=7, color='#BF360C')
    arr(ax, 5, 5.84, 5, 5.49, color='#BF360C')

    # 正则项
    node(ax, 2.5, 2.95, 3.8, 0.58,
         'Ω(f) = γT + ½λ‖w‖²\nL1(α)+L2(λ) 正则防过拟合',
         fc='#F3E5F5', ec='#7B1FA2', fs=7.5)

    # PSO 调参框
    node(ax, 7.5, 2.95, 3.6, 0.95,
         'PSO 粒子群优化\n8粒子×10迭代\n搜索7个超参数',
         fc='#FFF9C4', ec='#F9A825', fs=7.8, bold=True)
    # PSO 箭头（双向）
    ax.annotate('', xy=(6.5, 3.1), xytext=(7.3, 3.1),
                arrowprops=dict(arrowstyle='<->', color='#F9A825', lw=1.8, mutation_scale=12))

    arr(ax, 5, 3.5, 2.5, 3.24, color='#7B1FA2')
    arr(ax, 5, 3.5, 7.5, 3.44, color='#F9A825')

    # 集成输出
    node(ax, 5, 1.8, 4.5, 0.62,
         'y^ = Σ ft(x)   T棵树集成输出',
         fc='#FCE4EC', ec='#C62828', fs=8.5, bold=True)
    arr(ax, 2.5, 2.66, 5, 2.12, color='#C62828')
    arr(ax, 5, 2.66, 5, 2.12, color='#C62828')

    node(ax, 5, 0.88, 2.8, 0.58, '预测 CO 浓度  y^_CO',
         fc='#FCE4EC', ec='#B71C1C', fs=9, bold=True)
    arr(ax, 5, 1.49, 5, 1.18, color='#B71C1C')

    ax.text(0.1, 0.05,
            '● 二阶梯度：利用一、二阶导数更精确\n'
            '● 正则化：γ剪枝+L2权重惩罚\n'
            '● PSO自动搜索最优超参数组合',
            fontsize=7.2, va='bottom', color='#444',
            bbox=dict(fc='#FAFAFA', ec='#CCC', pad=4, boxstyle='round'))


# ═══════════════════════════════════════════════════════════════════════════
# 生成四张独立图
# ═══════════════════════════════════════════════════════════════════════════
configs = [
    ('rf',   draw_rf,   '随机森林'),
    ('gbr',  draw_gbr,  '梯度提升树GBR'),
    ('lgbm', draw_lgbm, 'LightGBM'),
    ('xgb',  draw_xgb,  'XGBoost+PSO'),
]

paths = []
for tag, func, title in configs:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor('white')
    func(ax)
    plt.tight_layout(pad=0.5)
    path = f'paper_figures/arch_{tag}.png'
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    paths.append(path)
    print(f'✓ {path}')

# 也生成一张4合1总览图
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('white')
for ax, (tag, func, title) in zip(axes.flat, configs):
    func(ax)
plt.tight_layout(pad=1.5)
combined = 'paper_figures/arch_all4.png'
plt.savefig(combined, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ {combined}')
