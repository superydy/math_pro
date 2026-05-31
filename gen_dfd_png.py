#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Q2 / Q3 数据流程图 PNG（Visio 风格）
"""
import os
os.chdir('/home/user/math_pro')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager
import numpy as np

fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']        = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

# ── 颜色 ─────────────────────────────────────────────────────────────────────
C_ENTITY = '#F5F5F5'   # 外部实体  灰白
C_PROC   = '#DAE8FC'   # 过程      浅蓝
C_PROC_H = '#2980B9'   # 过程标题栏
C_STORE  = '#FFF2CC'   # 数据存储  浅黄
C_BORDER = '#333333'
C_BLUE   = '#2980B9'
C_GOLD   = '#D6B656'
C_ARROW  = '#555555'

def draw_entity(ax, cx, cy, w, h, text, fontsize=9):
    """外部实体：实心边框矩形"""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="square,pad=0",
                          facecolor=C_ENTITY, edgecolor=C_BORDER, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=4, wrap=True,
            multialignment='center')

def draw_process(ax, cx, cy, w, h, num, text, fontsize=8.5):
    """过程：圆角矩形，左上角编号色块"""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=C_PROC, edgecolor=C_BLUE, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    # 编号小块（左上）
    nb_w, nb_h = 0.28, h
    nb = FancyBboxPatch((cx - w/2, cy - h/2), nb_w, nb_h,
                        boxstyle="round,pad=0.01",
                        facecolor=C_PROC_H, edgecolor=C_BLUE, linewidth=0, zorder=4)
    ax.add_patch(nb)
    ax.text(cx - w/2 + nb_w/2, cy, num, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=5)
    ax.text(cx - w/2 + nb_w + (w - nb_w)/2, cy, text,
            ha='center', va='center', fontsize=fontsize, zorder=4,
            multialignment='center')

def draw_store(ax, cx, cy, w, h, num, text, fontsize=8.5):
    """数据存储：上下双线，无左右边框（Gane-Sarson风格）"""
    # 填充背景
    rect = mpatches.Rectangle((cx - w/2, cy - h/2), w, h,
                               facecolor=C_STORE, edgecolor='none', zorder=3)
    ax.add_patch(rect)
    # 上线（双线）
    y_top = cy + h/2
    ax.plot([cx - w/2, cx + w/2], [y_top,        y_top       ], color=C_GOLD, lw=1.8, zorder=4)
    ax.plot([cx - w/2, cx + w/2], [y_top - 0.055, y_top - 0.055], color=C_GOLD, lw=0.8, zorder=4)
    # 下线（双线）
    y_bot = cy - h/2
    ax.plot([cx - w/2, cx + w/2], [y_bot,        y_bot       ], color=C_GOLD, lw=1.8, zorder=4)
    ax.plot([cx - w/2, cx + w/2], [y_bot + 0.055, y_bot + 0.055], color=C_GOLD, lw=0.8, zorder=4)
    # 编号左侧
    nb_w = 0.25
    ax.text(cx - w/2 + nb_w/2, cy, num, ha='center', va='center',
            fontsize=8, color=C_GOLD, fontweight='bold', zorder=5)
    ax.plot([cx - w/2 + nb_w, cx - w/2 + nb_w], [y_bot, y_top], color=C_GOLD, lw=1, zorder=4)
    ax.text(cx - w/2 + nb_w + (w - nb_w)/2, cy, text,
            ha='center', va='center', fontsize=fontsize, zorder=4,
            multialignment='center')

def arrow(ax, x0, y0, x1, y1, label='', lw=1.4, color=C_ARROW,
          label_side='top', waypoints=None):
    """带标签的折线箭头（支持中间转折点）"""
    if waypoints:
        pts = [(x0, y0)] + waypoints + [(x1, y1)]
    else:
        pts = [(x0, y0), (x1, y1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'),
                zorder=5)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=5)

    if label:
        # 标签放在第一段中点附近
        mx = (xs[0] + xs[1]) / 2
        my = (ys[0] + ys[1]) / 2
        dy = 0.07 if label_side == 'top' else -0.07
        dx = 0.0
        if xs[0] == xs[1]:   # 竖线
            dx = 0.08; dy = 0.0
        ax.text(mx + dx, my + dy, label, ha='center', va='center',
                fontsize=7.5, color='#333333', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

def polyarrow(ax, pts, label='', lw=1.4, color=C_ARROW, label_idx=0):
    """多折点箭头"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # 画中间线段
    for i in range(len(pts) - 2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, lw=lw, zorder=5)
    # 最后一段用箭头
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=5)
    if label:
        i = label_idx
        mx = (xs[i] + xs[i+1]) / 2
        my = (ys[i] + ys[i+1]) / 2
        ax.text(mx, my + 0.07, label, ha='center', va='center',
                fontsize=7.5, color='#333333', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

# ═══════════════════════════════════════════════════════════════════════════
# Q2 DFD
# ═══════════════════════════════════════════════════════════════════════════
def gen_q2():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    fig.patch.set_facecolor('white')

    # 标题
    ax.text(8, 9.6, '问题二：CO浓度预测系统  数据流程图 (DFD)',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(8, 9.25, '（Gane-Sarson 风格）',
            ha='center', va='center', fontsize=9, color='#555555')

    # ── 外部实体 ──
    #  E1 传感器  左上
    draw_entity(ax, 1.1, 8.0, 1.5, 0.65, '传感器\n现场采集', fontsize=8.5)
    # E2 历史CO  左下
    draw_entity(ax, 1.1, 2.2, 1.5, 0.65, 'CO 历史\n时序数据', fontsize=8.5)
    # E3 调度系统  右中
    draw_entity(ax, 14.9, 5.0, 1.6, 0.65, '调度系统\n报警平台', fontsize=8.5)

    # ── 过程 ──
    # P1 数据预处理  (3.5, 8.0)
    draw_process(ax, 3.5, 8.0, 2.6, 0.75, '1.0',
                 '数据预处理\n(异常检测/剔除/归一化)', fontsize=8)
    # P2 特征工程  (6.5, 8.0)
    draw_process(ax, 6.8, 8.0, 2.8, 0.75, '2.0',
                 '特征工程\n(滞后/梯度/统计/CO自回归)', fontsize=8)
    # P3 PSO优化  (10.2, 8.0)
    draw_process(ax, 10.2, 8.0, 2.6, 0.75, '3.0',
                 'PSO 超参数优化\n(8粒子 × 10次迭代)', fontsize=8)
    # P4 模型训练  (10.2, 5.8)
    draw_process(ax, 10.2, 5.8, 2.6, 0.75, '4.0',
                 'XGBoost 模型训练\n(TimeSeriesSplit 3折)', fontsize=8)
    # P5 在线预测  (13.0, 5.0)
    draw_process(ax, 13.1, 5.0, 2.4, 0.75, '5.0',
                 'CO 浓度\n在线预测', fontsize=8)

    # ── 数据存储 ──
    # D1 原始时序
    draw_store(ax, 3.5, 6.5, 2.4, 0.55, 'D1', '原始时序数据库')
    # D2 最优滞后量
    draw_store(ax, 6.8, 6.5, 2.6, 0.55, 'D2', 'Q1 最优滞后量')
    # D3 特征矩阵
    draw_store(ax, 6.8, 4.8, 2.6, 0.55, 'D3', '特征矩阵 (84 维)')
    # D4 训练/测试集
    draw_store(ax, 10.2, 4.3, 2.6, 0.55, 'D4', '训练集 / 测试集')
    # D5 最优超参数
    draw_store(ax, 10.2, 7.0, 2.6, 0.55, 'D5', '最优超参数')

    # ── 箭头 ──
    # E1 → P1
    arrow(ax, 1.85, 8.0, 2.2, 8.0, '原始传感器数据')
    # P1 → D1
    polyarrow(ax, [(3.5, 7.625), (3.5, 6.775)], '清洗后时序数据')
    # D1 → P2  (经D2滞后)
    polyarrow(ax, [(4.7, 6.5), (6.8, 6.5)], 'D1 时序')
    # D2 → P2
    polyarrow(ax, [(6.8, 6.775), (6.8, 7.625)], '滞后参数')
    # P2 → D3
    polyarrow(ax, [(6.8, 7.625), (6.8, 5.075)], '对齐后特征')
    # E2 → D3
    polyarrow(ax, [(1.1, 2.525), (1.1, 4.8), (5.5, 4.8)], 'CO 历史序列', label_idx=1)
    # D3 → P3  (用于PSO CV)
    polyarrow(ax, [(8.1, 4.8), (10.2, 4.8), (10.2, 7.625)], '特征向量', label_idx=0)
    # D3 → D4
    polyarrow(ax, [(8.1, 4.8), (10.2, 4.8), (10.2, 4.575)], '70/30划分')
    # P3 → D5
    polyarrow(ax, [(10.2, 7.625), (10.2, 7.275)], 'CV最优参数')
    # D5 → P4
    polyarrow(ax, [(10.2, 6.975), (10.2, 6.175)], '超参数配置')
    # D4 → P4
    polyarrow(ax, [(10.2, 4.575), (10.2, 5.425)], '训练样本')
    # P4 → P5
    polyarrow(ax, [(11.5, 5.8), (13.1, 5.8), (13.1, 5.375)], 'XGBoost 模型')
    # P2 → P5 (实时特征)
    polyarrow(ax, [(8.2, 8.0), (13.1, 8.0), (13.1, 5.375)], '实时特征向量', label_idx=0)
    # P5 → E3
    arrow(ax, 14.3, 5.0, 14.1, 5.0, 'CO 预测值 (ppm)')

    # 图例
    lx, ly = 0.3, 1.5
    draw_entity(ax, lx+0.45, ly, 0.8, 0.4, '外部实体', fontsize=7)
    draw_process(ax, lx+0.45+1.3, ly, 1.3, 0.4, 'n', '过程', fontsize=7)
    draw_store(ax, lx+0.45+3.0, ly, 1.3, 0.4, 'Dn', '数据存储', fontsize=7)
    ax.annotate('', xy=(lx+5.3, ly), xytext=(lx+4.6, ly),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2), zorder=5)
    ax.text(lx+5.5, ly, '数据流', va='center', fontsize=7)
    ax.text(lx, ly - 0.35, '图例', fontsize=7.5, color='#555555')

    plt.tight_layout(pad=0.5)
    plt.savefig('paper_figures/Q2_DFD.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✓ paper_figures/Q2_DFD.png')


# ═══════════════════════════════════════════════════════════════════════════
# Q3 DFD
# ═══════════════════════════════════════════════════════════════════════════
def gen_q3():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    fig.patch.set_facecolor('white')

    ax.text(8, 9.6, '问题三：风箱负压优化系统  数据流程图 (DFD)',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(8, 9.25, '（Gane-Sarson 风格）',
            ha='center', va='center', fontsize=9, color='#555555')

    # ── 外部实体 ──
    draw_entity(ax, 1.0, 8.2, 1.5, 0.65, '实时工况\n监测系统', fontsize=8.5)
    draw_entity(ax, 1.0, 2.5, 1.5, 0.65, '设备约束\n规格参数', fontsize=8.5)
    draw_entity(ax, 15.0, 5.2, 1.5, 0.65, '控制执行\n系统 / DCS', fontsize=8.5)

    # ── 过程 ──
    draw_process(ax, 3.5, 8.2, 2.6, 0.75, '1.0',
                 '状态变量采集\n(负压/温度/机速)', fontsize=8)
    draw_process(ax, 6.8, 8.2, 2.8, 0.75, '2.0',
                 '特征构造\n(与Q2特征体系一致)', fontsize=8)
    draw_process(ax, 10.0, 8.2, 2.6, 0.75, '3.0',
                 'CO 自回归\n不动点初始化', fontsize=8)
    draw_process(ax, 10.0, 6.0, 2.8, 0.75, '4.0',
                 'PSO 优化搜索\n(最小化 CO 预测值)', fontsize=8)
    draw_process(ax, 6.8, 4.0, 2.6, 0.75, '5.0',
                 '约束可行性\n校验', fontsize=8)
    draw_process(ax, 10.0, 4.0, 2.6, 0.75, '6.0',
                 '最优负压\n方案输出', fontsize=8)

    # ── 数据存储 ──
    draw_store(ax, 3.5, 6.6, 2.4, 0.55, 'D1', '当前工况快照')
    draw_store(ax, 6.8, 6.6, 2.6, 0.55, 'D2', '84维特征向量')
    draw_store(ax, 10.0, 7.0, 2.6, 0.55, 'D3', 'Q2 XGBoost 模型')
    draw_store(ax, 10.0, 5.0, 2.6, 0.55, 'D4', '粒子群状态')
    draw_store(ax, 6.8, 2.5, 2.6, 0.55, 'D5', '约束边界 (18维)')
    draw_store(ax, 13.0, 4.0, 2.4, 0.55, 'D6', '最优负压方案')

    # ── 箭头 ──
    # E1 → P1
    arrow(ax, 1.75, 8.2, 2.2, 8.2, '实时传感数据')
    # P1 → D1
    polyarrow(ax, [(3.5, 7.825), (3.5, 6.875)], '状态采样')
    # D1 → P2
    polyarrow(ax, [(4.7, 6.6), (5.4, 6.6), (6.8, 6.6)], '状态向量')
    # P2 → D2
    polyarrow(ax, [(6.8, 7.825), (6.8, 6.875)], '特征向量')
    # D2 → P3
    polyarrow(ax, [(8.1, 6.6), (10.0, 6.6), (10.0, 7.825)], 'CO相关特征')
    # D3 → P3
    polyarrow(ax, [(10.0, 7.275), (10.0, 7.825)], '模型参数')
    # P3 → P4
    polyarrow(ax, [(10.0, 7.825), (10.0, 6.375)], 'CO初始估计')
    # D2 → P4
    polyarrow(ax, [(8.1, 6.6), (10.0, 6.6)], '特征向量')
    # D3 → P4
    polyarrow(ax, [(10.0, 7.275), (10.0, 6.375)], '')
    # P4 → D4
    polyarrow(ax, [(10.0, 5.625), (10.0, 5.275)], '粒子位置/速度')
    # D4 → P4 (迭代回路)
    polyarrow(ax, [(11.3, 5.0), (12.5, 5.0), (12.5, 6.0), (11.4, 6.0)],
              '迭代更新', label_idx=1)
    # P4 → P5
    polyarrow(ax, [(8.6, 6.0), (6.8, 6.0), (6.8, 4.375)], 'PSO候选方案')
    # E2 → D5
    polyarrow(ax, [(1.75, 2.5), (5.5, 2.5)], '设备约束')
    # D5 → P5
    polyarrow(ax, [(6.8, 2.775), (6.8, 3.625)], '约束边界')
    # P5 → P6 (可行 → 输出)
    polyarrow(ax, [(8.1, 4.0), (8.7, 4.0), (10.0, 4.0)], '可行方案')
    # P5 → P4 (不可行 → 惩罚回路)
    polyarrow(ax, [(6.8, 4.375), (6.8, 3.2), (9.2, 3.2), (9.2, 5.8), (8.6, 5.8)],
              '惩罚/重采样', label_idx=2)
    # P6 → D6
    polyarrow(ax, [(11.3, 4.0), (11.8, 4.0)], '最优解')
    # D6 → E3
    polyarrow(ax, [(14.2, 4.0), (15.0, 4.0), (15.0, 4.875)], '负压指令 (18维)')

    # 图例
    lx, ly = 0.3, 1.3
    draw_entity(ax, lx+0.45, ly, 0.8, 0.4, '外部实体', fontsize=7)
    draw_process(ax, lx+0.45+1.3, ly, 1.3, 0.4, 'n', '过程', fontsize=7)
    draw_store(ax, lx+0.45+3.0, ly, 1.3, 0.4, 'Dn', '数据存储', fontsize=7)
    ax.annotate('', xy=(lx+5.3, ly), xytext=(lx+4.6, ly),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2), zorder=5)
    ax.text(lx+5.5, ly, '数据流', va='center', fontsize=7)
    ax.text(lx, ly - 0.35, '图例', fontsize=7.5, color='#555555')

    plt.tight_layout(pad=0.5)
    plt.savefig('paper_figures/Q3_DFD.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✓ paper_figures/Q3_DFD.png')


if __name__ == '__main__':
    gen_q2()
    gen_q3()
    print('\n全部完成。')
