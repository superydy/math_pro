#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSO 算法架构图 —— 仿参考图风格
  云朵形 = 主模型  |  椭圆 = 辅助模型  |  高矩形 = 特征列表
  粗蓝箭头 = 主流  |  细金箭头 = 数据连接
"""
import os
os.chdir('/home/user/math_pro')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.font_manager import fontManager
import numpy as np

fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']        = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

# ── 调色板（与参考图一致）─────────────────────────────────────────────────────
RED    = '#C0392B'    # 云朵主模型
TEAL   = '#148F77'    # 椭圆辅模型
CYAN   = '#1A8F7A'    # 特征列表背景
ORANGE = '#E67E22'    # 处理步骤
PURPLE = '#7D3C98'    # 注释框
GRAY   = '#808B96'    # 灰注释框
GOLD   = '#D4AC0D'    # 输出/存储
NAVY   = '#1C2833'    # 粗箭头
BEIGE  = '#C8A951'    # 细箭头
WHITE  = '#FFFFFF'
LBLUE  = '#D6EAF8'    # 表格背景

# ── 基础绘图函数 ───────────────────────────────────────────────────────────────
def cloud(ax, cx, cy, w, h, fc, text, fs=11, tc=WHITE, zorder=6, italic=False):
    """云朵形（主椭圆 + 顶部凸包）"""
    # 主体
    ax.add_patch(Ellipse((cx, cy-h*0.04), w*0.92, h*0.72,
                         fc=fc, ec='none', zorder=zorder))
    # 顶部凸包（4个小椭圆）
    for dx, dy_, bw, bh in [
        (-0.30, +0.30, 0.30, 0.38),
        (-0.07, +0.42, 0.32, 0.42),
        (+0.16, +0.40, 0.28, 0.40),
        (+0.35, +0.26, 0.24, 0.33),
    ]:
        ax.add_patch(Ellipse((cx+dx*w, cy+dy_*h), bw*w, bh*h,
                             fc=fc, ec='none', zorder=zorder+1))
    # 两侧凸包
    for dx, bw in [(-0.50, 0.20), (+0.50, 0.20)]:
        ax.add_patch(Ellipse((cx+dx*w, cy+0.02*h), bw*w, 0.38*h,
                             fc=fc, ec='none', zorder=zorder))
    style = 'italic' if italic else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', fontstyle=style,
            zorder=zorder+3, multialignment='center')


def oval(ax, cx, cy, w, h, fc, text, fs=11, tc=WHITE, zorder=5):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec='none', zorder=zorder))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', zorder=zorder+1,
            multialignment='center')


def feat_box(ax, cx, cy, w, h, title, items, bg=CYAN, zorder=4):
    """高矩形特征列表（仿参考图左侧青色列表框）"""
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                boxstyle='round,pad=0.05',
                                fc=bg, ec='none', zorder=zorder))
    ax.text(cx, cy+h/2-0.18, title, ha='center', va='center',
            fontsize=8, color=WHITE, fontweight='bold', zorder=zorder+1)
    item_h = (h-0.35) / len(items)
    for i, it in enumerate(items):
        iy = cy + h/2 - 0.35 - item_h*(i+0.5)
        ax.add_patch(FancyBboxPatch((cx-w/2+0.08, iy-item_h/2+0.04),
                                   w-0.16, item_h-0.08,
                                   boxstyle='round,pad=0.03',
                                   fc=WHITE, ec='none', zorder=zorder+1))
        ax.text(cx, iy, it, ha='center', va='center',
                fontsize=8.5, color='#1C2833', zorder=zorder+2)


def proc_box(ax, cx, cy, w, h, text, fc=ORANGE, fs=9, zorder=5):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                boxstyle='round,pad=0.05',
                                fc=fc, ec='none', zorder=zorder))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=WHITE, fontweight='bold', zorder=zorder+1,
            multialignment='center')


def annot_box(ax, cx, cy, w, h, text, fc=GRAY, fs=8, zorder=4):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                boxstyle='round,pad=0.06',
                                fc=fc, ec='none', alpha=0.88, zorder=zorder))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=WHITE, zorder=zorder+1, multialignment='center')


def table_box(ax, cx, cy, rows, fc=LBLUE, fs=8.5, zorder=3):
    """小表格样式的输入框"""
    col_w, row_h = 0.55, 0.28
    n_r, n_c = len(rows), len(rows[0]) if rows else 1
    tw = n_c * col_w + 0.1
    th = n_r * row_h + 0.1
    ax.add_patch(FancyBboxPatch((cx-tw/2, cy-th/2), tw, th,
                                boxstyle='round,pad=0.04',
                                fc=fc, ec='#2980B9', lw=1.2, zorder=zorder))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            tx = cx - tw/2 + 0.05 + col_w*(c+0.5)
            ty = cy + th/2 - 0.05 - row_h*(r+0.5)
            ax.text(tx, ty, cell, ha='center', va='center',
                    fontsize=fs, color='#1C2833', zorder=zorder+1)


def thick_arr(ax, x0, y0, x1, y1, lbl='', color=NAVY, lw=3.5, fs=8.5):
    """粗箭头（主数据流）—— shrink 防止插入形状内部"""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=20,
                                shrinkA=4, shrinkB=4), zorder=7)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.05, my+0.12, lbl, ha='center', va='center',
                fontsize=fs, color=color, zorder=8,
                bbox=dict(fc='white', ec='none', pad=0.1, alpha=0.85))


def thin_arr(ax, pts, lbl='', color=BEIGE, lw=1.5, fs=7.8, li=0, ls='top'):
    """细金箭头（数据连接）—— 最后一段 shrinkB 防止箭头插入目标"""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=lw, zorder=6)
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=13,
                                shrinkA=3, shrinkB=4), zorder=6)
    if lbl:
        mx = (xs[li]+xs[li+1])/2; my = (ys[li]+ys[li+1])/2
        dy = 0.12 if ls == 'top' else -0.12; dx = 0.0
        if abs(xs[li]-xs[li+1]) < 0.01: dx = 0.14; dy = 0.0
        ax.text(mx+dx, my+dy, lbl, ha='center', va='center',
                fontsize=fs, color='#444444', zorder=8,
                bbox=dict(fc='white', ec='none', pad=0.1, alpha=0.9))


def feat_group_grid(ax, cx, cy, items, zorder=4):
    """2列紧凑特征分组小色块（替代大背景椭圆）"""
    colors = ['#2980B9','#E67E22','#8E44AD','#148F77']
    cols, w, h, gap = 2, 1.35, 0.68, 0.10
    rows = (len(items)+1)//2
    total_w = cols*w + (cols-1)*gap
    total_h = rows*h + (rows-1)*gap
    x0 = cx - total_w/2
    y0 = cy + total_h/2
    for i,(label,fc) in enumerate(zip(items, colors)):
        r, c = divmod(i, cols)
        bx = x0 + c*(w+gap) + w/2
        by = y0 - r*(h+gap) - h/2
        ax.add_patch(FancyBboxPatch((bx-w/2, by-h/2), w, h,
                     boxstyle='round,pad=0.05', fc=fc, ec='none', zorder=zorder))
        ax.text(bx, by, label, ha='center', va='center',
                fontsize=9, color=WHITE, fontweight='bold',
                zorder=zorder+1, multialignment='center')


# ══════════════════════════════════════════════════════════════════════════════
# Q2 PSO 超参数优化架构图
# ══════════════════════════════════════════════════════════════════════════════
def gen_q2():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    # ── 左：输入数据表格 ──
    ax.text(1.4, 7.2, '输入训练数据', ha='center', fontsize=10,
            fontweight='bold', color='#1C2833')
    table_box(ax, 1.4, 6.35,
              [['x1', 'x2', '...', 'x84'],
               ['p11', 'p12', '...', 'CO1'],
               ['p21', 'p22', '...', 'CO2'],
               ['..',  '..',  '...',  '..' ]])
    ax.text(1.4, 5.1, '1624×84  训练样本',
            ha='center', fontsize=9, color='#555555')

    # ── 左下：特征分组（紧凑2×2色块）──
    feat_group_grid(ax, 1.5, 3.9,
                    ['物理特征\n41维', '梯度特征\n34维',
                     '统计特征\n4维',  'CO自回归\n5维'])
    ax.text(1.5, 2.45, '特征工程（84维）',
            ha='center', fontsize=9.5, color='#555555', style='italic', fontweight='bold')

    # ── 中：特征列表高矩形 ──
    feat_box(ax, 4.6, 4.8, 1.8, 5.2,
             '特征体系 84维',
             ['机速 × 1', '负压 × 18', '温度 × 18',
              '大烟道 × 4', '压力梯度 ×17', '温度梯度 ×17',
              '统计特征 × 4', 'CO自回归 × 5'],
             bg='#148F77')

    # ── 中下：3折CV注释 ──
    annot_box(ax, 4.6, 1.6, 2.5, 0.78,
              '3折 TimeSeriesSplit CV\n（PSO内部加速评估）',
              fc=PURPLE, fs=9)

    # ── 中右：搜索空间注释 ──
    annot_box(ax, 8.0, 2.0, 2.7, 1.0,
              '7维超参数搜索空间\nlr∈[0.01,0.50]  depth∈[2,8]\nn_est∈[50,500]  sub∈[0.5,1]',
              fc=GRAY, fs=9)

    # ── 右：PSO云朵（缩小，字体放大）──
    cloud(ax, 8.2, 5.6, 2.4, 1.7, RED,
          'PSO\n超参数优化', fs=13, italic=True)
    annot_box(ax, 8.2, 4.0, 2.6, 0.76,
              'N=8粒子  T=10迭代\nw=0.8,  c1=c2=2.0（固定）',
              fc='#922B21', fs=9.5)

    # ── 右：XGBoost 椭圆 ──
    oval(ax, 11.5, 5.4, 3.4, 1.5, TEAL,
         'XGBoost\n预测模型', fs=12)

    # ── 右下：5折CV ──
    annot_box(ax, 11.5, 4.1, 3.0, 0.76,
              '5折 TimeSeriesSplit CV（最终评估）\nR²=0.8862  ±  0.101',
              fc='#117A65', fs=9)

    # ── 输出 ──
    proc_box(ax, 11.5, 2.8, 3.2, 0.82,
             'CO 浓度预测模型\n测试集 R²=0.9493   MAE=43 ppm',
             fc=GOLD, fs=9.5)
    ax.text(11.5, 2.3, '输出', ha='center', fontsize=9,
            color=GOLD, fontweight='bold')

    # ── 连线（箭头标签全部去除）──
    thick_arr(ax, 2.55, 6.4,  3.7,  6.4)
    thick_arr(ax, 2.9,  3.85, 3.7,  4.2)
    thin_arr(ax, [(5.5, 6.6), (7.05, 6.2)])
    thin_arr(ax, [(5.5, 5.2), (7.05, 5.6)])
    thin_arr(ax, [(4.6, 2.2), (4.6, 1.99)])
    thin_arr(ax, [(5.85, 1.6), (6.65, 2.0)])
    thick_arr(ax, 9.55, 5.6,  9.8,  5.4)
    thin_arr(ax, [(9.5, 4.0), (10.15, 4.0), (10.15, 4.65)])
    thin_arr(ax, [(11.5, 4.65), (11.5, 4.48)])
    thick_arr(ax, 11.5, 3.72, 11.5, 3.21)

    plt.tight_layout(pad=0.3)
    plt.savefig('paper_figures/Q2_PSO_arch.png', dpi=250,
                bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print('✓ paper_figures/Q2_PSO_arch.png')


# ══════════════════════════════════════════════════════════════════════════════
# Q3 PSO 负压优化架构图
# ══════════════════════════════════════════════════════════════════════════════
def gen_q3():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    # ── 左：当前工况表格 ──
    ax.text(1.4, 7.2, '当前工况快照', ha='center', fontsize=10,
            fontweight='bold', color='#1C2833')
    table_box(ax, 1.4, 6.35,
              [['风箱', 'p1', 'p2', 'p18'],
               ['负压', '-11', '-14', '-13'],
               ['温度', '89', '112', '145'],
               ['机速', '', '1.46 m/min', '']],
              fs=8.2)
    ax.text(1.4, 5.1, '当前 CO = 3495 ppm',
            ha='center', fontsize=9, color='#C0392B', fontweight='bold')

    # ── 左下：Q2代理模型（紧凑色块组）──
    feat_group_grid(ax, 1.5, 3.8,
                    ['Q2 XGBoost\n代理模型', '标准化\nscaler',
                     '特征工程\n参数', '滞后量\n配置'])
    ax.text(1.5, 2.45, 'Q2 训练产物（代理模型）',
            ha='center', fontsize=9.5, color='#555555', style='italic', fontweight='bold')

    # ── 中：特征构造列表 ──
    feat_box(ax, 4.6, 4.8, 1.8, 5.2,
             '特征构造 84维',
             ['机速 × 1', '负压 × 18', '温度 × 18',
              '大烟道 × 4', '压力梯度 ×17', '温度梯度 ×17',
              '统计特征 × 4', 'CO自回归 × 5'],
             bg='#1A5276')

    # ── 中下：不动点注释 ──
    annot_box(ax, 4.6, 1.6, 2.7, 0.9,
              '不动点初始化\nX(t+1) = 0.5·X_t + 0.5·f(X_t)\n收敛条件: |ΔX| < 1 ppm',
              fc=PURPLE, fs=9)

    # ── 中右：惩罚项注释 ──
    annot_box(ax, 7.9, 2.0, 2.6, 0.9,
              '可靠性惩罚\npenalty = Σ exp(-10·dist)\n目标 = CO_pred + α·penalty',
              fc='#922B21', fs=9)

    # ── 右：PSO云朵 ──
    cloud(ax, 8.1, 5.6, 2.4, 1.7, RED,
          'PSO\n负压优化', fs=13, italic=True)
    annot_box(ax, 8.1, 4.0, 2.6, 0.9,
              'N=40粒子  T=150迭代\n前30%探索 | 中40%平衡 | 后30%开发\nα: 50 → 500 动态增大',
              fc=GRAY, fs=9)

    # ── 右：约束注释 ──
    annot_box(ax, 11.4, 5.6, 2.8, 0.76,
              '18维约束边界\n各风箱[10%分位, 90%分位]',
              fc='#117A65', fs=9.5)

    # ── 右：输出椭圆 ──
    oval(ax, 11.4, 4.3, 3.2, 1.4, TEAL,
         '最优负压方案\n输出', fs=12)

    # ── 输出结果框 ──
    proc_box(ax, 11.4, 2.8, 3.2, 0.82,
             '最优18维风箱负压\nCO降低 62.8%  (3495→1299 ppm)',
             fc=GOLD, fs=9.5)
    ax.text(11.4, 2.3, '输出至 DCS 控制系统',
            ha='center', fontsize=9, color=GOLD, fontweight='bold')

    # ── 连线（所有坐标均在形状边缘，不进入内部）──
    # 表格右边缘(2.55) → 特征列表左边缘(3.7)
    thick_arr(ax, 2.55, 6.4,  3.7,  6.4)
    # 代理模型色块右边缘(2.9) → 特征列表左边缘(3.7)
    thick_arr(ax, 2.9,  3.85, 3.7,  4.2)
    # 特征列表右边缘(5.5) → PSO云朵左侧(7.05)
    thin_arr(ax, [(5.5, 6.6), (7.05, 6.0)])
    thin_arr(ax, [(5.5, 5.0), (7.05, 5.6)])
    # 特征列表底边缘(2.2) → 不动点注释顶边缘(2.05)
    thin_arr(ax, [(4.6, 2.2), (4.6, 2.05)])
    # 不动点注释右边缘(5.95) → 惩罚注释左边缘(6.65)
    thin_arr(ax, [(5.95, 1.6), (6.65, 1.95)])
    # 惩罚注释顶边缘(2.45) → 折点 → PSO云朵左侧(7.05)
    thin_arr(ax, [(7.9, 2.45), (7.9, 4.7), (7.05, 5.2)], li=1)
    # PSO云朵右侧(9.5) → 约束注释左边缘(10.0)
    thick_arr(ax, 9.5,  5.6,  10.0, 5.6)
    # PSO注释右边缘(9.7) → 折点 → 输出椭圆左边缘(9.8)
    thin_arr(ax, [(9.7, 4.0), (10.2, 4.0), (10.2, 4.3)], li=0)
    # 代理模型色块底→折→惩罚注释（代理预测回路）
    thin_arr(ax, [(1.5, 2.45), (1.5, 1.5), (6.65, 1.5)], li=1)
    # 约束注释底边缘(5.22) → 输出椭圆顶边缘(5.0)
    thick_arr(ax, 11.4, 5.22, 11.4, 5.0)
    # 输出椭圆底边缘(3.65) → 结果框顶边缘(3.21)
    thick_arr(ax, 11.4, 3.65, 11.4, 3.21)

    plt.tight_layout(pad=0.3)
    plt.savefig('paper_figures/Q3_PSO_arch.png', dpi=250,
                bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print('✓ paper_figures/Q3_PSO_arch.png')


if __name__ == '__main__':
    gen_q2()
    gen_q3()
    print('\n完成。')
