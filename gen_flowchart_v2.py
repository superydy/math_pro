#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 / Q3 总体流程图
  风格：参考图配色（云朵 PSO / 椭圆模型 / 彩色矩形 / 平行四边形 I/O）
  内容：完全按照用户提供的两张流程图
"""
import os
os.chdir('/home/user/math_pro')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon
from matplotlib.font_manager import fontManager
import numpy as np

fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']        = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

# ── 颜色 ─────────────────────────────────────────────────────────────────────
C_START  = '#1C2833'
C_IO     = '#154360'
C_PROC   = '#1F618D'
C_TEAL   = '#148F77'
C_LOOP   = '#0E6655'
C_PURPLE = '#6C3483'
C_DEC    = '#922B21'
C_CLOUD  = '#C0392B'
C_GOLD   = '#B7770D'
C_ARROW  = '#1C2833'
WHITE    = '#FFFFFF'

# ── 形状绘制 ──────────────────────────────────────────────────────────────────
def _t(ax, cx, cy, s, fc=WHITE, fs=9, bold=False, zorder=5):
    ax.text(cx, cy, s, ha='center', va='center', fontsize=fs,
            color=fc, fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=zorder)

def node_oval(ax, cx, cy, w, h, txt, color=C_START, fs=10):
    ax.add_patch(Ellipse((cx, cy), w, h,
                         fc=color, ec=WHITE, lw=1.8, zorder=3))
    _t(ax, cx, cy, txt, fs=fs, bold=True)

def node_rect(ax, cx, cy, w, h, txt, color=C_PROC, fs=9):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle='round,pad=0.05',
                 fc=color, ec=WHITE, lw=1.5, zorder=3))
    _t(ax, cx, cy, txt, fs=fs)

def node_para(ax, cx, cy, w, h, txt, color=C_IO, fs=9):
    sk = 0.13
    pts = np.array([[cx-w/2+sk*h, cy+h/2],
                    [cx+w/2+sk*h, cy+h/2],
                    [cx+w/2-sk*h, cy-h/2],
                    [cx-w/2-sk*h, cy-h/2]])
    ax.add_patch(Polygon(pts, closed=True,
                 fc=color, ec=WHITE, lw=1.5, zorder=3))
    _t(ax, cx, cy, txt, fs=fs)

def node_diamond(ax, cx, cy, w, h, txt, color=C_DEC, fs=9):
    pts = np.array([[cx, cy+h/2],[cx+w/2, cy],
                    [cx, cy-h/2],[cx-w/2, cy]])
    ax.add_patch(Polygon(pts, closed=True,
                 fc=color, ec=WHITE, lw=1.5, zorder=3))
    _t(ax, cx, cy, txt, fs=fs)

def node_cloud(ax, cx, cy, w, h, txt, color=C_CLOUD, fs=12):
    """云朵形（与架构图一致）"""
    ax.add_patch(Ellipse((cx, cy-h*0.04), w*0.92, h*0.72,
                         fc=color, ec='none', zorder=3))
    for dx, dy_, bw, bh in [(-0.30,+0.30,0.30,0.38),
                              (-0.07,+0.42,0.32,0.42),
                              (+0.16,+0.40,0.28,0.40),
                              (+0.35,+0.26,0.24,0.33)]:
        ax.add_patch(Ellipse((cx+dx*w, cy+dy_*h), bw*w, bh*h,
                             fc=color, ec='none', zorder=4))
    for dx, bw in [(-0.50,0.20),(+0.50,0.20)]:
        ax.add_patch(Ellipse((cx+dx*w, cy+0.02*h), bw*w, 0.38*h,
                             fc=color, ec='none', zorder=3))
    _t(ax, cx, cy, txt, fs=fs, bold=True, zorder=6)

def node_model(ax, cx, cy, w, h, txt, color=C_TEAL, fs=10):
    """椭圆形（代表训练好的模型）"""
    ax.add_patch(Ellipse((cx, cy), w, h,
                         fc=color, ec=WHITE, lw=1.8, zorder=3))
    _t(ax, cx, cy, txt, fs=fs, bold=True)

# ── 箭头 ─────────────────────────────────────────────────────────────────────
def down_arr(ax, x, y0, y1, lbl='', lbl_x=None, color=C_ARROW, lw=2.2):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=14,
                                shrinkA=3, shrinkB=3), zorder=6)
    if lbl:
        lx = lbl_x if lbl_x else x + 0.18
        ax.text(lx, (y0+y1)/2, lbl, ha='left', va='center',
                fontsize=9, color='#222222', zorder=7,
                bbox=dict(fc=WHITE, ec='none', pad=0.1, alpha=0.9))

def side_arr(ax, pts, lbl='', lbl_idx=0, lbl_side='right',
             color=C_ARROW, lw=2.2):
    """折线反馈箭头"""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=lw, zorder=6)
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=14,
                                shrinkA=3, shrinkB=3), zorder=6)
    if lbl:
        i = lbl_idx
        mx = (xs[i]+xs[i+1])/2; my = (ys[i]+ys[i+1])/2
        dx = 0.12 if lbl_side == 'right' else -0.12; dy = 0.0
        if abs(xs[i]-xs[i+1]) < 0.01: dx = 0.13; dy = 0.0
        ax.text(mx+dx, my+dy, lbl, ha='center', va='center',
                fontsize=9, color='#222222', zorder=7,
                bbox=dict(fc=WHITE, ec='none', pad=0.1, alpha=0.9))


# ══════════════════════════════════════════════════════════════════════════════
# Q2 总体流程图
# ══════════════════════════════════════════════════════════════════════════════
def gen_q2():
    H = 25
    fig, ax = plt.subplots(figsize=(7, H*0.42))
    ax.set_xlim(0, 7); ax.set_ylim(0, H)
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    ax.text(3.5, H-0.35, 'Q2  CO浓度预测  总体流程图',
            ha='center', fontsize=13, fontweight='bold', color='#1C2833')

    cx = 3.5          # 中心 x
    SW, SH = 4.8, 0.82  # 标准矩形宽高
    DW, DH = 3.6, 1.05  # 菱形宽高

    # ── 节点（从上到下） ────────────────────────────────────────────────────
    y1  = H-1.1;  node_oval   (ax, cx, y1,  2.6,  0.70, '开  始')
    y2  = H-2.6;  node_para   (ax, cx, y2,  SW,   SH,   '输入烧结机传感时序数据')
    y3  = H-4.2;  node_rect   (ax, cx, y3,  SW,   SH,
                               '数据预处理\n(异常剔除 / 归一化)',
                               color=C_PROC)
    y4  = H-5.9;  node_rect   (ax, cx, y4,  SW,   SH+0.1,
                               '时间滞后对齐 + 特征提取\n(互相关分析 / 84维特征工程)',
                               color=C_TEAL)
    y5  = H-7.8;  node_cloud  (ax, cx, y5,  4.6,  1.55, 'PSO算法调参', fs=13)
    y6  = H-10.0; node_rect   (ax, cx, y6,  SW,   SH+0.1,
                               '计算粒子适应度，更新粒子位置\n(3折CV评估 XGBoost R²)',
                               color=C_LOOP)
    y7  = H-11.8; node_diamond(ax, cx, y7,  DW,   DH,   '迭代次数>10?')
    y8  = H-13.6; node_model  (ax, cx, y8,  4.8,  0.95,
                               '以最好超参数  训练 XGBoost 模型')
    y9  = H-15.4; node_diamond(ax, cx, y9,  DW,   DH,   'R²>0.9?')
    y10 = H-17.1; node_para   (ax, cx, y10, SW,   SH,
                               '输出结果与评价指标',
                               color=C_GOLD)
    y11 = H-18.6; node_oval   (ax, cx, y11, 2.6,  0.70, '结  束')

    # ── 主干箭头 ────────────────────────────────────────────────────────────
    # 两节点之间取边缘 y
    down_arr(ax, cx, y1-0.35,  y2+SH/2)
    down_arr(ax, cx, y2-SH/2,  y3+SH/2)
    down_arr(ax, cx, y3-SH/2,  y4+0.46)
    down_arr(ax, cx, y4-0.46,  y5+0.80)   # 进云朵顶
    down_arr(ax, cx, y5-0.55,  y6+0.46)   # 出云朵底
    down_arr(ax, cx, y6-0.46,  y7+DH/2)
    down_arr(ax, cx, y7-DH/2,  y8+0.475,  lbl='是', lbl_x=cx+0.15)
    down_arr(ax, cx, y8-0.475, y9+DH/2)
    down_arr(ax, cx, y9-DH/2,  y10+SH/2,  lbl='是', lbl_x=cx+0.15)
    down_arr(ax, cx, y10-SH/2, y11+0.35)

    # ── 反馈箭头 ────────────────────────────────────────────────────────────
    # 迭代次数 "否" → 右侧 → 上 → 计算粒子适应度 右边缘
    rx1 = cx + DW/2          # 菱形右尖 = 5.3
    rx_side1 = 6.35           # 右侧折返 x
    side_arr(ax,
             [(rx1, y7), (rx_side1, y7), (rx_side1, y6), (cx+SW/2, y6)],
             lbl='否', lbl_idx=0, lbl_side='right')

    # R²>0.9 "否" → 更右侧 → 上 → PSO云朵 右边缘
    rx2 = cx + DW/2           # 5.3
    rx_side2 = 6.8            # 更右的折返 x
    cloud_rx  = cx + 0.5*4.6 + 0.2*4.6/2  # 云朵右边缘 ≈ 3.5+2.3+0.46=6.26
    side_arr(ax,
             [(rx2, y9), (rx_side2, y9), (rx_side2, y5), (cloud_rx, y5)],
             lbl='否', lbl_idx=1, lbl_side='right')

    plt.tight_layout(pad=0.4)
    plt.savefig('paper_figures/Q2_flowchart_v2.png', dpi=170,
                bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print('✓ paper_figures/Q2_flowchart_v2.png')


# ══════════════════════════════════════════════════════════════════════════════
# Q3 总体流程图
# ══════════════════════════════════════════════════════════════════════════════
def gen_q3():
    H = 30
    fig, ax = plt.subplots(figsize=(7, H*0.38))
    ax.set_xlim(0, 7); ax.set_ylim(0, H)
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    ax.text(3.5, H-0.35, 'Q3  风箱负压优化  总体流程图',
            ha='center', fontsize=13, fontweight='bold', color='#1C2833')

    cx = 3.5
    SW, SH = 4.8, 0.82
    DW, DH = 3.8, 1.05

    # ── 节点 ──────────────────────────────────────────────────────────────
    y1  = H-1.1;  node_oval   (ax, cx, y1,  2.6,  0.70, '开  始')
    y2  = H-2.6;  node_para   (ax, cx, y2,  SW,   SH,
                               '输入当前烧结机工况数据')
    y3  = H-4.2;  node_model  (ax, cx, y3,  4.8,  0.95,
                               '载入问题2训练好的\nXGBoost模型及参数')
    y4  = H-5.9;  node_rect   (ax, cx, y4,  SW,   SH+0.1,
                               '构造特征向量及CO不动点初始化\n(不动点迭代至收敛)',
                               color=C_PURPLE)
    y5  = H-7.9;  node_cloud  (ax, cx, y5,  4.6,  1.55, 'PSO负压优化搜索', fs=13)
    y6  = H-10.1; node_diamond(ax, cx, y6,  DW,   DH,
                               '压强是否\n满足范围?')
    y7  = H-12.1; node_rect   (ax, cx, y7,  SW,   SH,
                               '更新粒子速度和位置\n(自适应 w / c1 / c2)',
                               color=C_LOOP)
    y8  = H-14.0; node_diamond(ax, cx, y8,  DW,   DH,
                               '迭代次数是否\n达到上限?')
    y9  = H-15.9; node_rect   (ax, cx, y9,  SW,   SH,
                               '提取全局最优方案 gbest\n(18维最优风箱负压)',
                               color=C_PROC)
    y10 = H-17.8; node_diamond(ax, cx, y10, DW,   DH,
                               '所有负压是否\n在物理约束内?')
    y11 = H-19.8; node_rect   (ax, cx, y11, SW,   SH+0.1,
                               '计算优化效果及最佳方案\n(CO降低量 / 贡献度分析)',
                               color=C_LOOP)
    y12 = H-21.5; node_para   (ax, cx, y12, SW,   SH,
                               '输出最优18维风箱负压方案',
                               color=C_GOLD)
    y13 = H-23.0; node_oval   (ax, cx, y13, 2.6,  0.70, '结  束')

    # ── 主干箭头 ────────────────────────────────────────────────────────────
    down_arr(ax, cx, y1-0.35,   y2+SH/2)
    down_arr(ax, cx, y2-SH/2,   y3+0.475)
    down_arr(ax, cx, y3-0.475,  y4+0.46)
    down_arr(ax, cx, y4-0.46,   y5+0.80)
    down_arr(ax, cx, y5-0.55,   y6+DH/2)
    down_arr(ax, cx, y6-DH/2,   y7+SH/2,   lbl='满足',    lbl_x=cx+0.15)
    down_arr(ax, cx, y7-SH/2,   y8+DH/2)
    down_arr(ax, cx, y8-DH/2,   y9+SH/2,   lbl='是',       lbl_x=cx+0.15)
    down_arr(ax, cx, y9-SH/2,   y10+DH/2)
    down_arr(ax, cx, y10-DH/2,  y11+0.46,  lbl='是',       lbl_x=cx+0.15)
    down_arr(ax, cx, y11-0.46,  y12+SH/2)
    down_arr(ax, cx, y12-SH/2,  y13+0.35)

    # ── 反馈箭头 ────────────────────────────────────────────────────────────
    cloud_rx = cx + 0.5*4.6 + 0.2*4.6/2   # 云朵右边缘 ≈ 6.26
    cloud_lx = cx - 0.5*4.6 - 0.2*4.6/2   # 云朵左边缘 ≈ 0.74

    # 压强不满足 → 右侧 → 上 → PSO云朵右边缘（惩罚后重新搜索）
    side_arr(ax,
             [(cx+DW/2, y6), (6.5, y6), (6.5, y5), (cloud_rx, y5)],
             lbl='不满足，惩罚', lbl_idx=0, lbl_side='right')

    # 迭代次数未达上限 → 左侧 → 上 → PSO云朵左边缘
    side_arr(ax,
             [(cx-DW/2, y8), (0.5, y8), (0.5, y5), (cloud_lx, y5)],
             lbl='否', lbl_idx=1, lbl_side='right')

    # 负压不在约束内 → 右侧更远 → 上 → PSO云朵右边缘（扩大搜索）
    side_arr(ax,
             [(cx+DW/2, y10), (6.8, y10), (6.8, y5), (cloud_rx, y5)],
             lbl='否，扩大搜索', lbl_idx=0, lbl_side='right')

    plt.tight_layout(pad=0.4)
    plt.savefig('paper_figures/Q3_flowchart_v2.png', dpi=170,
                bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print('✓ paper_figures/Q3_flowchart_v2.png')


if __name__ == '__main__':
    gen_q2()
    gen_q3()
    print('\n完成。')
