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
    """粗箭头（主数据流）"""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=20), zorder=7)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.05, my+0.12, lbl, ha='center', va='center',
                fontsize=fs, color=color, zorder=8,
                bbox=dict(fc='white', ec='none', pad=0.1, alpha=0.85))


def thin_arr(ax, pts, lbl='', color=BEIGE, lw=1.5, fs=7.8, li=0, ls='top'):
    """细金箭头（数据连接）"""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=lw, zorder=6)
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=13), zorder=6)
    if lbl:
        mx = (xs[li]+xs[li+1])/2; my = (ys[li]+ys[li+1])/2
        dy = 0.12 if ls == 'top' else -0.12; dx = 0.0
        if abs(xs[li]-xs[li+1]) < 0.01: dx = 0.14; dy = 0.0
        ax.text(mx+dx, my+dy, lbl, ha='center', va='center',
                fontsize=fs, color='#444444', zorder=8,
                bbox=dict(fc='white', ec='none', pad=0.1, alpha=0.9))


# ══════════════════════════════════════════════════════════════════════════════
# Q2 PSO 超参数优化架构图
# ══════════════════════════════════════════════════════════════════════════════
def gen_q2():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    # ── 标题 ──
    ax.text(7.5, 8.65,
            'Q2  XGBoost-PSO 超参数优化  系统架构图',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1C2833')

    # ── 左：输入数据 ──
    ax.text(1.5, 7.85, '输入训练数据', ha='center', fontsize=9,
            fontweight='bold', color='#1C2833')
    table_box(ax, 1.5, 7.1,
              [['x1', 'x2', '...', 'x84'],
               ['p11', 'p12', '...', 'CO1'],
               ['p21', 'p22', '...', 'CO2'],
               ['.',  '.',  '..',  '.' ]])
    ax.text(1.5, 5.75, '1624 × 84  训练样本',
            ha='center', fontsize=8, color='#555555')

    # ── 左下：特征分组说明（仿椭圆内含圆）──
    ax.add_patch(Ellipse((1.8, 4.0), 3.2, 2.6,
                         fc='#EBF5FB', ec='#2980B9', lw=1.5, zorder=2))
    ax.add_patch(Ellipse((2.9, 4.5), 1.4, 1.0,
                         fc='#D6EAF8', ec='#2980B9', lw=1, zorder=3))
    ax.text(2.9, 4.5, 'p1\n物理特征\n41维',
            ha='center', va='center', fontsize=7.5,
            color='#1C2833', zorder=4, multialignment='center')
    ax.add_patch(Ellipse((1.0, 4.6), 1.1, 0.8,
                         fc='#FDEBD0', ec='#E67E22', lw=1, zorder=3))
    ax.text(1.0, 4.6, 'p2\n梯度\n34维',
            ha='center', va='center', fontsize=7.5,
            color='#1C2833', zorder=4, multialignment='center')
    ax.add_patch(Ellipse((1.2, 3.3), 1.0, 0.78,
                         fc='#E8DAEF', ec='#8E44AD', lw=1, zorder=3))
    ax.text(1.2, 3.3, 'p3\n统计\n4维',
            ha='center', va='center', fontsize=7.5,
            color='#1C2833', zorder=4, multialignment='center')
    ax.add_patch(Ellipse((2.5, 3.2), 1.1, 0.78,
                         fc='#D5F5E3', ec='#148F77', lw=1, zorder=3))
    ax.text(2.5, 3.2, 'p4\nCO自回归\n5维',
            ha='center', va='center', fontsize=7.5,
            color='#1C2833', zorder=4, multialignment='center')
    ax.text(1.8, 2.55, '特征分组（84维）',
            ha='center', fontsize=8, color='#555555', style='italic')

    # ── 中：特征列表高矩形 ──
    feat_box(ax, 5.2, 5.8, 1.7, 4.2,
             '特征体系',
             ['机速 × 1', '负压 × 18', '温度 × 18',
              '大烟道 × 4', '压力梯度×17', '温度梯度×17',
              '统计特征×4', 'CO自回归×5'],
             bg='#148F77')

    # ── 中：3折CV注释 ──
    annot_box(ax, 5.2, 2.4, 2.4, 0.7,
              '3折 TimeSeriesSplit CV\n（PSO内部适应度评估）',
              fc=PURPLE, fs=8)

    # ── 中右：PSO 搜索空间注释 ──
    annot_box(ax, 8.8, 3.0, 2.5, 1.0,
              '7维超参数搜索空间\nlr∈[0.01,0.50]\ndepth∈[2,8]  n_est∈[50,500]',
              fc=GRAY, fs=7.8)

    # ── 右上：PSO云朵主模型 ──
    cloud(ax, 9.5, 6.8, 3.2, 2.2, RED,
          'PSO\n超参数优化',
          fs=12, italic=True)
    annot_box(ax, 9.5, 5.1, 2.8, 0.72,
              'N=8粒子  ×  T=10迭代\nw=0.8,  c1=c2=2.0（固定）',
              fc='#922B21', fs=8)

    # ── 右：XGBoost模型椭圆 ──
    oval(ax, 12.6, 6.2, 3.5, 1.6, TEAL,
         'XGBoost\n预测模型', fs=11)

    # ── 右下：5折最终评估注释 ──
    annot_box(ax, 12.6, 4.8, 3.0, 0.72,
              '5折 TimeSeriesSplit CV\n最终评估  R²=0.8862',
              fc='#117A65', fs=8)

    # ── 右下：输出 ──
    proc_box(ax, 12.6, 3.5, 3.0, 0.75,
             'CO 浓度预测模型\n测试集 R²=0.9493  MAE=43.18 ppm',
             fc=GOLD, fs=8.5)
    ax.text(12.6, 3.05, '输出', ha='center', fontsize=8,
            color=GOLD, fontweight='bold')

    # ── 连线 ──
    # 输入 → 特征列表
    thick_arr(ax, 2.55, 7.1, 4.3, 6.8, '')
    # 特征分组 → 特征列表
    thick_arr(ax, 2.85, 4.5, 4.3, 5.0, '')
    # 特征列表 → PSO
    thin_arr(ax, [(6.05, 7.5), (8.0, 7.5)], '特征矩阵 X_tr')
    thin_arr(ax, [(6.05, 6.5), (8.0, 6.1)], '84维特征')
    thin_arr(ax, [(6.05, 5.2), (8.0, 5.5)], '')
    # 特征列表 → 3折CV
    thin_arr(ax, [(5.2, 3.7), (5.2, 2.75)], 'CV数据')
    # 3折CV → PSO（反馈适应度）
    thin_arr(ax, [(6.4, 2.4), (8.0, 3.0)], 'CV-R² 适应度')
    # PSO → XGBoost
    thick_arr(ax, 10.65, 6.5, 10.8, 6.3, '')
    # gbest → XGBoost
    thin_arr(ax, [(9.5, 5.7), (9.5, 4.7), (10.8, 4.7)],
             'gbest 最优参数', li=1)
    # XGBoost → 5折CV
    thin_arr(ax, [(12.6, 5.4), (12.6, 5.15)], '')
    # 5折CV → 输出
    thick_arr(ax, 12.6, 4.44, 12.6, 3.88, '')

    plt.tight_layout(pad=0.3)
    plt.savefig('paper_figures/Q2_PSO_arch.png', dpi=170,
                bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print('✓ paper_figures/Q2_PSO_arch.png')


# ══════════════════════════════════════════════════════════════════════════════
# Q3 PSO 负压优化架构图
# ══════════════════════════════════════════════════════════════════════════════
def gen_q3():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    ax.text(7.5, 8.65,
            'Q3  PSO 风箱负压优化  系统架构图',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1C2833')

    # ── 左：当前工况输入 ──
    ax.text(1.5, 7.85, '当前工况快照', ha='center', fontsize=9,
            fontweight='bold', color='#1C2833')
    table_box(ax, 1.5, 7.0,
              [['风箱', 'p1', 'p2', '...', 'p18'],
               ['负压', '-11', '-14', '...', '-13'],
               ['温度', '89', '112', '...', '145'],
               ['机速', '1.46', 'm/min', '', '']],
              fs=7.8)

    # ── 左下：Q2 模型组（椭圆内含圆）──
    ax.add_patch(Ellipse((1.9, 3.9), 3.4, 2.8,
                         fc='#EBF5FB', ec='#148F77', lw=1.8, zorder=2))
    ax.add_patch(Ellipse((2.0, 4.3), 1.5, 1.0,
                         fc='#D1F2EB', ec='#148F77', lw=1, zorder=3))
    ax.text(2.0, 4.3, 'Q2 XGBoost\n预测模型',
            ha='center', va='center', fontsize=7.8,
            color='#0E6655', fontweight='bold', zorder=4, multialignment='center')
    ax.add_patch(Ellipse((1.1, 3.2), 1.2, 0.85,
                         fc='#FDEBD0', ec='#E67E22', lw=1, zorder=3))
    ax.text(1.1, 3.2, '标准化\n参数 scaler',
            ha='center', va='center', fontsize=7.5,
            color='#1C2833', zorder=4, multialignment='center')
    ax.add_patch(Ellipse((2.8, 3.2), 1.2, 0.85,
                         fc='#E8DAEF', ec='#8E44AD', lw=1, zorder=3))
    ax.text(2.8, 3.2, '特征\n工程参数',
            ha='center', va='center', fontsize=7.5,
            color='#1C2833', zorder=4, multialignment='center')
    ax.text(1.9, 2.5, 'Q2 训练产物（代理模型）',
            ha='center', fontsize=8, color='#555555', style='italic')

    # ── 中：特征构造列表 ──
    feat_box(ax, 5.2, 5.7, 1.7, 4.0,
             '特征构造\n(84维)',
             ['机速 × 1', '负压 × 18', '温度 × 18',
              '大烟道 × 4', '压力梯度×17', '温度梯度×17',
              '统计特征 × 4', 'CO自回归 × 5'],
             bg='#1A5276')

    # ── 中：不动点初始化注释 ──
    annot_box(ax, 5.2, 2.4, 2.6, 0.88,
              '不动点初始化\nX(t+1) = 0.5·X_t + 0.5·f(X_t)\n迭代至 |ΔX| < 1 ppm 收敛',
              fc=PURPLE, fs=7.8)

    # ── 中右：可靠性惩罚注释 ──
    annot_box(ax, 8.8, 2.8, 2.6, 0.88,
              '可靠性惩罚项\npenalty = Σ exp(-10·dist)\n目标 = CO_pred + α·penalty',
              fc='#922B21', fs=7.8)

    # ── 右上：PSO云朵主模型 ──
    cloud(ax, 9.5, 6.8, 3.2, 2.2, RED,
          'PSO\n负压优化',
          fs=12, italic=True)

    # ── PSO 自适应参数注释 ──
    annot_box(ax, 9.5, 4.95, 2.9, 0.88,
              'N=40粒子  ×  T=150迭代\n前30%探索→中40%平衡→后30%开发\nα: 50 → 500 动态增大',
              fc=GRAY, fs=7.8)

    # ── 右：约束边界注释 ──
    annot_box(ax, 12.7, 5.9, 2.8, 0.72,
              '18维约束边界\n各风箱: [10%分位数, 90%分位数]',
              fc='#117A65', fs=8)

    # ── 右下：输出椭圆 ──
    oval(ax, 12.7, 4.7, 3.4, 1.4, TEAL,
         '最优负压方案\n输出', fs=11)

    # ── 输出结果框 ──
    proc_box(ax, 12.7, 3.4, 3.2, 0.82,
             '最优 18 维风箱负压\nCO 降低 62.8%  (3495→1299 ppm)',
             fc=GOLD, fs=8.5)
    ax.text(12.7, 2.95, '输出至 DCS 控制系统',
            ha='center', fontsize=8, color=GOLD, fontweight='bold')

    # ── 连线 ──
    # 当前工况 → 特征构造
    thick_arr(ax, 2.55, 7.0, 4.3, 6.6, '')
    # Q2模型 → 特征构造
    thick_arr(ax, 3.05, 4.3, 4.3, 5.5, '')
    # 特征构造 → 不动点初始化
    thin_arr(ax, [(5.2, 3.7), (5.2, 2.84)], 'CO自回归占位')
    # 不动点 → PSO
    thin_arr(ax, [(6.5, 2.4), (8.0, 3.2)], 'CO初始估计')
    # 特征构造 → PSO
    thin_arr(ax, [(6.05, 7.0), (8.0, 7.5)], '84维特征向量')
    thin_arr(ax, [(6.05, 6.0), (8.0, 6.2)], '')
    # 可靠性惩罚 → PSO
    thin_arr(ax, [(8.8, 3.24), (8.8, 5.8), (8.0, 6.0)],
             '惩罚函数', li=1)
    # PSO → 约束
    thin_arr(ax, [(10.65, 7.1), (12.7, 6.26)], '候选方案')
    # PSO → 自适应参数（自环标注）
    thin_arr(ax, [(9.5, 5.7), (9.5, 5.39)], '参数调度')
    # 约束 → 输出椭圆
    thick_arr(ax, 12.7, 5.54, 12.7, 5.4, '')
    # 输出椭圆 → 结果框
    thick_arr(ax, 12.7, 4.0, 12.7, 3.81, '')
    # Q2模型 → PSO（代理预测）
    thin_arr(ax, [(3.05, 3.9), (4.3, 3.9), (4.3, 2.4), (6.5, 2.4)],
             'CO预测代理', li=2)

    plt.tight_layout(pad=0.3)
    plt.savefig('paper_figures/Q3_PSO_arch.png', dpi=170,
                bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print('✓ paper_figures/Q3_PSO_arch.png')


if __name__ == '__main__':
    gen_q2()
    gen_q3()
    print('\n完成。')
