#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Q2 / Q3 标准 Gane-Sarson 数据流程图 PNG
符号规范：
  外部实体 → 矩形
  过  程  → 圆角矩形，中间横线分隔（上编号/下名称）
  数据存储 → 上下双横线（无左右边框）
  数据流  → 带标签箭头
"""
import os
os.chdir('/home/user/math_pro')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import fontManager
import numpy as np

fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']        = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

# ── 颜色常量 ─────────────────────────────────────────────────────────────────
CE = '#F0F0F0'    # 外部实体填充（浅灰）
CP = '#DEEBF7'    # 过程填充（浅蓝）
CS = '#FFFDE7'    # 数据存储填充（浅黄）
CEB = '#555555'   # 外部实体边框
CPB = '#2471A3'   # 过程边框/编号区背景
CSB = '#B7950B'   # 数据存储线色
CAR = '#333333'   # 箭头颜色


# ──────────────────────────────────────────────────────────────────────────────
# 绘制元素函数
# ──────────────────────────────────────────────────────────────────────────────

def draw_entity(ax, cx, cy, w, h, text, fs=9):
    """外部实体：普通矩形"""
    r = mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="square,pad=0", facecolor=CE, edgecolor=CEB, lw=2, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', zorder=4, multialignment='center')


def draw_process(ax, cx, cy, w, h, num, name, fs=8):
    """
    过程：圆角矩形，中间横线分隔
      上半：编号（浅蓝深色背景）
      下半：名称
    """
    # 外框
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.04", facecolor=CP, edgecolor=CPB, lw=1.8, zorder=3)
    ax.add_patch(r)
    # 中间横线
    mid = cy
    ax.plot([cx-w/2+0.04, cx+w/2-0.04], [mid, mid],
            color=CPB, lw=1.2, zorder=4)
    # 上半（编号）
    top_cy = cy + h/4
    ax.text(cx, top_cy, num, ha='center', va='center',
            fontsize=fs-0.5, color=CPB, fontweight='bold', zorder=5)
    # 下半（名称）
    bot_cy = cy - h/4
    ax.text(cx, bot_cy, name, ha='center', va='center',
            fontsize=fs, zorder=5, multialignment='center')


def draw_store(ax, cx, cy, w, h, sid, name, fs=8):
    """
    数据存储：上下双横线，无左右边框（标准 Gane-Sarson）
    左侧竖线分隔编号区
    """
    lx = cx - w/2
    rx = cx + w/2
    ty = cy + h/2
    by = cy - h/2
    # 填充背景
    bg = mpatches.Rectangle((lx, by), w, h, facecolor=CS, edgecolor='none', zorder=3)
    ax.add_patch(bg)
    # 上线（双）
    ax.plot([lx, rx], [ty,        ty       ], color=CSB, lw=2.0, zorder=4)
    ax.plot([lx, rx], [ty-0.045,  ty-0.045 ], color=CSB, lw=0.7, zorder=4)
    # 下线（双）
    ax.plot([lx, rx], [by,        by       ], color=CSB, lw=2.0, zorder=4)
    ax.plot([lx, rx], [by+0.045,  by+0.045 ], color=CSB, lw=0.7, zorder=4)
    # 左侧编号区竖线
    id_w = 0.30
    ax.plot([lx+id_w, lx+id_w], [by, ty], color=CSB, lw=1.0, zorder=4)
    ax.text(lx+id_w/2, cy, sid, ha='center', va='center',
            fontsize=fs-0.5, color=CSB, fontweight='bold', zorder=5)
    ax.text(lx+id_w+(w-id_w)/2, cy, name, ha='center', va='center',
            fontsize=fs, zorder=5, multialignment='center')


def flow(ax, pts, label='', label_pos=0.5, side='top', fs=7.5, lw=1.5, color=CAR):
    """
    数据流箭头（折线 + 箭头 + 标签）
    pts: [(x0,y0),(x1,y1),...,(xn,yn)]
    label_pos: 0~1，标签放在整条路径的哪个比例处（默认中点）
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # 画中间线段
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, lw=lw, zorder=5)
    # 最后一段末端箭头
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=12),
                zorder=6)

    if label:
        # 计算总路径长，找到 label_pos 对应的点
        segs = [(xs[i], ys[i], xs[i+1], ys[i+1]) for i in range(len(pts)-1)]
        lengths = [np.hypot(s[2]-s[0], s[3]-s[1]) for s in segs]
        total = sum(lengths)
        target = total * label_pos
        acc = 0
        for i, (x0, y0, x1, y1) in enumerate(segs):
            if acc + lengths[i] >= target:
                t = (target - acc) / lengths[i] if lengths[i] > 0 else 0
                lbx = x0 + t*(x1-x0)
                lby = y0 + t*(y1-y0)
                break
            acc += lengths[i]
        else:
            lbx, lby = xs[-1], ys[-1]

        dy = 0.09 if side == 'top' else -0.09
        dx = 0.0
        # 若该段是竖线，标签放右边
        seg_dx = segs[i][2] - segs[i][0]
        seg_dy = segs[i][3] - segs[i][1]
        if abs(seg_dx) < 0.01 and abs(seg_dy) > 0.01:
            dx = 0.10; dy = 0.0
        ax.text(lbx+dx, lby+dy, label,
                ha='center', va='center', fontsize=fs, color='#333333', zorder=7,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.9))


def legend(ax, lx=0.15, ly=0.55):
    draw_entity(ax, lx+0.45, ly, 0.82, 0.42, '外部实体', fs=7)
    draw_process(ax, lx+1.9, ly, 1.1, 0.42, 'n', '过程名称', fs=7)
    draw_store(ax, lx+3.5, ly, 1.3, 0.42, 'Dn', '数据存储', fs=7)
    ax.annotate('', xy=(lx+5.4, ly), xytext=(lx+4.8, ly),
                arrowprops=dict(arrowstyle='->', color=CAR, lw=1.3), zorder=6)
    ax.text(lx+5.65, ly, '数据流', va='center', fontsize=7.5)
    ax.text(lx, ly-0.28, '图例：', fontsize=7.5, color='#555555')


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 DFD  —  CO 浓度预测系统
# ═══════════════════════════════════════════════════════════════════════════════
#
# 布局（单位：英寸坐标，画布 0~16 × 0~10）
#
#  E1(传感器)  P1(预处理)  D1(原始数据)
#              ↓               ↑
#  E2(历史CO)  P2(特征工程) ← D2(滞后参数)
#              ↓               ↓
#              D3(特征矩阵)    P3(PSO优化)
#              ↓               ↓
#              P4(模型训练) ← D5(超参数)
#              ↓
#              D4(训练好模型)
#              ↓
#              P5(在线预测) → E3(调度系统)
#
def gen_q2():
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(7.5, 9.15, '问题二：烧结机 CO 浓度预测系统  数据流程图',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(7.5, 8.8,  'Level-1 DFD（Gane-Sarson 符号规范）',
            ha='center', va='center', fontsize=9, color='#666666')

    # ── 外部实体 ────────────────────────────────────────────────────────────
    draw_entity(ax, 1.1, 7.6,  1.6, 0.65, '传感器\n现场采集', fs=8.5)   # E1
    draw_entity(ax, 1.1, 2.8,  1.6, 0.65, 'CO 历史\n时序数据', fs=8.5)  # E2
    draw_entity(ax, 13.8, 3.8, 1.6, 0.65, '调度系统\n报警平台', fs=8.5) # E3

    # ── 过程 ────────────────────────────────────────────────────────────────
    # P1: 数据预处理
    draw_process(ax, 4.2, 7.6, 2.8, 0.90, '1.0',
                 '数据预处理\n(异常剔除/归一化)', fs=8)
    # P2: 特征工程
    draw_process(ax, 4.2, 5.5, 2.8, 0.90, '2.0',
                 '特征工程\n(滞后/梯度/统计/CO自回归)', fs=7.8)
    # P3: PSO 优化
    draw_process(ax, 8.5, 7.6, 2.8, 0.90, '3.0',
                 'PSO 超参数优化\n(8粒子×10次迭代)', fs=8)
    # P4: 模型训练
    draw_process(ax, 8.5, 5.5, 2.8, 0.90, '4.0',
                 'XGBoost 模型训练\n(TimeSeriesSplit 3折交叉验证)', fs=7.5)
    # P5: 在线预测
    draw_process(ax, 11.8, 3.8, 2.8, 0.90, '5.0',
                 'CO 浓度\n在线预测', fs=8)

    # ── 数据存储 ─────────────────────────────────────────────────────────────
    draw_store(ax, 4.2, 6.35, 2.8, 0.50, 'D1', '清洗后时序数据')  # D1
    draw_store(ax, 8.5, 6.35, 2.8, 0.50, 'D2', '最优滞后参数')    # D2（Q1输出）
    draw_store(ax, 4.2, 4.3,  2.8, 0.50, 'D3', '特征矩阵 (84维)') # D3
    draw_store(ax, 8.5, 4.3,  2.8, 0.50, 'D4', '最优超参数配置')  # D4
    draw_store(ax, 11.8, 5.5, 2.8, 0.50, 'D5', '训练好的 XGBoost 模型') # D5

    # ── 数据流 ───────────────────────────────────────────────────────────────
    # E1 → P1
    flow(ax, [(1.9, 7.6), (2.8, 7.6)], '原始传感器数据')
    # P1 → D1
    flow(ax, [(4.2, 7.15), (4.2, 6.6)], '清洗后数据')
    # D1 → P2
    flow(ax, [(4.2, 6.1), (4.2, 5.95)], '时序样本')
    # D2(Q1滞后参数) → P2
    flow(ax, [(7.1, 6.35), (6.0, 6.35), (6.0, 5.95)],
         'Q1 最优滞后量', label_pos=0.4)
    # P2 → D3
    flow(ax, [(4.2, 5.05), (4.2, 4.55)], '对齐后特征向量')
    # E2 → D3
    flow(ax, [(1.9, 2.8), (4.2, 2.8), (4.2, 4.05)],
         'CO 历史序列', label_pos=0.35, side='bot')
    # D3 → P3（用于PSO交叉验证）
    flow(ax, [(5.6, 4.3), (8.5, 4.3), (8.5, 7.15)],
         '训练样本（CV用）', label_pos=0.3, side='bot')
    # D3 → P4
    flow(ax, [(5.6, 4.3), (7.1, 4.3)], '训练样本', label_pos=0.5)
    # P3 → D4
    flow(ax, [(8.5, 7.15), (8.5, 6.6)], 'CV最优参数')
    # D4 → P4
    flow(ax, [(8.5, 6.1), (8.5, 5.95)], '超参数配置')
    # P4 → D5
    flow(ax, [(9.9, 5.5), (10.4, 5.5)], '训练好的模型')
    # D5 → P5
    flow(ax, [(13.2, 5.25), (13.2, 4.25), (13.1, 4.25)],
         '预测模型', side='bot')
    # P2 → P5（实时特征）
    flow(ax, [(5.6, 5.5), (6.5, 5.5), (6.5, 3.8), (10.4, 3.8)],
         '实时特征向量', label_pos=0.5, side='top')
    # P5 → E3
    flow(ax, [(13.2, 3.8), (13.0, 3.8)], 'CO 预测值 (ppm)')

    legend(ax)
    plt.tight_layout(pad=0.3)
    plt.savefig('paper_figures/Q2_DFD.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✓ paper_figures/Q2_DFD.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 DFD  —  风箱负压优化系统
# ═══════════════════════════════════════════════════════════════════════════════
def gen_q3():
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(7.5, 9.15, '问题三：烧结机风箱负压优化系统  数据流程图',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(7.5, 8.8,  'Level-1 DFD（Gane-Sarson 符号规范）',
            ha='center', va='center', fontsize=9, color='#666666')

    # ── 外部实体 ─────────────────────────────────────────────────────────────
    draw_entity(ax, 1.0, 7.6,  1.6, 0.65, '实时工况\n监测系统', fs=8.5)  # E1
    draw_entity(ax, 1.0, 3.2,  1.6, 0.65, '设备约束\n规格参数', fs=8.5)  # E2
    draw_entity(ax, 13.8, 4.5, 1.6, 0.65, '控制执行\n系统 / DCS', fs=8.5) # E3

    # ── 过程 ─────────────────────────────────────────────────────────────────
    draw_process(ax, 4.0, 7.6, 2.8, 0.90, '1.0',
                 '状态变量采集\n(负压/温度/机速)', fs=8)
    draw_process(ax, 4.0, 5.5, 2.8, 0.90, '2.0',
                 '特征向量构造\n(与Q2特征体系一致)', fs=7.8)
    draw_process(ax, 8.5, 7.6, 2.8, 0.90, '3.0',
                 'CO 自回归特征\n不动点初始化', fs=8)
    draw_process(ax, 8.5, 5.5, 2.8, 0.90, '4.0',
                 'PSO 优化搜索\n(最小化预测 CO)', fs=8)
    draw_process(ax, 8.5, 3.5, 2.8, 0.90, '5.0',
                 '约束可行性\n校验', fs=8)
    draw_process(ax, 11.8, 4.5, 2.8, 0.90, '6.0',
                 '最优负压\n方案输出', fs=8)

    # ── 数据存储 ──────────────────────────────────────────────────────────────
    draw_store(ax, 4.0, 6.35, 2.8, 0.50, 'D1', '当前工况快照')
    draw_store(ax, 4.0, 4.3,  2.8, 0.50, 'D2', '84维特征向量')
    draw_store(ax, 8.5, 6.35, 2.8, 0.50, 'D3', 'Q2 XGBoost 模型')
    draw_store(ax, 8.5, 4.3,  2.8, 0.50, 'D4', '粒子群状态 (18维)')
    draw_store(ax, 5.5, 2.3,  2.8, 0.50, 'D5', '约束边界参数')
    draw_store(ax, 11.8, 3.0, 2.8, 0.50, 'D6', '最优风箱负压方案')

    # ── 数据流 ────────────────────────────────────────────────────────────────
    # E1 → P1
    flow(ax, [(1.8, 7.6), (2.6, 7.6)], '实时传感数据')
    # P1 → D1
    flow(ax, [(4.0, 7.15), (4.0, 6.6)], '状态采样')
    # D1 → P2
    flow(ax, [(4.0, 6.1), (4.0, 5.95)], '状态向量')
    # P2 → D2
    flow(ax, [(4.0, 5.05), (4.0, 4.55)], '特征向量')
    # D2 → P3（CO 自回归特征）
    flow(ax, [(5.4, 4.3), (7.0, 4.3), (7.0, 7.6), (7.1, 7.6)],
         'CO 自回归特征', label_pos=0.45)
    # D3 → P3
    flow(ax, [(7.1, 6.35), (7.1, 7.6)], 'Q2 模型', side='bot')
    # P3 → P4（初始 CO 估计）
    flow(ax, [(8.5, 7.15), (8.5, 5.95)], '初始 CO 估计')
    # D3 → P4（预测用）
    flow(ax, [(9.9, 6.35), (9.9, 5.95)], '模型（预测用）')
    # D2 → P4
    flow(ax, [(5.4, 4.3), (8.5, 4.3), (8.5, 5.05)], '特征向量', label_pos=0.4)
    # P4 → D4（粒子更新）
    flow(ax, [(9.9, 5.5), (10.5, 5.5), (10.5, 4.3), (9.9, 4.3)],
         '粒子位置/速度', label_pos=0.3)
    # D4 → P4（迭代）
    flow(ax, [(9.9, 4.55), (9.9, 5.05)], '迭代更新')
    # P4 → P5（候选方案）
    flow(ax, [(8.5, 5.05), (8.5, 3.95)], '候选负压方案')
    # E2 → D5
    flow(ax, [(1.8, 3.2), (3.7, 3.2), (3.7, 2.3), (4.1, 2.3)],
         '设备约束参数', label_pos=0.35)
    # D5 → P5
    flow(ax, [(6.9, 2.3), (8.5, 2.3), (8.5, 3.05)],
         '约束边界', label_pos=0.5)
    # P5 → P6（可行 → 输出）
    flow(ax, [(9.9, 3.5), (10.4, 3.5), (10.4, 4.5), (10.4, 4.5)],
         '可行方案')
    # P6 → D6
    flow(ax, [(11.8, 4.05), (11.8, 3.25)], '最优解')
    # D6 → E3
    flow(ax, [(13.2, 3.0), (13.8, 3.0), (13.8, 4.175)],
         '负压指令 (18维)', label_pos=0.4)
    # P5 → P4（不可行，惩罚回路）
    flow(ax, [(7.1, 3.5), (6.2, 3.5), (6.2, 5.5), (7.1, 5.5)],
         '惩罚/淘汰', label_pos=0.4)

    legend(ax)
    plt.tight_layout(pad=0.3)
    plt.savefig('paper_figures/Q3_DFD.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✓ paper_figures/Q3_DFD.png')


if __name__ == '__main__':
    gen_q2()
    gen_q3()
    print('\n全部完成。')
