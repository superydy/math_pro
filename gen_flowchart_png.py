#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Q2 / Q3 标准流程图 PNG
符号规范：
  椭圆   → 开始 / 结束
  平行四边形 → 输入 / 输出
  矩形   → 处理步骤
  菱形   → 判断 / 条件
"""
import os
os.chdir('/home/user/math_pro')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.font_manager import fontManager
import numpy as np

fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']        = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

# ── 颜色 ─────────────────────────────────────────────────────────────────────
C_START  = '#2C3E50'   # 开始/结束  深色
C_IO     = '#1A5276'   # 输入/输出  深蓝
C_PROC   = '#1F618D'   # 处理       蓝
C_DEC    = '#C0392B'   # 判断       红
C_LOOP   = '#117A65'   # 循环体     绿
C_WHITE  = '#FFFFFF'
C_ARROW  = '#2C3E50'

def _text(ax, cx, cy, txt, color=C_WHITE, fs=8.5, bold=False):
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
            color=color, fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=5)

def node_start(ax, cx, cy, w, h, txt, fs=9):
    """椭圆：开始/结束"""
    e = mpatches.Ellipse((cx, cy), w, h,
        facecolor=C_START, edgecolor=C_WHITE, lw=1.5, zorder=3)
    ax.add_patch(e)
    _text(ax, cx, cy, txt, fs=fs, bold=True)

def node_proc(ax, cx, cy, w, h, txt, color=C_PROC, fs=8.5):
    """矩形：处理步骤"""
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor='white', lw=1.2, zorder=3)
    ax.add_patch(r)
    _text(ax, cx, cy, txt, fs=fs)

def node_io(ax, cx, cy, w, h, txt, fs=8.5):
    """平行四边形：输入/输出"""
    sk = 0.15   # 斜度
    pts = np.array([
        [cx - w/2 + sk*h, cy + h/2],
        [cx + w/2 + sk*h, cy + h/2],
        [cx + w/2 - sk*h, cy - h/2],
        [cx - w/2 - sk*h, cy - h/2],
    ])
    poly = Polygon(pts, closed=True,
        facecolor=C_IO, edgecolor='white', lw=1.2, zorder=3)
    ax.add_patch(poly)
    _text(ax, cx, cy, txt, fs=fs)

def node_dec(ax, cx, cy, w, h, txt, fs=8):
    """菱形：判断"""
    pts = np.array([
        [cx,       cy+h/2],
        [cx+w/2,   cy    ],
        [cx,       cy-h/2],
        [cx-w/2,   cy    ],
    ])
    poly = Polygon(pts, closed=True,
        facecolor=C_DEC, edgecolor='white', lw=1.2, zorder=3)
    ax.add_patch(poly)
    _text(ax, cx, cy, txt, fs=fs)

def arr(ax, x0, y0, x1, y1, lbl='', lbl_side='right', lw=1.5,
        color=C_ARROW, fs=7.5):
    """直线箭头"""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=12), zorder=4)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx = 0.12 if lbl_side == 'right' else -0.12
        dy = 0.0
        if abs(x1-x0) < 0.01:   # 竖线 → 标签在右
            dx = 0.13; dy = 0.0
        elif abs(y1-y0) < 0.01:  # 横线 → 标签在上
            dx = 0.0; dy = 0.10
        ax.text(mx+dx, my+dy, lbl, ha='center', va='center',
                fontsize=fs, color='#333333', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.9))

def polyarr(ax, pts, lbl='', lbl_idx=0, lbl_side='top',
            lw=1.5, color=C_ARROW, fs=7.5):
    """折线箭头"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=lw, zorder=4)
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=12), zorder=4)
    if lbl:
        i = lbl_idx
        mx = (xs[i]+xs[i+1])/2
        my = (ys[i]+ys[i+1])/2
        dy = 0.10 if lbl_side == 'top' else -0.10
        dx = 0.0
        if abs(xs[i]-xs[i+1]) < 0.01:
            dx = 0.13; dy = 0.0
        ax.text(mx+dx, my+dy, lbl, ha='center', va='center',
                fontsize=fs, color='#333333', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 流程图
# ═══════════════════════════════════════════════════════════════════════════════
def gen_q2():
    fig, ax = plt.subplots(figsize=(11, 20))
    ax.set_xlim(0, 11); ax.set_ylim(0, 20)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    ax.text(5.5, 19.6, '问题二：XGBoost-PSO CO浓度预测模型  流程图',
            ha='center', va='center', fontsize=13, fontweight='bold')

    W, H = 4.2, 0.72   # 矩形默认宽高
    WD, HD = 3.0, 0.90  # 菱形
    cx = 5.5            # 中心 x

    # ── 节点 ─────────────────────────────────────────────────
    # S1 开始
    node_start(ax, cx, 18.9, 2.4, 0.6, '开  始')

    # I1 输入
    node_io(ax, cx, 18.0, W, H,
            '输入：烧结机传感器时序数据\n(机速 / 风箱负压 / 风箱温度 / CO浓度)')

    # P1 异常检测
    node_proc(ax, cx, 16.95, W, H,
              '异常数据检测与剔除\n(剔除样本 1045~1061 共17行)')

    # P2 滞后对齐
    node_proc(ax, cx, 15.9, W, H,
              '时间滞后对齐\n(读取 Q1 最优滞后量，shift(-lag))')

    # P3 特征工程
    node_proc(ax, cx, 14.85, W, H,
              '特征工程（84维）\n物理特征41 + 梯度特征34 + 统计特征4 + CO自回归5',
              color=C_LOOP)

    # P4 划分
    node_proc(ax, cx, 13.8, W, H,
              '划分训练集 / 测试集（7:3）\n标准化（StandardScaler）')

    # --- PSO 优化子流程 ---
    ax.text(cx, 13.1, '── PSO 超参数优化 ──',
            ha='center', va='center', fontsize=8.5, color='#888888')

    # P5 初始化粒子群
    node_proc(ax, cx, 12.45, W, H,
              '初始化粒子群\n(N=8粒子，随机位置/速度，搜索7维超参数空间)',
              color=C_LOOP)

    # P6 CV评估
    node_proc(ax, cx, 11.4, W, H,
              '计算每粒子适应度\n(TimeSeriesSplit 3折交叉验证，R²)',
              color=C_LOOP)

    # P7 更新pbest/gbest
    node_proc(ax, cx, 10.35, W, H,
              '更新个体最优 pbest\n& 全局最优 gbest',
              color=C_LOOP)

    # P8 更新速度/位置
    node_proc(ax, cx, 9.3, W, H,
              '更新粒子速度与位置\n(w=0.8, c1=c2=2.0，边界裁剪)',
              color=C_LOOP)

    # D1 判断迭代
    node_dec(ax, cx, 8.2, WD, HD, '迭代次数\n达到 10？')

    # P9 训练最终模型
    node_proc(ax, cx, 7.05, W, H,
              '以 gbest 超参数训练 XGBoost 模型\n(全训练集，verbosity=0)')

    # O1 输出预测
    node_io(ax, cx, 6.0, W, H,
            '测试集预测：y_pred = model.predict(X_test)')

    # P10 评估指标
    node_proc(ax, cx, 4.95, W, H,
              '计算评估指标\nR²  RMSE  MAE')

    # D2 性能判断
    node_dec(ax, cx, 3.85, WD, HD, 'R² ≥ 阈值\n(0.90)？')

    # O2 输出结果
    node_io(ax, cx, 2.75, W, H,
            '输出：CO浓度预测值\n保存模型参数与对比结果')

    # S2 结束
    node_start(ax, cx, 1.75, 2.4, 0.6, '结  束')

    # ── 连线 ─────────────────────────────────────────────────
    arr(ax, cx, 18.6, cx, 18.36)
    arr(ax, cx, 17.64, cx, 17.31)
    arr(ax, cx, 16.59, cx, 16.26)
    arr(ax, cx, 15.54, cx, 15.21)
    arr(ax, cx, 14.49, cx, 14.16)
    # 小标题间距
    arr(ax, cx, 13.44, cx, 12.81)
    arr(ax, cx, 12.09, cx, 11.76)
    arr(ax, cx, 11.04, cx, 10.71)
    arr(ax, cx, 10.0,  cx, 9.66)
    arr(ax, cx, 8.94,  cx, 8.65)

    # 菱形→P9 (是)
    arr(ax, cx, 7.75, cx, 7.41, lbl='是')

    # 菱形 否 → 回到P6 (右侧折回)
    polyarr(ax, [(cx+WD/2, 8.2), (9.2, 8.2), (9.2, 11.4), (cx+W/2, 11.4)],
            lbl='否', lbl_idx=1, lbl_side='top')

    arr(ax, cx, 6.64, cx, 6.36)
    arr(ax, cx, 5.64, cx, 5.31)
    arr(ax, cx, 4.39, cx, 3.10, lbl='是')

    # 否 → 回到P3特征工程（左侧折回，提示调整特征）
    polyarr(ax, [(cx-WD/2, 3.85), (1.8, 3.85), (1.8, 14.85), (cx-W/2, 14.85)],
            lbl='否(重新调整特征)', lbl_idx=1, lbl_side='top')

    arr(ax, cx, 2.39, cx, 2.05)

    plt.tight_layout(pad=0.4)
    plt.savefig('paper_figures/Q2_flowchart.png', dpi=170,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print('✓ paper_figures/Q2_flowchart.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 流程图
# ═══════════════════════════════════════════════════════════════════════════════
def gen_q3():
    fig, ax = plt.subplots(figsize=(11, 21))
    ax.set_xlim(0, 11); ax.set_ylim(0, 21)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    ax.text(5.5, 20.6, '问题三：PSO 风箱负压优化模型  流程图',
            ha='center', va='center', fontsize=13, fontweight='bold')

    W, H = 4.2, 0.72
    WD, HD = 3.2, 0.90
    cx = 5.5

    # ── 节点 ─────────────────────────────────────────────────
    node_start(ax, cx, 20.0, 2.4, 0.6, '开  始')

    node_io(ax, cx, 19.1, W, H,
            '输入：当前烧结机工况快照\n(风箱负压×18, 温度×18, 机速, 当前CO)')

    node_proc(ax, cx, 18.05, W, H,
              '载入 Q2 训练好的 XGBoost 模型\n及特征工程参数（滞后量/统计量）')

    node_proc(ax, cx, 17.0, W, H,
              '构造 84 维特征向量\n(物理 + 梯度 + 统计 + CO 自回归占位)')

    # 不动点初始化
    node_proc(ax, cx, 15.95, W, H,
              'CO 自回归特征不动点初始化\n令 co_lag = 当前 CO，代入模型迭代至收敛',
              color=C_LOOP)

    ax.text(cx, 15.22, '── PSO 负压优化搜索 ──',
            ha='center', va='center', fontsize=8.5, color='#888888')

    node_proc(ax, cx, 14.55, W, H,
              '初始化粒子群（N=30粒子）\n粒子位置 = 18维风箱负压，在约束范围内随机',
              color=C_LOOP)

    node_proc(ax, cx, 13.5, W, H,
              '对每个粒子构造完整特征向量\n用 XGBoost 模型预测 CO 浓度 y_pred',
              color=C_LOOP)

    # 约束检验
    node_dec(ax, cx, 12.4, WD, HD, '满足约束？\n(负压下界 ≤ p ≤ 上界)')

    node_proc(ax, cx, 11.3, W, H,
              '更新个体最优 pbest（最小化 CO）\n& 全局最优 gbest',
              color=C_LOOP)

    node_proc(ax, cx, 10.25, W, H,
              '更新粒子速度与位置\n(w衰减, c1=c2=2.0, 边界裁剪)',
              color=C_LOOP)

    node_dec(ax, cx, 9.1, WD, HD, '迭代次数\n达到上限？')

    # gbest
    node_proc(ax, cx, 7.95, W, H,
              '提取全局最优方案 gbest\n(18维最优风箱负压组合)')

    # 可行性校验
    node_dec(ax, cx, 6.85, WD, HD, '所有负压均\n在物理约束内？')

    node_proc(ax, cx, 5.75, W, H,
              '计算优化效果\nCO 降低量 / 降低比例 / 各风箱贡献度分析',
              color=C_LOOP)

    node_io(ax, cx, 4.7, W, H,
            '输出：最优风箱负压方案（18维）\n预测 CO = {val} ppm，降低 {pct}%')

    node_dec(ax, cx, 3.6, WD, HD, '优化效果\n满足目标？')

    node_io(ax, cx, 2.5, W, H,
            '下发负压指令至 DCS 控制系统\n记录优化结果，更新历史数据库')

    node_start(ax, cx, 1.5, 2.4, 0.6, '结  束')

    # ── 连线 ─────────────────────────────────────────────────
    arr(ax, cx, 19.7,  cx, 19.46)
    arr(ax, cx, 18.74, cx, 18.41)
    arr(ax, cx, 17.69, cx, 17.36)
    arr(ax, cx, 16.64, cx, 16.31)
    arr(ax, cx, 15.59, cx, 14.91)
    arr(ax, cx, 14.19, cx, 13.86)
    arr(ax, cx, 13.14, cx, 12.85)

    # 菱形"不满足约束" → 惩罚，右侧折回P6(预测CO)
    polyarr(ax, [(cx+WD/2, 12.4), (9.0, 12.4), (9.0, 13.5), (cx+W/2, 13.5)],
            lbl='不满足\n(惩罚)', lbl_idx=0)

    # 菱形"满足" → 更新pbest
    arr(ax, cx, 11.95, cx, 11.66, lbl='满足')
    arr(ax, cx, 10.94, cx, 10.61)
    arr(ax, cx, 9.79,  cx, 9.55)

    # 未达上限 → 折回预测CO
    polyarr(ax, [(cx-WD/2, 9.1), (1.8, 9.1), (1.8, 13.5), (cx-W/2, 13.5)],
            lbl='否', lbl_idx=1)

    # 达到上限
    arr(ax, cx, 8.65, cx, 8.31, lbl='是')
    arr(ax, cx, 7.59, cx, 7.30)

    # 不在约束内 → 重新PSO
    polyarr(ax, [(cx+WD/2, 6.85), (9.2, 6.85), (9.2, 14.55), (cx+W/2, 14.55)],
            lbl='否(扩大搜索)', lbl_idx=1)

    arr(ax, cx, 6.39, cx, 6.11, lbl='是')
    arr(ax, cx, 5.34, cx, 5.06)
    arr(ax, cx, 4.34, cx, 3.95)

    # 不满足目标 → 重新初始化PSO
    polyarr(ax, [(cx-WD/2, 3.6), (1.5, 3.6), (1.5, 14.55), (cx-W/2, 14.55)],
            lbl='否(重新优化)', lbl_idx=1)

    arr(ax, cx, 3.24, cx, 2.86, lbl='是')
    arr(ax, cx, 2.14, cx, 1.80)

    plt.tight_layout(pad=0.4)
    plt.savefig('paper_figures/Q3_flowchart.png', dpi=170,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print('✓ paper_figures/Q3_flowchart.png')


if __name__ == '__main__':
    gen_q2()
    gen_q3()
    print('\n全部完成。')
