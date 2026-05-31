#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 PSO（超参数调优）和 Q3 PSO（负压优化）流程图
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
C_START = '#1C2833'
C_IO    = '#154360'
C_PROC  = '#1A5276'
C_DEC   = '#922B21'
C_LOOP  = '#0E6655'
C_INIT  = '#6C3483'
C_WHITE = '#FFFFFF'
C_ARROW = '#2C3E50'

def _txt(ax, cx, cy, s, fc=C_WHITE, fs=8.2, bold=False):
    ax.text(cx, cy, s, ha='center', va='center', fontsize=fs,
            color=fc, fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=5)

def oval(ax, cx, cy, w, h, txt, color=C_START, fs=9):
    e = mpatches.Ellipse((cx, cy), w, h,
        facecolor=color, edgecolor='white', lw=1.5, zorder=3)
    ax.add_patch(e)
    _txt(ax, cx, cy, txt, fs=fs, bold=True)

def rect(ax, cx, cy, w, h, txt, color=C_PROC, fs=8.2):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle='round,pad=0.04',
        facecolor=color, edgecolor='white', lw=1.2, zorder=3)
    ax.add_patch(r)
    _txt(ax, cx, cy, txt, fs=fs)

def para(ax, cx, cy, w, h, txt, color=C_IO, fs=8.2):
    sk = 0.12
    pts = np.array([
        [cx-w/2+sk*h, cy+h/2],
        [cx+w/2+sk*h, cy+h/2],
        [cx+w/2-sk*h, cy-h/2],
        [cx-w/2-sk*h, cy-h/2],
    ])
    ax.add_patch(Polygon(pts, closed=True,
        facecolor=color, edgecolor='white', lw=1.2, zorder=3))
    _txt(ax, cx, cy, txt, fs=fs)

def diamond(ax, cx, cy, w, h, txt, color=C_DEC, fs=7.8):
    pts = np.array([[cx,cy+h/2],[cx+w/2,cy],[cx,cy-h/2],[cx-w/2,cy]])
    ax.add_patch(Polygon(pts, closed=True,
        facecolor=color, edgecolor='white', lw=1.2, zorder=3))
    _txt(ax, cx, cy, txt, fs=fs)

def arr(ax, x0,y0,x1,y1, lbl='', lbl_side='right', fs=7.5, lw=1.5, color=C_ARROW):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=11),
        zorder=4)
    if lbl:
        mx,my=(x0+x1)/2,(y0+y1)/2
        dx=0.12 if lbl_side=='right' else -0.12; dy=0.0
        if abs(x1-x0)<0.01: dx=0.13; dy=0.0
        elif abs(y1-y0)<0.01: dx=0.0; dy=0.10
        ax.text(mx+dx,my+dy,lbl,ha='center',va='center',fontsize=fs,
                color='#222222',zorder=6,
                bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.9))

def parr(ax, pts, lbl='', li=0, ls='top', lw=1.5, fs=7.5, color=C_ARROW):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    for i in range(len(pts)-2):
        ax.plot([xs[i],xs[i+1]],[ys[i],ys[i+1]],color=color,lw=lw,zorder=4)
    ax.annotate('',xy=(xs[-1],ys[-1]),xytext=(xs[-2],ys[-2]),
        arrowprops=dict(arrowstyle='->',color=color,lw=lw,mutation_scale=11),zorder=4)
    if lbl:
        mx=(xs[li]+xs[li+1])/2; my=(ys[li]+ys[li+1])/2
        dy=0.09 if ls=='top' else -0.09; dx=0.0
        if abs(xs[li]-xs[li+1])<0.01: dx=0.13; dy=0.0
        ax.text(mx+dx,my+dy,lbl,ha='center',va='center',fontsize=fs,color='#222222',
                zorder=6,bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 PSO：超参数调优流程
# ═══════════════════════════════════════════════════════════════════════════════
def gen_q2_pso():
    fig, ax = plt.subplots(figsize=(9, 18))
    ax.set_xlim(0, 9); ax.set_ylim(0, 18)
    ax.axis('off'); fig.patch.set_facecolor('#F8F9FA')

    ax.text(4.5, 17.65, 'Q2  PSO 超参数优化流程',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(4.5, 17.25, '目标：最大化 XGBoost 交叉验证 R²  |  搜索空间：7 维超参数\nPSO内部用 3折CV（加速）；最终评估用 5折CV（论文指标）',
            ha='center', va='center', fontsize=8.5, color='#555555')

    W, H  = 5.8, 0.72
    WD,HD = 4.2, 0.88
    cx    = 4.5

    # 节点
    oval(ax, cx, 16.7, 2.6, 0.6, '开  始')

    para(ax, cx, 15.85, W, H,
         '输入：训练集 X_tr_s（1624×84）/ y_tr\n超参数搜索范围（7维下界/上界）')

    rect(ax, cx, 14.85, W, H,
         '初始化粒子群\nN=8 粒子，在 [lb, ub] 内随机生成位置\n速度初始化为 0', color=C_INIT)

    rect(ax, cx, 13.75, W, H,
         '对每个粒子，用当前超参数训练 XGBoost\nTimeSeriesSplit 3折 CV（内部加速用）\n计算平均 R² 作为粒子适应度',
         color=C_LOOP)

    rect(ax, cx, 12.65, W, H,
         '更新个体最优 pbest\n若 R²(当前) > R²(pbest)  →  pbest = 当前位置',
         color=C_LOOP)

    rect(ax, cx, 11.55, W, H,
         '更新全局最优 gbest\n若 R²(当前) > R²(gbest)  →  gbest = 当前位置',
         color=C_LOOP)

    rect(ax, cx, 10.45, W, H,
         '更新速度与位置\nv = w·v + c1·r1·(pbest-x) + c2·r2·(gbest-x)\nx = clip(x + v,  lb,  ub)\nw=0.8, c1=c2=2.0（固定）',
         color=C_LOOP)

    diamond(ax, cx, 9.25, WD, HD, '迭代次数\n达到 10？')

    rect(ax, cx, 8.1, W, H,
         '以 gbest 超参数重新训练 XGBoost\n（全训练集）\n再做 TimeSeriesSplit 5折 CV 最终评估')

    para(ax, cx, 7.05, W, H,
         '输出：最优超参数组合（7维）\n5折 CV 平均 R²（论文报告指标）\n最终预测模型')

    oval(ax, cx, 6.15, 2.6, 0.6, '结  束')

    # 参数注释框
    ax.text(0.2, 10.5,
            '固定参数\n─────────\nw = 0.8\nc1 = 2.0\nc2 = 2.0\nN = 8粒子\nT = 10轮',
            fontsize=7.5, va='top', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', fc='#EBF5FB', ec='#AED6F1', lw=1))

    ax.text(6.5, 10.5,
            '搜索范围\n─────────\nlr: [0.01, 0.50]\ndepth: [2, 8]\nn_est: [50, 500]\nsub: [0.5, 1.0]\ncol: [0.5, 1.0]\nalpha: [0.0, 2.0]\nlambda: [0.0, 5.0]',
            fontsize=7.2, va='top', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', fc='#EAFAF1', ec='#A9DFBF', lw=1))

    # 连线
    arr(ax, cx,16.40, cx,16.21)
    arr(ax, cx,15.49, cx,15.21)
    arr(ax, cx,14.49, cx,14.11)
    arr(ax, cx,13.39, cx,13.01)
    arr(ax, cx,12.29, cx,11.91)
    arr(ax, cx,11.19, cx,10.81)
    arr(ax, cx,10.09, cx, 9.69)
    arr(ax, cx, 8.81, cx, 8.46, lbl='是')
    arr(ax, cx, 7.74, cx, 7.41)
    arr(ax, cx, 6.69, cx, 6.45)
    # 否 → 回到"对每个粒子评估"
    parr(ax, [(cx+WD/2, 9.25),(7.8, 9.25),(7.8,13.75),(cx+W/2,13.75)],
         lbl='否', li=1, ls='top')

    plt.tight_layout(pad=0.4)
    plt.savefig('paper_figures/Q2_PSO_flowchart.png', dpi=170,
                bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print('✓ paper_figures/Q2_PSO_flowchart.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 PSO：风箱负压优化流程
# ═══════════════════════════════════════════════════════════════════════════════
def gen_q3_pso():
    fig, ax = plt.subplots(figsize=(9, 22))
    ax.set_xlim(0, 9); ax.set_ylim(0, 22)
    ax.axis('off'); fig.patch.set_facecolor('#F8F9FA')

    ax.text(4.5, 21.65, 'Q3  PSO 负压优化流程',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(4.5, 21.2,
            '目标：最小化预测 CO + 可靠性惩罚\n搜索空间：18 维风箱负压  |  自适应参数 + 动态惩罚权重',
            ha='center', va='center', fontsize=8.5, color='#555555')

    W, H  = 5.8, 0.72
    WD,HD = 4.4, 0.90
    cx    = 4.5

    oval(ax, cx, 20.65, 2.6, 0.6, '开  始')

    para(ax, cx, 19.8, W, H,
         '输入：当前工况快照（18路负压/温度/机速）\n载入 Q2 XGBoost 模型 + 标准化参数')

    rect(ax, cx, 18.75, W, H,
         'CO 自回归不动点初始化\n令 co_lag1/2/5 = co_init，迭代 50 次\nX_{t+1} = 0.5·X_t + 0.5·f(X_t) 直至收敛',
         color=C_INIT)

    rect(ax, cx, 17.65, W, H,
         '确定 18 维搜索边界\n各风箱负压的 [10%分位数, 90%分位数]\n初始化粒子：中心±30%范围内随机，N=40',
         color=C_INIT)

    rect(ax, cx, 16.55, W, H,
         '计算自适应参数（按迭代进度 ratio=t/T）\nratio<0.3 → w=0.9, c1=2.0, c2=1.5  （探索）\n0.3≤ratio<0.7 → w=0.7, c1=1.5, c2=1.5  （平衡）\nratio≥0.7 → w=0.4, c1=1.0, c2=2.0  （开发）\n动态惩罚权重 α = 50 + 450·ratio',
         color=C_LOOP)

    rect(ax, cx, 15.35, W, H,
         '对每个粒子调用不动点法计算稳态 CO\n更新 co_lag 后代入模型迭代至收敛',
         color=C_LOOP)

    rect(ax, cx, 14.3, W, H,
         '计算可靠性惩罚项\npenalty = Σ exp(-10 × 归一化边界距离)\n综合目标 = CO_pred + α × penalty',
         color=C_LOOP)

    diamond(ax, cx, 13.1, WD, HD, '满足约束？\n（负压在 [lo, hi] 内）')

    rect(ax, cx, 11.9, W, H,
         '更新个体最优 pbest\n若 obj(当前) < obj(pbest)  →  pbest = 当前位置',
         color=C_LOOP)

    rect(ax, cx, 10.8, W, H,
         '更新全局最优 gbest\n若 obj(当前) < obj(gbest)  →  gbest = 当前位置',
         color=C_LOOP)

    rect(ax, cx, 9.7, W, H,
         '更新速度与位置\nv = w·v + c1·r1·(pbest-x) + c2·r2·(gbest-x)\nx = clip(x + v,  lo,  hi)',
         color=C_LOOP)

    diamond(ax, cx, 8.5, WD, HD, '迭代次数\n达到 150？')

    rect(ax, cx, 7.35, W, H,
         '提取 gbest：最优 18 维风箱负压组合\n计算 CO 降低量与降低比例')

    diamond(ax, cx, 6.2, WD, HD, '所有负压\n在物理约束内？')

    rect(ax, cx, 5.05, W, H,
         '输出最优负压方案\n计算各风箱贡献度（敏感性分析）')

    para(ax, cx, 4.0, W, H,
         '输出：最优负压指令（18维）\n预测 CO 降低量 / 降低比例 / 敏感风箱排名')

    oval(ax, cx, 3.1, 2.6, 0.6, '结  束')

    # 对比注释框（左侧）
    ax.text(0.1, 17.0,
            'Q3 vs Q2\nPSO 对比\n──────────\n维度: 18 vs 7\n粒子: 40 vs 8\n迭代: 150 vs 10\n参数: 自适应\n      vs 固定\n惩罚: 有 vs 无\n目标: min CO\n      vs max R²',
            fontsize=7.2, va='top', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', fc='#FEF9E7', ec='#F9E79F', lw=1))

    # 连线
    arr(ax, cx,20.35, cx,20.16)
    arr(ax, cx,19.44, cx,19.11)
    arr(ax, cx,18.39, cx,18.01)
    arr(ax, cx,17.29, cx,16.91)
    arr(ax, cx,16.19, cx,15.71)
    arr(ax, cx,14.94, cx,14.66)  # 评估 → 惩罚
    arr(ax, cx,13.94, cx,13.55)  # 惩罚 → 约束菱形
    arr(ax, cx,12.65, cx,12.26, lbl='满足')  # 菱形 → pbest
    arr(ax, cx,11.54, cx,11.16)
    arr(ax, cx,10.44, cx,10.06)
    arr(ax, cx, 9.34, cx, 8.95)
    arr(ax, cx, 8.05, cx, 7.71, lbl='是')
    arr(ax, cx, 6.99, cx, 6.65)
    arr(ax, cx, 5.75, cx, 5.41, lbl='是')
    arr(ax, cx, 4.69, cx, 4.36)
    arr(ax, cx, 3.64, cx, 3.40)

    # 约束不满足 → 重新计算（右折回）
    parr(ax, [(cx+WD/2,13.1),(7.7,13.1),(7.7,15.35),(cx+W/2,15.35)],
         lbl='不满足(惩罚)', li=1, ls='top')

    # 未达迭代上限 → 回到自适应参数
    parr(ax, [(cx-WD/2, 8.5),(1.3, 8.5),(1.3,16.55),(cx-W/2,16.55)],
         lbl='否', li=1, ls='top')

    # 不在物理约束 → 扩大搜索
    parr(ax, [(cx+WD/2, 6.2),(7.7, 6.2),(7.7,17.65),(cx+W/2,17.65)],
         lbl='否(扩边界重搜)', li=1, ls='top')

    plt.tight_layout(pad=0.4)
    plt.savefig('paper_figures/Q3_PSO_flowchart.png', dpi=170,
                bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print('✓ paper_figures/Q3_PSO_flowchart.png')


if __name__ == '__main__':
    gen_q2_pso()
    gen_q3_pso()
    print('\n完成。')
