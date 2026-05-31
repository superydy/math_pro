#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Q2 和 Q3 数据流图（DFD），Gane-Sarson 规范
输出：results/Q2_DFD_final.png, results/Q3_DFD_final.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
import numpy as np
import os

# ─── 字体 ──────────────────────────────────────────────────────────────────────
_fp_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
if os.path.exists(_fp_path):
    fm.fontManager.addfont(_fp_path)
    _fp = fm.FontProperties(fname=_fp_path)
    _FN = _fp.get_name()
else:
    _FN = 'DejaVu Sans'
    _fp = fm.FontProperties(family=_FN)

plt.rcParams['font.family'] = _FN
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('results', exist_ok=True)


# ─── 基础绘图原语 ───────────────────────────────────────────────────────────────

def draw_external_entity(ax, cx, cy, w, h, lines, fs=8.5):
    """外部实体：白底黑边矩形"""
    rect = plt.Rectangle((cx-w/2, cy-h/2), w, h,
                          linewidth=1.6, edgecolor='black', facecolor='white', zorder=3)
    ax.add_patch(rect)
    text = '\n'.join(lines) if isinstance(lines, list) else lines
    ax.text(cx, cy, text, ha='center', va='center', fontproperties=_fp, fontsize=fs,
            zorder=4, linespacing=1.4)


def draw_process(ax, cx, cy, w, h, lines, fs=8.5):
    """处理过程：白底圆角矩形"""
    fancy = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                           boxstyle='round,pad=0.12',
                           linewidth=1.6, edgecolor='#1a5276', facecolor='#eaf4fb', zorder=3)
    ax.add_patch(fancy)
    text = '\n'.join(lines) if isinstance(lines, list) else lines
    ax.text(cx, cy, text, ha='center', va='center', fontproperties=_fp, fontsize=fs,
            zorder=4, linespacing=1.4, color='#1a3c50')


def draw_datastore(ax, cx, cy, w, h, id_str, name_lines, fs=8.5):
    """数据存储：Gane-Sarson 开口矩形（灰底 + 上下边 + 左侧ID列）"""
    x = cx - w/2
    y = cy - h/2
    tab = w * 0.22

    bg = plt.Rectangle((x, y), w, h, linewidth=0, facecolor='#f0f0f0', zorder=3)
    ax.add_patch(bg)
    ax.plot([x, x+w], [y+h, y+h], 'k-', lw=1.6, zorder=4)
    ax.plot([x, x+w], [y,   y  ], 'k-', lw=1.6, zorder=4)
    ax.plot([x+tab, x+tab], [y, y+h], 'k-', lw=1.1, zorder=4)

    ax.text(x+tab/2, cy, id_str, ha='center', va='center',
            fontproperties=_fp, fontsize=fs, fontweight='bold', zorder=5)
    nm = '\n'.join(name_lines) if isinstance(name_lines, list) else name_lines
    ax.text(x+tab+(w-tab)/2, cy, nm, ha='center', va='center',
            fontproperties=_fp, fontsize=fs, zorder=5, linespacing=1.4)


def draw_arrow(ax, x1, y1, x2, y2, label='', label_seg=None, fs=7.5, color='#2c3e50'):
    """绘制箭头（折线 or 直线），label_seg指定标注在哪段（索引元组）"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.1,
                                mutation_scale=12))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center', fontproperties=_fp,
                fontsize=fs, zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))


def draw_polyline_arrow(ax, pts, label='', label_idx=0, fs=7.5, color='#2c3e50'):
    """折线箭头，pts 为 [(x,y),...] 列表"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # 绘制折线段（除最后一段用带箭头的annotate）
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], '-', color=color, lw=1.1, zorder=2)
    # 最后一段用箭头
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.1, mutation_scale=12))
    # 标注
    if label:
        i = min(label_idx, len(pts)-2)
        mx = (xs[i]+xs[i+1])/2
        my = (ys[i]+ys[i+1])/2
        ax.text(mx, my, label, ha='center', va='center', fontproperties=_fp,
                fontsize=fs, zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))


def add_legend(ax, x0, y0, gap=1.5):
    """图例行"""
    draw_external_entity(ax, x0, y0, 1.4, 0.55, '外部实体', fs=7)
    ax.text(x0, y0-0.42, '外部实体', ha='center', va='top', fontproperties=_fp, fontsize=6.5)

    draw_process(ax, x0+gap, y0, 1.8, 0.55, 'Px 处理过程', fs=7)
    ax.text(x0+gap, y0-0.42, '处理过程', ha='center', va='top', fontproperties=_fp, fontsize=6.5)

    draw_datastore(ax, x0+gap*2.2, y0, 2.6, 0.55, 'Dx', '数据存储', fs=7)
    ax.text(x0+gap*2.2, y0-0.42, '数据存储', ha='center', va='top', fontproperties=_fp, fontsize=6.5)

    ax.annotate('', xy=(x0+gap*3.5+0.7, y0), xytext=(x0+gap*3.5, y0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.1, mutation_scale=12))
    ax.text(x0+gap*3.5+0.35, y0+0.2, '数据流', ha='center', fontproperties=_fp, fontsize=6.5)


# ══════════════════════════════════════════════════════════════════════════════
# Q2 DFD
# ══════════════════════════════════════════════════════════════════════════════

def gen_q2_dfd():
    fig, ax = plt.subplots(figsize=(17, 11))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 17)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.text(12, 16.5, '问题二：CO浓度预测模型建立  数据流图（Gane-Sarson规范）',
            ha='center', va='center', fontproperties=_fp, fontsize=13, fontweight='bold')

    EW, EH = 2.8, 1.0
    PW, PH = 3.4, 1.4
    SW, SH = 5.0, 1.0

    # 坐标（中心）
    EE1 = (2.0, 14.5)
    EE2 = (20.0, 1.8)

    P1 = (9.0, 14.5)
    P2 = (9.0, 11.5)
    P3 = (5.5,  8.5)
    P4 = (12.5,  8.5)
    P5 = (9.0,  5.5)

    D1 = (20.0, 14.5)
    D2 = ( 2.2, 11.5)
    D3 = (20.0, 11.5)
    D4 = (20.0,  8.5)

    # 外部实体
    draw_external_entity(ax, *EE1, EW, EH, ['烧结', '生产系统'])
    draw_external_entity(ax, *EE2, EW, EH, ['模型', '使用方'])

    # 处理过程
    draw_process(ax, *P1, PW, PH, ['P1', '数据采集与预处理'])
    draw_process(ax, *P2, PW, PH, ['P2', '特征工程（84维）'])
    draw_process(ax, *P3, PW, PH, ['P3', 'PSO超参数', '寻优（8×10）'])
    draw_process(ax, *P4, PW, PH, ['P4', 'XGBoost', '模型训练'])
    draw_process(ax, *P5, PW, PH, ['P5', '模型评估验证'])

    # 数据存储
    draw_datastore(ax, *D1, SW, SH, 'D1', '原始工况数据库')
    draw_datastore(ax, *D2, SW, SH, 'D2', ['清洗后数据集', '（2315条）'])
    draw_datastore(ax, *D3, SW, SH, 'D3', ['84维特征矩阵', '(物理+梯度+统计+CO自回归)'])
    draw_datastore(ax, *D4, SW, SH, 'D4', ['XGBoost预测模型', 'R²=0.9388(测试集)'])

    # 数据流
    draw_arrow(ax, EE1[0]+EW/2, EE1[1], P1[0]-PW/2, P1[1],
               '原始压力/温度/速度/CO记录')

    draw_arrow(ax, P1[0]+PW/2, P1[1], D1[0]-SW/2, D1[1], '写入原始记录')

    draw_arrow(ax, P1[0]-PW/4, P1[1]-PH/2, D2[0]+SW*0.35, D2[1]+SH/2,
               '2315条清洗后记录')

    draw_arrow(ax, D2[0]+SW/2, D2[1], P2[0]-PW/2, P2[1], '读取清洗数据')

    draw_arrow(ax, P2[0]+PW/2, P2[1], D3[0]-SW/2, D3[1],
               '84维特征矩阵')

    draw_arrow(ax, D3[0]-SW/2, D3[1]-SH/4, P3[0]+PW/2, P3[1]+PH/4,
               '特征集(CV折叠)')

    draw_arrow(ax, D3[0]-SW/4, D3[1]-SH/2, P4[0]+PW/4, P4[1]+PH/2, '训练特征集')

    draw_arrow(ax, P3[0]+PW/2, P3[1], P4[0]-PW/2, P4[1],
               'lr=0.206,depth=3,n=262')

    draw_arrow(ax, P4[0]+PW/2, P4[1], D4[0]-SW/2, D4[1], '保存训练完成模型')

    draw_arrow(ax, D4[0]-SW/4, D4[1]-SH/2, P5[0]+PW/3, P5[1]+PH/2, '调用预测模型')

    # D2 → P5（折线：向下 → 右折 → 进P5）
    draw_polyline_arrow(ax,
        [(D2[0], D2[1]-SH/2), (D2[0], 4.2), (P5[0]-PW/2, 4.2), (P5[0]-PW/2, P5[1]-PH/2)],
        '测试集(695条)', label_idx=1)

    draw_arrow(ax, P5[0]+PW/2, P5[1], EE2[0]-EW/2, EE2[1],
               'R²=0.9388/MAE=50.1 评估报告')

    draw_arrow(ax, D4[0], D4[1]-SH/2, EE2[0], EE2[1]+EH/2, '部署XGBoost预测模型')

    # 图例
    add_legend(ax, 0.8, 0.7, 1.55)

    # 注释框（随机森林分析）
    ax.text(2.2, 8.5,
            'RF重要性分析:\n温度58.5%/负压25.2%\nRidge方向分析:\n大烟道负压-241',
            ha='center', va='center', fontproperties=_fp, fontsize=6.5,
            bbox=dict(boxstyle='round', fc='#fffde7', ec='#f0c040', lw=1))
    ax.annotate('', xy=(P2[0]-PW/2, P2[1]+0.1), xytext=(2.2+0.8, 8.9),
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.9,
                                linestyle='dashed', mutation_scale=10))

    plt.tight_layout(pad=0.3)
    out = 'results/Q2_DFD_final.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✓ 已保存: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Q3 DFD
# ══════════════════════════════════════════════════════════════════════════════

def gen_q3_dfd():
    fig, ax = plt.subplots(figsize=(17, 13))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 22)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.text(12, 21.4, '问题三：PSO风箱负压优化调控  数据流图（Gane-Sarson规范）',
            ha='center', va='center', fontproperties=_fp, fontsize=13, fontweight='bold')

    EW, EH = 3.0, 1.1
    PW, PH = 3.8, 1.5
    SW, SH = 5.4, 1.0

    EE1 = (2.0, 20.0)
    EE2 = (20.0, 1.5)

    P1 = (9.5, 20.0)
    P2 = (9.5, 16.5)
    P3 = (9.5, 13.0)
    P4 = (9.5,  9.5)
    P5 = (9.5,  6.0)
    P6 = (9.5,  2.8)

    D1 = (20.0, 20.0)
    D2 = ( 2.0, 16.5)
    D3 = (20.0, 16.5)
    D4 = (20.0, 13.0)
    D5 = (20.0,  9.5)
    D6 = (20.0,  6.0)

    # 外部实体
    draw_external_entity(ax, *EE1, EW, EH, ['工况', '监测系统'])
    draw_external_entity(ax, *EE2, EW, EH, ['压力调控', '执行系统'])

    # 处理过程
    draw_process(ax, *P1, PW, PH, ['P1', '稳态CO计算', '(不动点迭代α=0.4×80次)'])
    draw_process(ax, *P2, PW, PH, ['P2', '调节边界确定', '(10%~90%分位数截取)'])
    draw_process(ax, *P3, PW, PH, ['P3', 'PSO粒子群初始化', '(40粒子×18维)'])
    draw_process(ax, *P4, PW, PH, ['P4', '稳态适应度评估', '+可靠性惩罚'])
    draw_process(ax, *P5, PW, PH, ['P5', '自适应粒子更新', '(w:0.9→0.4, α:50→500)'])
    draw_process(ax, *P6, PW, PH, ['P6', '收敛判断 & 输出最优解'])

    # 数据存储
    draw_datastore(ax, *D1, SW, SH, 'D1', ['稳态CO基准值', '(3495.4 mg/m³)'])
    draw_datastore(ax, *D2, SW, SH, 'D2', ['历史压力分位数表', '(10%~90%, 18维)'])
    draw_datastore(ax, *D3, SW, SH, 'D3', ['18维压力调节范围', '[下限, 上限]'])
    draw_datastore(ax, *D4, SW, SH, 'D4', ['Q2 XGBoost预测模型', '(R²=0.9388)'])
    draw_datastore(ax, *D5, SW, SH, 'D5', ['粒子群状态矩阵', '(40×18位置+速度)'])
    draw_datastore(ax, *D6, SW, SH, 'D6', ['全局最优解记录', '(最优压力+最优CO)'])

    # ── 数据流 ────────────────────────────────────────────────────────────────

    # EE1 → P1（当前负压）
    draw_arrow(ax, EE1[0]+EW/2, EE1[1], P1[0]-PW/2, P1[1], '18个风箱当前工况')

    # D4 → P1（CO预测函数，折线绕行）
    draw_polyline_arrow(ax,
        [(D4[0]-SW/2, D4[1]),
         (D4[0]-SW/2-1.0, D4[1]),
         (D4[0]-SW/2-1.0, P1[1]+PH/2+0.4),
         (P1[0]+PW/2, P1[1]+PH/2+0.4),
         (P1[0]+PW/2, P1[1])],
        'CO预测函数', label_idx=2)

    # P1 → D1
    draw_arrow(ax, P1[0]+PW/2, P1[1], D1[0]-SW/2, D1[1], '稳态CO=3495.4 mg/m³')

    # D1 → P2（折线）
    draw_polyline_arrow(ax,
        [(D1[0]-SW/2, D1[1]),
         (D1[0]-SW/2-0.6, D1[1]),
         (D1[0]-SW/2-0.6, P2[1]+PH/2+0.35),
         (P2[0]+PW/2, P2[1]+PH/2+0.35),
         (P2[0]+PW/2, P2[1])],
        '稳态CO基准值', label_idx=2)

    # D2 → P2
    draw_arrow(ax, D2[0]+SW/2, D2[1], P2[0]-PW/2, P2[1], '18维历史分位数')

    # EE1 → P2（折线：经工况监测下行至P2左侧）
    draw_polyline_arrow(ax,
        [(EE1[0]+EW/4, EE1[1]-EH/2),
         (EE1[0]+EW/4, P2[1]+PH/2+0.8),
         (P2[0]-PW/2, P2[1]+PH/2+0.8),
         (P2[0]-PW/2, P2[1])],
        '当前工况参考', label_idx=1)

    # P2 → D3
    draw_arrow(ax, P2[0]+PW/2, P2[1], D3[0]-SW/2, D3[1], '18维压力[下限,上限]')

    # D3 → P3（折线）
    draw_polyline_arrow(ax,
        [(D3[0]-SW/2, D3[1]),
         (D3[0]-SW/2-0.7, D3[1]),
         (D3[0]-SW/2-0.7, P3[1]+PH/2+0.35),
         (P3[0]+PW/2, P3[1]+PH/2+0.35),
         (P3[0]+PW/2, P3[1])],
        '调节范围边界', label_idx=2)

    # P3 → D5
    draw_arrow(ax, P3[0]+PW/2, P3[1], D5[0]-SW/2, D5[1]+SH/3,
               '40×18初始位置+速度')

    # D5 → P4
    draw_arrow(ax, D5[0]-SW/2, D5[1], P4[0]+PW/2, P4[1], '当前粒子位置(18维压力)')

    # D4 → P4
    draw_arrow(ax, D4[0]-SW/2, D4[1], P4[0]+PW/2, P4[1]-PH/4, 'XGBoost CO预测')

    # P4 → P5
    draw_arrow(ax, P4[0], P4[1]-PH/2, P5[0], P5[1]+PH/2,
               'CO_稳态+α·Σexp(-10d)')

    # P5 → D5（更新粒子）
    draw_arrow(ax, P5[0]+PW/2, P5[1]+PH/4, D5[0]-SW/2, D5[1]-SH/4, '更新粒子位置与速度')

    # P5 → D6
    draw_arrow(ax, P5[0]+PW/2, P5[1], D6[0]-SW/2, D6[1], '写入新全局最优')

    # D6 → P6
    draw_arrow(ax, D6[0]-SW/2, D6[1], P6[0]+PW/2, P6[1], '读取当前最优解')

    # P6 → P4（反馈回路，左侧折线）
    loop_x = 3.2
    draw_polyline_arrow(ax,
        [(P6[0]-PW/2, P6[1]),
         (loop_x, P6[1]),
         (loop_x, P4[1]),
         (P4[0]-PW/2, P4[1])],
        '未达150次迭代/继续评估', label_idx=1)

    # P6 → EE2（输出）
    draw_arrow(ax, P6[0]+PW/2, P6[1], EE2[0]-EW/2, EE2[1],
               '最优18维负压/CO=1299.5 mg/m³/减排62.8%')

    # 图例
    add_legend(ax, 0.8, 0.5, 1.55)

    plt.tight_layout(pad=0.3)
    out = 'results/Q3_DFD_final.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✓ 已保存: {out}')


if __name__ == '__main__':
    gen_q2_dfd()
    gen_q3_dfd()
    print('全部DFD生成完成')
