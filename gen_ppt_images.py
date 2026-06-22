#!/usr/bin/env python3
"""生成PPT配套科学图表：地层剖面、时间轴、区位示意"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
import numpy as np
import os

OUT = '/home/user/math_pro/ppt_images'
os.makedirs(OUT, exist_ok=True)

# 尝试使用中文字体
plt.rcParams['font.family'] = 'sans-serif'
font_found = False
for font in ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']:
    result = fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
    if result and 'ttf' in result.lower() or 'ttc' in result.lower():
        plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
        font_found = True
        break
if not font_found:
    import glob as _glob
    wqy = _glob.glob('/usr/share/fonts/truetype/wqy/*.ttc')
    if wqy:
        fm.fontManager.addfont(wqy[0])
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BG    = '#100A04'
DARK2 = '#1C1208'
BRONZE= '#C8942A'
BRONZE2='#D8A438'
BRONZE3='#E8B850'
CREAM = '#F0E6CC'
CREAM2= '#C8B898'
STONE = '#786850'

# ═══════════════════════════════════════════════
# 图1：泥河湾地质地层剖面图
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(DARK2)

layers = [
    (0.0,  0.5,  '#C8A060', '#A07830', '现代土壤层',      '0 - 0.5m',     '全新世'),
    (0.5,  1.2,  '#B89050', '#906020', '黄土堆积层',      '0.5 - 1.7m',   '晚更新世\n约1-10万年前'),
    (1.7,  1.5,  '#A08040', '#785010', '红色黏土层',      '1.7 - 3.2m',   '中更新世\n约10-78万年前'),
    (3.2,  2.0,  '#887060', '#604030', '泥河湾层（主）',  '3.2 - 5.2m',   '早更新世\n约78-258万年前\n★ 石器主要出土层'),
    (5.2,  1.8,  '#6B5040', '#483020', '古湖相沉积层',    '5.2 - 7.0m',   '上新世末\n约258-500万年前'),
    (7.0,  1.0,  '#483020', '#301808', '基岩（玄武岩）',  '7.0m以下',     '前上新世'),
]

y_pos = 0
for (depth, thick, color, edge, name, range_str, age) in layers:
    # 地层色块
    bar = mpatches.FancyBboxPatch((0.05, y_pos), 0.55, thick*0.85,
        boxstyle='square,pad=0', facecolor=color,
        edgecolor=edge, linewidth=1.5)
    ax.add_patch(bar)
    # 纹理线（地层感）
    for k in range(int(thick*4)):
        y_line = y_pos + k * thick*0.85/max(int(thick*4),1)
        ax.plot([0.05, 0.6], [y_line, y_line],
                color=edge, alpha=0.3, linewidth=0.4)
    # 地层名称
    ax.text(0.63, y_pos + thick*0.85/2, name,
            va='center', ha='left', fontsize=11, fontweight='bold',
            color=BRONZE3 if '泥河湾' in name else CREAM)
    ax.text(0.63, y_pos + thick*0.85/2 - 0.18, range_str,
            va='center', ha='left', fontsize=9, color=STONE)
    # 年代（右侧）
    ax.text(1.35, y_pos + thick*0.85/2, age,
            va='center', ha='left', fontsize=9, color=CREAM2,
            linespacing=1.4)
    y_pos += thick * 0.85

total_h = y_pos

# 深度标尺
for d, label in [(0,'0m'),(1,'2m'),(2,'4m'),(3,'6m'),(3.5,'7m+')]:
    ax.plot([0.02, 0.06], [d*total_h/3.6, d*total_h/3.6],
            color=BRONZE, linewidth=1)
    ax.text(0.01, d*total_h/3.6, label,
            va='center', ha='right', fontsize=8, color=CREAM2)

# 石器标记
star_y = layers[3][0]
bar_thick = layers[3][1]
star_center = (layers[3][0]*0.85 + layers[3][1]*0.85*0.5) - 0.05
ax.annotate('马圈沟石器\n166万年前', xy=(0.6, total_h*0.62),
            xytext=(0.62, total_h*0.52),
            fontsize=9, color=BRONZE3, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BRONZE, lw=1.5))

ax.set_xlim(-0.05, 2.2)
ax.set_ylim(-0.2, total_h + 0.3)
ax.axis('off')
ax.set_title('泥河湾盆地地质地层剖面示意图', fontsize=14,
             fontweight='bold', color=BRONZE3, pad=12)
fig.tight_layout()
fig.savefig(f'{OUT}/01_地质地层剖面.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print('✓ 图1 地质地层剖面')

# ═══════════════════════════════════════════════
# 图2：时间轴——166万年人类史
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

events = [
    (300, '300万年前\n泥河湾盆地断陷\n形成大型古湖泊',   STONE),
    (200, '200万年前\n湖岸温暖湿润\n动植物极为丰富',     STONE),
    (166, '166万年前\n东亚最早先民\n在此打制石器',       BRONZE3),
    (78,  '78万年前\n红色黏土堆积\n气候逐渐变冷',        STONE),
    (10,  '10万年前\n晚更新世人类\n继续在此活动',        STONE),
    (0.1, '近现代\n考古发掘揭示\n文明起源密码',          BRONZE),
]

# 主轴线
ax.plot([0.1, 12.8], [2.5, 2.5], color=BRONZE2, linewidth=2, zorder=1)

for i, (year, label, color) in enumerate(events):
    x = 0.5 + i * 2.45
    # 节点圆
    circle = plt.Circle((x, 2.5), 0.22, color=color,
                         zorder=3, ec=DARK2, lw=2)
    ax.add_patch(circle)
    # 竖线
    if i % 2 == 0:
        ax.plot([x, x], [2.72, 3.6], color=color, linewidth=1.2, zorder=2)
        ax.text(x, 3.7, label, ha='center', va='bottom',
                fontsize=8.5, color=CREAM if color==BRONZE3 else CREAM2,
                fontweight='bold' if color==BRONZE3 else 'normal',
                linespacing=1.5)
    else:
        ax.plot([x, x], [1.3, 2.28], color=color, linewidth=1.2, zorder=2)
        ax.text(x, 1.1, label, ha='center', va='top',
                fontsize=8.5, color=CREAM if color==BRONZE3 else CREAM2,
                linespacing=1.5)

ax.set_xlim(0, 13.3)
ax.set_ylim(-0.3, 6.2)
ax.axis('off')
ax.set_title('泥河湾  ·  166万年人类文明时间轴', fontsize=14,
             fontweight='bold', color=BRONZE3, pad=10)
fig.tight_layout()
fig.savefig(f'{OUT}/02_时间轴.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print('✓ 图2 时间轴')

# ═══════════════════════════════════════════════
# 图3：区位示意图（简化版中国地图 + 标注）
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor('#0A1825')

# 简化轮廓（粗略形状）
# 用矩形和椭圆近似主要地理区域
china = mpatches.Ellipse((0.5, 0.48), 0.75, 0.72,
    facecolor='#1C2E1A', edgecolor='#3A5A30', linewidth=1.5)
ax.add_patch(china)

# 河北省区域（近似位置）
hebei = mpatches.FancyBboxPatch((0.53, 0.58), 0.12, 0.10,
    boxstyle='round,pad=0.01',
    facecolor='#2A4020', edgecolor=BRONZE2, linewidth=1.5)
ax.add_patch(hebei)
ax.text(0.595, 0.63, '河北省', ha='center', va='center',
        fontsize=10, color=BRONZE, fontweight='bold')

# 泥河湾标记
ax.plot(0.565, 0.685, 'o', markersize=14, color=BRONZE3,
        zorder=5, markeredgecolor=BG, markeredgewidth=2)
ax.plot(0.565, 0.685, '*', markersize=10, color=BG, zorder=6)

# 标注
ax.annotate('泥河湾遗址\n张家口·阳原县', xy=(0.565, 0.685),
            xytext=(0.72, 0.77),
            fontsize=11, color=BRONZE3, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BRONZE, lw=2),
            linespacing=1.5)

# 北京标记（参照点）
ax.plot(0.62, 0.66, 's', markersize=7, color='#FF8888', zorder=5)
ax.text(0.635, 0.66, '北京', fontsize=9, color='#FF9999', va='center')

# 图例
ax.text(0.5, 0.08,
        '泥河湾遗址位于河北省张家口市阳原县\n地处太行山北麓，距北京约280公里',
        ha='center', va='center', fontsize=10, color=CREAM2,
        bbox=dict(boxstyle='round', facecolor=DARK2,
                  edgecolor=BRONZE2, linewidth=1),
        linespacing=1.6)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('泥河湾遗址区位示意图  ·  河北省张家口市', fontsize=13,
             fontweight='bold', color=BRONZE3, pad=12)
fig.tight_layout()
fig.savefig(f'{OUT}/03_区位示意图.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print('✓ 图3 区位示意图')

# ═══════════════════════════════════════════════
# 图4：数字化方案框架图
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# 中心圆
center = plt.Circle((6, 3), 1.1, facecolor='#2A1C0A',
                     edgecolor=BRONZE, linewidth=2.5, zorder=3)
ax.add_patch(center)
ax.text(6, 3.15, '泥河湾', ha='center', va='center',
        fontsize=14, fontweight='bold', color=BRONZE3, zorder=4)
ax.text(6, 2.7, '数字化平台', ha='center', va='center',
        fontsize=10, color=CREAM2, zorder=4)

branches = [
    (1.8, 5.2, '三维地质复原', ['精准复原古地貌', '地层可视化', '石器出土定位'], BRONZE3),
    (1.8, 0.8, 'VR时空穿越',  ['166万年前场景', '沉浸式体验', '科学场景重建'], '#E8B850'),
    (10.2,5.2, '数字博物馆',  ['文物3D展示', '科普图文', '全球开放访问'], BRONZE),
    (10.2,0.8, '智慧传播',    ['多语言平台', '社交媒体', '国际学术共享'], STONE),
]
for (x, y, title, pts, color) in branches:
    # 连接线
    ax.plot([6 + (x-6)*0.22, x - (x-6)*0.22],
            [3 + (y-3)*0.22, y - (y-3)*0.22],
            color=color, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
    # 节点框
    box = FancyBboxPatch((x-1.5, y-0.9), 3.0, 1.8,
        boxstyle='round,pad=0.1',
        facecolor='#1C1208', edgecolor=color, linewidth=1.8, zorder=2)
    ax.add_patch(box)
    ax.text(x, y+0.55, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color, zorder=3)
    for j, pt in enumerate(pts):
        ax.text(x, y+0.1-j*0.38, f'· {pt}', ha='center', va='center',
                fontsize=8.5, color=CREAM2, zorder=3)

ax.set_xlim(0, 12)
ax.set_ylim(0, 6.2)
ax.axis('off')
ax.set_title('泥河湾遗址数字化保护与传播方案架构图', fontsize=13,
             fontweight='bold', color=BRONZE3, pad=10)
fig.tight_layout()
fig.savefig(f'{OUT}/04_方案架构图.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print('✓ 图4 方案架构图')

print(f'\n全部图表已保存至：{OUT}/')
print('文件列表：')
for f in sorted(os.listdir(OUT)):
    print(f'  {f}')
