#!/usr/bin/env python3
"""生成 Visio Gane-Sarson 标准 DFD 图 —— Q2 & Q3"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import os

os.makedirs('/home/user/math_pro/figures', exist_ok=True)

FONT = 'WenQuanYi Zen Hei'
plt.rcParams['font.family'] = FONT
plt.rcParams['axes.unicode_minus'] = False

FP = fm.FontProperties(family=FONT)

# ── 基础绘图函数 ───────────────────────────────────────────────────────────────

def text(ax, x, y, s, fs=9, bold=False, **kw):
    w = 'bold' if bold else 'normal'
    ax.text(x, y, s, ha='center', va='center', fontsize=fs,
            fontweight=w, fontproperties=FP, multialignment='center',
            linespacing=1.45, zorder=10, **kw)

def ext_entity(ax, cx, cy, w, h, label, fs=9):
    """外部实体：矩形，白底黑边"""
    ax.add_patch(mpatches.Rectangle(
        (cx-w/2, cy-h/2), w, h,
        lw=1.3, edgecolor='black', facecolor='white', zorder=4))
    text(ax, cx, cy, label, fs)

def process(ax, cx, cy, w, h, label, fs=8.5):
    """处理过程：圆角矩形，白底黑边"""
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle='round,pad=0.12',
        lw=1.3, edgecolor='black', facecolor='white', zorder=4))
    text(ax, cx, cy, label, fs)

def data_store(ax, cx, cy, w, h, id_lbl, name_lbl, fs=8.5):
    """数据存储：Gane-Sarson 双横线 + 左侧 ID 标签，灰底"""
    x0, y0 = cx - w/2, cy - h/2
    tab = w * 0.20
    # 灰色填充
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), w, h, lw=0, facecolor='#E6E6E6', zorder=4))
    # 上下边线
    ax.plot([x0, x0+w], [y0+h, y0+h], 'k-', lw=1.3, zorder=5)
    ax.plot([x0, x0+w], [y0,    y0],   'k-', lw=1.3, zorder=5)
    # 左侧竖分隔线
    ax.plot([x0+tab, x0+tab], [y0, y0+h], 'k-', lw=1.0, zorder=5)
    # ID 编号（粗）
    text(ax, x0+tab/2, cy, id_lbl, fs=8, bold=True)
    # 名称
    text(ax, x0+tab+(w-tab)/2, cy, name_lbl, fs)

def flow(ax, x1, y1, x2, y2, label='', fs=7.5, pos=0.5, off=(0, 0)):
    """数据流：黑色实线末端箭头 + 标注"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=6,
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=0.95, mutation_scale=13))
    if label:
        lx = x1 + (x2-x1)*pos + off[0]
        ly = y1 + (y2-y1)*pos + off[1]
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fs,
                fontproperties=FP, multialignment='center', zorder=11,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.8, alpha=0.93))

def flow_path(ax, pts, label='', fs=7.5, li=0, off=(0, 0)):
    """折线数据流（直角连线）"""
    for i in range(len(pts)-1):
        x1, y1 = pts[i]; x2, y2 = pts[i+1]
        if i < len(pts)-2:
            ax.plot([x1, x2], [y1, y2], 'k-', lw=0.95, zorder=6)
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=6,
                        arrowprops=dict(arrowstyle='->', color='black',
                                        lw=0.95, mutation_scale=13))
    if label:
        lx = (pts[li][0]+pts[li+1][0])/2 + off[0]
        ly = (pts[li][1]+pts[li+1][1])/2 + off[1]
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fs,
                fontproperties=FP, multialignment='center', zorder=11,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.8, alpha=0.93))

# ════════════════════════════════════════════════════════════════════════════════
# Q2  DFD：CO 预测模型建立
# ════════════════════════════════════════════════════════════════════════════════
def draw_q2():
    W, H = 24, 16
    fig, ax = plt.subplots(figsize=(W*0.9, H*0.9), dpi=150)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_title('问题二：CO 预测模型建立  数据流图（DFD）',
                 fontsize=13, fontweight='bold', pad=14,
                 fontproperties=FP)

    # 尺寸
    EW, EH = 3.0, 1.1   # 外部实体
    PW, PH = 3.4, 1.5   # 处理过程
    SW, SH = 5.2, 1.0   # 数据存储

    # ── 坐标（cx, cy）──────────────────────────────────
    # 三列布局：左列 x≈2.2  中列 x≈9  右列 x≈18
    EE1 = (2.0,  14.5)   # 烧结生产系统
    EE2 = (18.0,  1.5)   # 模型使用方

    P1  = (9.0,  14.5)   # 数据采集与预处理
    P2  = (9.0,  11.5)   # 特征工程
    P3  = (5.5,   8.5)   # PSO 超参数寻优
    P4  = (12.5,  8.5)   # XGBoost 模型训练
    P5  = (9.0,   5.5)   # 模型评估验证

    D1  = (18.5,  14.5)  # D1 原始工况数据库
    D2  = (2.2,   11.5)  # D2 清洗后数据集
    D3  = (18.5,  11.5)  # D3 84 维特征数据集
    D4  = (18.5,   8.5)  # D4 XGBoost 预测模型

    # ── 形状 ───────────────────────────────────────────
    ext_entity(ax, *EE1, EW, EH, '烧结\n生产系统')
    ext_entity(ax, *EE2, EW, EH, '模型\n使用方')

    process(ax, *P1, PW, PH, 'P1\n数据采集与预处理')
    process(ax, *P2, PW, PH, 'P2\n特征工程')
    process(ax, *P3, PW, PH, 'P3\nPSO 超参数\n寻优')
    process(ax, *P4, PW, PH, 'P4\nXGBoost\n模型训练')
    process(ax, *P5, PW, PH, 'P5\n模型评估\n验证')

    data_store(ax, *D1, SW, SH, 'D1', '原始工况数据库')
    data_store(ax, *D2, SW, SH, 'D2', '清洗后数据集（2425 条）')
    data_store(ax, *D3, SW, SH, 'D3', '84 维特征数据集')
    data_store(ax, *D4, SW, SH, 'D4', 'XGBoost 预测模型\nR²=0.9992')

    # ── 数据流 ─────────────────────────────────────────
    # EE1 → P1
    flow(ax, EE1[0]+EW/2, EE1[1], P1[0]-PW/2, P1[1],
         '原始压力/温度\n/速度/CO 记录')

    # P1 → D1（写入原始）
    flow(ax, P1[0]+PW/2, P1[1], D1[0]-SW/2, D1[1], '写入原始记录')

    # P1 → D2（清洗结果 向下偏左）
    flow(ax, P1[0]-PW/4, P1[1]-PH/2, D2[0]+SW/3, D2[1]+SH/2,
         '2425 条清洗后记录')

    # D2 → P2（读取）
    flow(ax, D2[0]+SW/2, D2[1], P2[0]-PW/2, P2[1], '读取清洗数据')

    # P2 → D3（特征矩阵）
    flow(ax, P2[0]+PW/2, P2[1], D3[0]-SW/2, D3[1],
         '84 维特征矩阵\n(物理+梯度+统计+CO 滞后)')

    # D3 → P3（CV 折叠特征集）
    flow(ax, D3[0]-SW/2, D3[1]-SH/4, P3[0]+PW/2, P3[1]+PH/4,
         '特征集\n(CV 折叠)', pos=0.45, off=(0, 0.35))

    # D3 → P4（训练特征集）
    flow(ax, D3[0]-SW/4, D3[1]-SH/2, P4[0]+PW/4, P4[1]+PH/2,
         '训练特征集', pos=0.4, off=(0.3, 0))

    # P3 → P4（最优超参数）
    flow(ax, P3[0]+PW/2, P3[1], P4[0]-PW/2, P4[1],
         '最优超参数\n(lr=0.206, depth=3, n=262)')

    # P4 → D4（保存模型）
    flow(ax, P4[0]+PW/2, P4[1], D4[0]-SW/2, D4[1], '保存训练完成模型')

    # D4 → P5（调用预测模型）
    flow(ax, D4[0]-SW/4, D4[1]-SH/2, P5[0]+PW/3, P5[1]+PH/2,
         '调用预测模型', pos=0.4, off=(0.4, 0))

    # D2 → P5（测试集，长路径折线）
    flow_path(ax,
              [(D2[0], D2[1]-SH/2), (D2[0], 4.0), (P5[0]-PW/2, 4.0)],
              '测试集样本\n(695 条)', li=1, off=(0.6, 0.3))

    # P5 → EE2（评估报告）
    flow(ax, P5[0]+PW/2, P5[1], EE2[0]-SW/2, EE2[1],
         'R²=0.9992 / MAE=50.1\n评估报告')

    # D4 → EE2（部署模型）
    flow(ax, D4[0], D4[1]-SH/2, EE2[0], EE2[1]+EH/2, '部署 XGBoost 预测模型')

    # ── 图例 ───────────────────────────────────────────
    legend_y = 0.6
    ext_entity(ax, 1.5, legend_y, 1.5, 0.55, '外部实体', fs=7)
    process(ax, 4.5, legend_y, 1.7, 0.55, 'Px 处理过程', fs=7)
    data_store(ax, 8.2, legend_y, 2.5, 0.55, 'Dx', '数据存储', fs=7)
    ax.annotate('', xy=(11.5, legend_y), xytext=(10.2, legend_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9))
    ax.text(12.2, legend_y, '数据流', va='center', fontsize=7, fontproperties=FP, zorder=10)

    path = '/home/user/math_pro/figures/Q2_DFD.png'
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  ✓ {path}')


# ════════════════════════════════════════════════════════════════════════════════
# Q3  DFD：PSO 压力优化
# ════════════════════════════════════════════════════════════════════════════════
def draw_q3():
    W, H = 24, 22
    fig, ax = plt.subplots(figsize=(W*0.85, H*0.85), dpi=150)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_title('问题三：PSO 压力优化  数据流图（DFD）',
                 fontsize=13, fontweight='bold', pad=14, fontproperties=FP)

    EW, EH = 3.2, 1.1
    PW, PH = 3.6, 1.6
    SW, SH = 5.4, 1.0

    # 三列：左 x≈2  中 x≈9  右 x≈18.5
    EE1 = (2.0,  20.5)   # 工况监测系统
    EE2 = (19.5,  1.5)   # 压力调控执行系统

    P1  = (9.0,  20.5)   # P1 稳态 CO 计算
    P2  = (9.0,  17.5)   # P2 调节边界确定
    P3  = (9.0,  14.5)   # P3 PSO 粒子群初始化
    P4  = (9.0,  11.5)   # P4 稳态适应度评估+惩罚
    P5  = (9.0,   8.5)   # P5 自适应粒子更新
    P6  = (9.0,   5.5)   # P6 收敛判断&输出

    D1  = (19.0,  20.5)  # D1 稳态 CO 基准值
    D2  = (2.0,   17.5)  # D2 历史压力分位数表
    D3  = (19.0,  17.5)  # D3 18 维压力调节范围
    D4  = (19.0,  14.5)  # D4 Q2 XGBoost 预测模型
    D5  = (19.0,  11.5)  # D5 粒子群状态矩阵
    D6  = (19.0,   8.5)  # D6 全局最优解记录

    # ── 形状 ───────────────────────────────────────────
    ext_entity(ax, *EE1, EW, EH, '工况\n监测系统')
    ext_entity(ax, *EE2, EW, EH, '压力调控\n执行系统')

    process(ax, *P1, PW, PH, 'P1\n稳态 CO 计算\n(固定点迭代 α=0.5×50 次)')
    process(ax, *P2, PW, PH, 'P2\n调节边界确定\n(10%～90% 分位数截取)')
    process(ax, *P3, PW, PH, 'P3\nPSO 粒子群初始化\n(40 粒子 × 18 维)')
    process(ax, *P4, PW, PH, 'P4\n稳态适应度评估\n+ 可靠性惩罚')
    process(ax, *P5, PW, PH, 'P5\n自适应粒子更新\n(w:0.9→0.4, α:50→500)')
    process(ax, *P6, PW, PH, 'P6\n收敛判断\n& 输出最优解')

    data_store(ax, *D1, SW, SH, 'D1', '稳态 CO 基准值\n(3495.4 mg/m³)')
    data_store(ax, *D2, SW, SH, 'D2', '历史压力分位数表\n(10%～90%, 18 维)')
    data_store(ax, *D3, SW, SH, 'D3', '18 维压力调节范围\n[下限, 上限]')
    data_store(ax, *D4, SW, SH, 'D4', 'Q2 XGBoost 预测模型\n(R²=0.9992)')
    data_store(ax, *D5, SW, SH, 'D5', '粒子群状态矩阵\n(40×18 位置 + 速度)')
    data_store(ax, *D6, SW, SH, 'D6', '全局最优解记录\n(最优压力 + 最优 CO)')

    # ── 数据流 ─────────────────────────────────────────

    # EE1 → P1（当前负压）
    flow(ax, EE1[0]+EW/2, EE1[1], P1[0]-PW/2, P1[1],
         '18 个风箱当前负压值')

    # D4 → P1（CO 预测函数，绕路：D4 左边 → 上行 → P1 右边）
    flow_path(ax,
              [(D4[0]-SW/2, D4[1]+SH/2+0.15),
               (D4[0]-SW/2-1.2, D4[1]+SH/2+0.15),
               (D4[0]-SW/2-1.2, P1[1]+PH/2+0.3),
               (P1[0]+PW/2,     P1[1]+PH/2+0.3),
               (P1[0]+PW/2,     P1[1])],
              'CO 预测函数', li=2, off=(0, 0.35))

    # P1 → D1（稳态 CO）
    flow(ax, P1[0]+PW/2, P1[1], D1[0]-SW/2, D1[1],
         '稳态 CO=3495.4 mg/m³')

    # D1 → P2（基准值向下）
    flow_path(ax,
              [(D1[0]-SW/2, D1[1]-SH/2-0.05),
               (D1[0]-SW/2-0.5, D1[1]-SH/2-0.05),
               (D1[0]-SW/2-0.5, P2[1]+PH/2+0.15),
               (P2[0]+PW/2, P2[1]+PH/2+0.15),
               (P2[0]+PW/2, P2[1])],
              '稳态 CO 基准值', li=2, off=(0, 0.35))

    # EE1 → P2（当前工况参考，折线沿左侧向下）
    flow_path(ax,
              [(EE1[0]+EW/4, EE1[1]-EH/2),
               (EE1[0]+EW/4, P2[1]+PH/2+0.5),
               (P2[0]-PW/2,  P2[1]+PH/2+0.5),
               (P2[0]-PW/2,  P2[1])],
              '当前工况参考值', li=1, off=(0, 0.35))

    # D2 → P2（历史分位数）
    flow(ax, D2[0]+SW/2, D2[1], P2[0]-PW/2, P2[1], '18 维历史分位数')

    # P2 → D3（调节范围）
    flow(ax, P2[0]+PW/2, P2[1], D3[0]-SW/2, D3[1], '18 维压力 [下限, 上限]')

    # D3 → P3（调节范围边界）
    flow_path(ax,
              [(D3[0]-SW/2, D3[1]-SH/2-0.05),
               (D3[0]-SW/2-0.5, D3[1]-SH/2-0.05),
               (D3[0]-SW/2-0.5, P3[1]+PH/2+0.2),
               (P3[0]+PW/2, P3[1]+PH/2+0.2),
               (P3[0]+PW/2, P3[1])],
              '调节范围边界', li=2, off=(0, 0.35))

    # P3 → D5（初始粒子状态）
    flow(ax, P3[0]+PW/2, P3[1], D5[0]-SW/2, D5[1]+SH/3,
         '40×18 初始位置\n+ 速度矩阵', pos=0.45, off=(0.4, 0))

    # ── 迭代循环 ──
    # D5 → P4（读粒子位置）
    flow(ax, D5[0]-SW/2, D5[1], P4[0]+PW/2, P4[1], '当前粒子位置（18 维压力）')

    # D4 → P4（XGBoost 预测）
    flow(ax, D4[0]-SW/2, D4[1], P4[0]+PW/2, P4[1]-PH/4,
         'XGBoost CO 预测', pos=0.5, off=(0.3, -0.3))

    # P4 → P5（目标值）
    flow(ax, P4[0], P4[1]-PH/2, P5[0], P5[1]+PH/2,
         '目标值 = CO稳态 + α·Σexp(-10·d)')

    # P5 → D5（更新粒子状态）
    flow(ax, P5[0]+PW/2, P5[1]+PH/4, D5[0]-SW/2, D5[1]-SH/4,
         '更新粒子位置与速度')

    # P5 → D6（更新最优）
    flow(ax, P5[0]+PW/2, P5[1], D6[0]-SW/2, D6[1], '写入新全局最优解')

    # D6 → P6（读最优）
    flow(ax, D6[0]-SW/2, D6[1], P6[0]+PW/2, P6[1], '读取当前最优解')

    # P6 → P4（反馈：未收敛，沿左侧折回）
    loop_x = 2.8
    flow_path(ax,
              [(P6[0]-PW/2, P6[1]),
               (loop_x,      P6[1]),
               (loop_x,      P4[1]),
               (P4[0]-PW/2,  P4[1])],
              '未达 150 次迭代\n继续评估', li=1, off=(-0.8, 0.4))

    # P6 → EE2（最优方案输出）
    flow(ax, P6[0]+PW/2, P6[1], EE2[0]-EW/2, EE2[1],
         '最优 18 维压力配置\nCO=1299.5 mg/m³ / 减排 62.8%\n可靠性: 1/18 贴近边界')

    # ── 图例 ───────────────────────────────────────────
    legend_y = 0.55
    ext_entity(ax,  1.8, legend_y, 1.8, 0.6, '外部实体', fs=7)
    process(ax,     4.8, legend_y, 2.0, 0.6, 'Px 处理过程', fs=7)
    data_store(ax,  8.5, legend_y, 3.0, 0.6, 'Dx', '数据存储', fs=7)
    ax.annotate('', xy=(12.5, legend_y), xytext=(11.0, legend_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9))
    ax.text(13.3, legend_y, '数据流', va='center', fontsize=7, fontproperties=FP, zorder=10)

    path = '/home/user/math_pro/figures/Q3_DFD.png'
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  ✓ {path}')


if __name__ == '__main__':
    print('生成 Q2 DFD...')
    draw_q2()
    print('生成 Q3 DFD...')
    draw_q3()
    print('完成！')
