#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 draw.io XML 格式的 Q2/Q3 数据流程图
用户可直接用 https://app.diagrams.net 打开并导出为 Visio .vsdx 格式
"""
import os, textwrap
os.chdir('/home/user/math_pro')

# ── Gane-Sarson 风格样式定义 ──────────────────────────────────────────────────
# 外部实体：黑色细边框实心矩形
STYLE_ENTITY = (
    "rounded=0;whiteSpace=wrap;html=1;"
    "fillColor=#f5f5f5;strokeColor=#333333;fontColor=#333333;"
    "fontStyle=1;fontSize=11;"
)
# 过程：圆角矩形，顶部有编号分隔线
STYLE_PROC = (
    "shape=mxgraph.flowchart.process;whiteSpace=wrap;html=1;"
    "fillColor=#dae8fc;strokeColor=#2980b9;fontColor=#1a1a1a;"
    "fontSize=10;fontStyle=1;"
)
# 过程编号标签（左上角小格）
STYLE_PROC_NUM = (
    "rounded=0;whiteSpace=wrap;html=1;"
    "fillColor=#2980b9;strokeColor=#2980b9;fontColor=#ffffff;"
    "fontSize=10;fontStyle=1;"
)
# 数据存储：上下双横线（Gane-Sarson风格）
STYLE_STORE = (
    "shape=mxgraph.dfd.dataStore;whiteSpace=wrap;html=1;"
    "fillColor=#fff2cc;strokeColor=#d6b656;fontColor=#1a1a1a;"
    "fontSize=10;fontStyle=1;"
)
# 箭头（数据流）
STYLE_ARROW = (
    "edgeStyle=orthogonalEdgeStyle;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
    "entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    "strokeColor=#555555;strokeWidth=1.5;fontSize=9;fontColor=#555555;"
)
STYLE_ARROW_FREE = (
    "edgeStyle=orthogonalEdgeStyle;html=1;"
    "strokeColor=#555555;strokeWidth=1.5;fontSize=9;fontColor=#555555;"
)

# ── XML 模板 ─────────────────────────────────────────────────────────────────
def make_cell(cid, value, style, x, y, w, h, vertex=True, parent="1"):
    kind = 'vertex="1"' if vertex else 'edge="1"'
    return (f'    <mxCell id="{cid}" value="{value}" style="{style}" '
            f'{kind} parent="{parent}">\n'
            f'      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f'    </mxCell>\n')

def make_edge(cid, value, style, src, tgt, parent="1",
              pts=None, ex=None, ey=None, nx=None, ny=None):
    exit_attr  = f'exitX="{ex}";exitY="{ey}";exitDx="0";exitDy="0";' if ex is not None else ""
    entry_attr = f'entryX="{nx}";entryY="{ny}";entryDx="0";entryDy="0";' if nx is not None else ""
    full_style = (
        f"edgeStyle=orthogonalEdgeStyle;html=1;{exit_attr}{entry_attr}"
        "strokeColor=#555555;strokeWidth=1.5;fontSize=9;fontColor=#333333;"
    )
    xml = (f'    <mxCell id="{cid}" value="{value}" style="{full_style}" '
           f'edge="1" source="{src}" target="{tgt}" parent="{parent}">\n'
           f'      <mxGeometry relative="1" as="geometry"')
    if pts:
        xml += '>\n        <Array as="points">\n'
        for px, py in pts:
            xml += f'          <mxPoint x="{px}" y="{py}" />\n'
        xml += '        </Array>\n      </mxGeometry>\n'
    else:
        xml += ' />\n'
    xml += '    </mxCell>\n'
    return xml

def wrap_diagram(inner, page_name="DFD"):
    return textwrap.dedent(f"""\
<?xml version="1.0" encoding="UTF-8"?>
<mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1"
              tooltips="1" connect="1" arrows="1" fold="1"
              page="1" pageScale="1" pageWidth="1654" pageHeight="1169"
              math="0" shadow="0">
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
{inner}  </root>
</mxGraphModel>
""")

# ═══════════════════════════════════════════════════════════════════════════
# Q2 DFD：CO浓度预测
# ═══════════════════════════════════════════════════════════════════════════
#
#  布局（像素坐标，A3横向 1654×1169）：
#
#   [E1 烧结机传感器] ──→ [P1 数据预处理/滞后对齐] ──→ [D1 对齐后工况数据]
#                                  │                            │
#                                  ↓                            ↓
#                         [D2 异常区段标记]           [P2 特征工程(84维)]
#                                                               │
#                          ┌────────────────────────────────────┘
#                          ↓
#                    [D3 特征矩阵 X(n×84)]
#                     ↙              ↘
#           [P3 PSO优化XGBoost]   [P4 模型训练]
#                     │                  ↑
#                     ↓                  │
#           [D4 最优超参数θ*] ────────────┘
#                                        │
#                                        ↓
#                               [D5 XGBoost预测模型]
#                                        │
#                    [E2 当前工况输入] ──→ [P5 在线CO浓度预测]
#                                        │
#                                        ↓
#                               [E3 CO浓度预测值输出]
#
def gen_q2():
    cells = []
    i = 100  # ID 起始

    # ── 外部实体 ──────────────────────────────────────────────────────────────
    # E1: 烧结机传感器（左上）
    cells.append(make_cell(i, "E1\n烧结机\n传感器数据", STYLE_ENTITY, 60, 80, 130, 70)); E1=i; i+=1
    # E2: 当前工况输入（左下）
    cells.append(make_cell(i, "E2\n当前工况\n在线输入", STYLE_ENTITY, 60, 700, 130, 70)); E2=i; i+=1
    # E3: 输出
    cells.append(make_cell(i, "E3\n预测输出\nCO(ppm)", STYLE_ENTITY, 1450, 700, 130, 70)); E3=i; i+=1

    # ── 过程节点 ──────────────────────────────────────────────────────────────
    # P1: 数据预处理
    cells.append(make_cell(i, "P1\n数据预处理\n与滞后对齐\n(FFT互相关)", STYLE_PROC, 310, 60, 170, 90)); P1=i; i+=1
    # P2: 特征工程
    cells.append(make_cell(i, "P2\n特征工程\n(物理41+梯度34\n+统计4+CO自回归5)", STYLE_PROC, 620, 240, 200, 90)); P2=i; i+=1
    # P3: PSO优化
    cells.append(make_cell(i, "P3\nPSO超参数\n寻优\n(8粒子×10迭代)", STYLE_PROC, 310, 450, 170, 90)); P3=i; i+=1
    # P4: 模型训练
    cells.append(make_cell(i, "P4\nXGBoost\n模型训练\n(3折TimeSeriesCV)", STYLE_PROC, 620, 450, 200, 90)); P4=i; i+=1
    # P5: 在线预测
    cells.append(make_cell(i, "P5\n在线CO\n浓度预测\n(实时推理)", STYLE_PROC, 1100, 680, 200, 90)); P5=i; i+=1

    # ── 数据存储 ──────────────────────────────────────────────────────────────
    # D1: 对齐后数据
    cells.append(make_cell(i, "D1  对齐后工况时序数据", STYLE_STORE, 580, 80, 260, 50)); D1=i; i+=1
    # D2: 异常标记
    cells.append(make_cell(i, "D2  异常区段标记 (1045-1061)", STYLE_STORE, 310, 220, 260, 50)); D2=i; i+=1
    # D3: 特征矩阵
    cells.append(make_cell(i, "D3  特征矩阵 X  (n×84维)", STYLE_STORE, 580, 370, 260, 50)); D3=i; i+=1
    # D4: 最优超参数
    cells.append(make_cell(i, "D4  最优超参数 θ*  (7个参数)", STYLE_STORE, 310, 600, 260, 50)); D4=i; i+=1
    # D5: 模型
    cells.append(make_cell(i, "D5  XGBoost 预测模型 (已训练)", STYLE_STORE, 900, 560, 280, 50)); D5=i; i+=1

    # ── 数据流（箭头）─────────────────────────────────────────────────────────
    # E1 → P1
    cells.append(make_edge(i,"原始传感器\n时序数据","",E1,P1,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    # P1 → D1
    cells.append(make_edge(i,"对齐后\n时序数据","",P1,D1,ex=1,ey=0.3,nx=0,ny=0.5)); i+=1
    # P1 → D2
    cells.append(make_edge(i,"异常时段\n标记","",P1,D2,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    # D1 → P2
    cells.append(make_edge(i,"去除异常后\n时序数据","",D1,P2,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    # P2 → D3
    cells.append(make_edge(i,"84维\n特征矩阵","",P2,D3,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    # D3 → P3
    cells.append(make_edge(i,"训练特征\n(CV评估)","",D3,P3,ex=0,ey=0.5,nx=1,ny=0.5)); i+=1
    # D3 → P4
    cells.append(make_edge(i,"训练集\nX_train","",D3,P4,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    # P3 → D4
    cells.append(make_edge(i,"最优参数\n(CV-R²最大)","",P3,D4,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    # D4 → P4
    cells.append(make_edge(i,"超参数\n配置","",D4,P4,ex=1,ey=0.5,nx=0,ny=1)); i+=1
    # P4 → D5
    cells.append(make_edge(i,"已训练\nXGBoost模型","",P4,D5,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    # D5 → P5
    cells.append(make_edge(i,"预测模型\n加载","",D5,P5,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    # E2 → P5
    cells.append(make_edge(i,"当前时刻\n传感器+CO滞后","",E2,P5,ex=1,ey=0.5,nx=0,ny=1)); i+=1
    # P5 → E3
    cells.append(make_edge(i,"CO浓度\n预测值(ppm)","",P5,E3,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1

    # ── 标题 ──────────────────────────────────────────────────────────────────
    title_style = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=16;fontStyle=1;"
    cells.append(make_cell(i, "问题二数据流程图：基于PSO-XGBoost的CO浓度预测系统", title_style, 400, 10, 700, 35)); i+=1

    # 图例
    legend = ("text;html=1;strokeColor=#cccccc;fillColor=#f9f9f9;align=left;"
              "verticalAlign=middle;whiteSpace=wrap;fontSize=9;")
    cells.append(make_cell(i,
        "图例说明\n■ 矩形(灰)：外部实体\n■ 圆角矩形(蓝)：过程\n■ 双线矩形(黄)：数据存储\n→ 箭头：数据流",
        legend, 60, 850, 200, 100)); i+=1

    return wrap_diagram("".join(cells), "Q2-CO预测DFD")


# ═══════════════════════════════════════════════════════════════════════════
# Q3 DFD：风箱负压优化
# ═══════════════════════════════════════════════════════════════════════════
def gen_q3():
    cells = []
    i = 200

    # ── 外部实体 ──────────────────────────────────────────────────────────────
    cells.append(make_cell(i, "E1\n烧结机\n当前工况", STYLE_ENTITY, 60, 80, 130, 70)); E1=i; i+=1
    cells.append(make_cell(i, "E2\n历史\n生产数据", STYLE_ENTITY, 60, 350, 130, 70)); E2=i; i+=1
    cells.append(make_cell(i, "E3\n操作控制\n系统", STYLE_ENTITY, 1460, 420, 130, 70)); E3=i; i+=1

    # ── 过程节点 ──────────────────────────────────────────────────────────────
    cells.append(make_cell(i, "P1\n当前工况\n状态分析\n(CO=3495 ppm)", STYLE_PROC, 290, 60, 180, 90)); P1=i; i+=1
    cells.append(make_cell(i, "P2\n压力约束\n构建\n(Q10%-Q90%分位数)", STYLE_PROC, 290, 330, 180, 90)); P2=i; i+=1
    cells.append(make_cell(i, "P3\n不动点迭代\nCO*计算\n(阻尼α=0.4,80步)", STYLE_PROC, 630, 200, 200, 90)); P3=i; i+=1
    cells.append(make_cell(i, "P4\nPSO压力优化\n(40粒子×150迭代\n可靠性惩罚)", STYLE_PROC, 900, 330, 200, 90)); P4=i; i+=1
    cells.append(make_cell(i, "P5\n约束可行性\n验证\n(18风箱全部✓)", STYLE_PROC, 900, 560, 200, 90)); P5=i; i+=1
    cells.append(make_cell(i, "P6\n最优压力\n方案输出\n(CO*=1299 ppm)", STYLE_PROC, 1200, 420, 200, 90)); P6=i; i+=1

    # ── 数据存储 ──────────────────────────────────────────────────────────────
    cells.append(make_cell(i, "D1  Q2 XGBoost 预测模型 (已加载)", STYLE_STORE, 590, 80, 280, 50)); D1=i; i+=1
    cells.append(make_cell(i, "D2  历史压力数据 (18风箱×N时刻)", STYLE_STORE, 290, 480, 280, 50)); D2=i; i+=1
    cells.append(make_cell(i, "D3  压力约束边界 [LB_i, UB_i] i=1..18", STYLE_STORE, 620, 480, 310, 50)); D3=i; i+=1
    cells.append(make_cell(i, "D4  粒子位置与速度 (40×18维)", STYLE_STORE, 900, 200, 280, 50)); D4=i; i+=1
    cells.append(make_cell(i, "D5  全局最优压力向量 P* (18维)", STYLE_STORE, 1200, 200, 280, 50)); D5=i; i+=1
    cells.append(make_cell(i, "D6  最优方案配置表 (18风箱调整量)", STYLE_STORE, 1200, 600, 280, 50)); D6=i; i+=1

    # ── 数据流 ────────────────────────────────────────────────────────────────
    cells.append(make_edge(i,"当前传感器\n状态","",E1,P1,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"当前CO\n及压力","",P1,D1,ex=1,ey=0.3,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"历史压力\n时序","",E2,P2,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"历史压力\n数据","",P2,D2,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    cells.append(make_edge(i,"约束范围\n[LB,UB]","",P2,D3,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"预测模型\nf_XGB","",D1,P3,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    cells.append(make_edge(i,"当前压力\n初始点","",P1,P3,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"CO*(P)\n稳态预测","",P3,D4,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"粒子适应度\nf(P)=CO*+惩罚","",D4,P4,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    cells.append(make_edge(i,"约束边界\n参考","",D3,P4,ex=0.5,ey=1,nx=0,ny=0.5)); i+=1
    cells.append(make_edge(i,"全局最优\np_best","",P4,D5,ex=0.5,ey=0,nx=0.5,ny=1)); i+=1
    cells.append(make_edge(i,"最优压力\nP*","",D5,P6,ex=1,ey=0.5,nx=0,ny=0)); i+=1
    cells.append(make_edge(i,"候选解\n验证","",P4,P5,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    cells.append(make_edge(i,"可行解\n反馈","",P5,P4,ex=0,ey=0.5,nx=0,ny=1,
                           pts=[(860,605),(860,375)])); i+=1
    cells.append(make_edge(i,"最优配置\n存储","",P6,D6,ex=0.5,ey=1,nx=0.5,ny=0)); i+=1
    cells.append(make_edge(i,"18风箱最优\n负压指令","",P6,E3,ex=1,ey=0.5,nx=0,ny=0.5)); i+=1

    # ── 标题 & 图例 ──────────────────────────────────────────────────────────
    title_style = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=16;fontStyle=1;"
    cells.append(make_cell(i, "问题三数据流程图：基于PSO的烧结机风箱负压优化系统", title_style, 400, 10, 750, 35)); i+=1

    legend = ("text;html=1;strokeColor=#cccccc;fillColor=#f9f9f9;align=left;"
              "verticalAlign=middle;whiteSpace=wrap;fontSize=9;")
    cells.append(make_cell(i,
        "图例说明\n■ 矩形(灰)：外部实体\n■ 圆角矩形(蓝)：过程\n■ 双线矩形(黄)：数据存储\n→ 箭头：数据流",
        legend, 60, 720, 200, 100)); i+=1

    return wrap_diagram("".join(cells), "Q3-压力优化DFD")


# ── 保存文件 ─────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

q2_xml = gen_q2()
q3_xml = gen_q3()

with open('results/Q2_DFD.drawio', 'w', encoding='utf-8') as f:
    f.write(q2_xml)
print('✓ results/Q2_DFD.drawio')

with open('results/Q3_DFD.drawio', 'w', encoding='utf-8') as f:
    f.write(q3_xml)
print('✓ results/Q3_DFD.drawio')

print('\n使用方法：')
print('1. 打开 https://app.diagrams.net（免费）')
print('2. File → Import from → Device，选择 .drawio 文件')
print('3. 调整布局后，File → Export as → VSDX (Visio) 即可导出 Visio 格式')
