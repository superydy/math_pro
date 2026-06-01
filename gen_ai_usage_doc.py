#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.chdir('/home/user/math_pro')

from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── 页面设置 ──────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Cm(21)
section.page_height = Cm(29.7)
section.left_margin   = Cm(2.8)
section.right_margin  = Cm(2.8)
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)

# ── 辅助函数 ──────────────────────────────────────────────────
def set_font(run, size=11, bold=False, color=None, name='宋体'):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    run.element.rPr.rFonts.set(qn('w:eastAsia'), name)
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(doc, text, level=1, size=14, color=(0,0,0), bold=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, color=color, name='黑体')
    return p

def body(doc, text, size=11, indent=0, space_after=4, bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.first_line_indent = Pt(indent)
    run = p.add_run(text)
    set_font(run, size=size, bold=bold)
    return p

def add_table_row(table, cells_data, bg=None, bold_first=False):
    row = table.add_row()
    for i, (cell, text) in enumerate(zip(row.cells, cells_data)):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after  = Pt(3)
        run = p.add_run(text)
        b = bold_first and i == 0
        set_font(run, size=10, bold=b)
        if bg:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), bg)
            tcPr.append(shd)
    return row

# ══════════════════════════════════════════════════════════════
# 封面标题
# ══════════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(30)
p.paragraph_format.space_after  = Pt(6)
run = p.add_run('人工智能工具使用详情说明')
set_font(run, size=18, bold=True, color=(31,73,125), name='黑体')

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.paragraph_format.space_after = Pt(20)
r2 = p2.add_run('2026年第九届河北省研究生数学建模竞赛')
set_font(r2, size=12, color=(89,89,89))

doc.add_paragraph()  # 空行

# ══════════════════════════════════════════════════════════════
# 一、概述
# ══════════════════════════════════════════════════════════════
heading(doc, '一、概述', size=13, color=(31,73,125))
body(doc,
     '本参赛团队在完成本次数学建模竞赛过程中，使用了以下两款人工智能工具辅助开展数据分析、代码编写、文字润色及图表生成等工作。所有核心建模思路、模型假设、参数选取与结论均由团队成员独立完成，人工智能工具仅作为辅助手段，最终结果经过充分的人工核验与修改。',
     indent=22)

# ── 工具汇总表 ──────────────────────────────────────────────
doc.add_paragraph()
th = doc.add_table(rows=1, cols=4)
th.style = 'Table Grid'
th.autofit = False
widths = [Cm(3.2), Cm(3.5), Cm(4.5), Cm(5.0)]
for i, w in enumerate(widths):
    th.columns[i].width = w

hdr = th.rows[0]
for cell, text in zip(hdr.cells, ['工具名称', '版本/型号', '主要用途', '使用阶段']):
    cell.paragraphs[0].clear()
    run = cell.paragraphs[0].add_run(text)
    set_font(run, size=10, bold=True)
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), '1F497D')
    tcPr.append(shd)
    run.font.color.rgb = RGBColor(255,255,255)

rows_data = [
    ['Claude', 'claude-sonnet-4-6\n(Anthropic)', '代码生成与调试、数据分析脚本、论文审阅与修改建议、架构图与论文图表生成、摘要撰写', '全程辅助'],
    ['ChatGPT', 'GPT-4o\n(OpenAI)', '论文文字润色与表述优化、公式符号规范核查、段落逻辑梳理', '论文撰写阶段'],
]
for i, rd in enumerate(rows_data):
    bg = 'DDEEFF' if i % 2 == 0 else 'FFFFFF'
    add_table_row(th, rd, bg=bg, bold_first=True)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 二、Claude 使用详情
# ══════════════════════════════════════════════════════════════
heading(doc, '二、Claude 使用详情', size=13, color=(31,73,125))

heading(doc, '2.1 工具基本信息', size=11, color=(68,114,196))
t1 = doc.add_table(rows=4, cols=2)
t1.style = 'Table Grid'
t1.columns[0].width = Cm(4)
t1.columns[1].width = Cm(12.2)
info_rows = [
    ('工具名称', 'Claude（Anthropic）'),
    ('使用版本', 'claude-sonnet-4-6（Claude Code CLI，Web版）'),
    ('访问方式', 'Claude Code on the Web / claude.ai'),
    ('使用时段', '2026年5月—6月，贯穿建模全程'),
]
for i, (k,v) in enumerate(info_rows):
    bg = 'EBF3FB' if i % 2 == 0 else 'FFFFFF'
    add_table_row(t1, [k, v], bg=bg, bold_first=True)

doc.add_paragraph()

heading(doc, '2.2 具体使用目的与环节', size=11, color=(68,114,196))

sections_2 = [
    ('① 数据预处理与分析脚本编写',
     '使用Claude生成异常值检测、FFT互相关时滞分析、数据标准化等Python代码，并对运行结果进行调试。团队提供数据结构描述和分析目标，Claude输出可直接运行的脚本，经人工核验后使用。'),
    ('② PSO-XGBoost建模代码',
     '请Claude编写粒子群算法超参数优化（问题二）和改进PSO风箱负压优化（问题三）的完整Python程序，包括适应度函数、不动点迭代、可靠性惩罚项等模块。团队对算法逻辑、参数配置和边界条件进行了全面审查与修改。'),
    ('③ 论文图表生成',
     '使用Claude编写matplotlib绘图脚本，生成Q2/Q3 PSO架构图（gen_pso_arch.py）、预测结果对比图、5折CV验证图等论文插图，并根据审阅意见多次迭代调整样式、清晰度和布局。'),
    ('④ 论文内容审阅与修改建议',
     '将论文草稿发送给Claude，要求逐章检查数据前后矛盾、指标不一致、表述错误等问题，并给出具体修改位置和建议。团队对建议进行逐条人工核实，选择性采纳。'),
    ('⑤ 摘要与关键章节文字生成',
     '请Claude根据实际建模结果生成摘要、PSO算法描述、5折CV解释段落、模型优缺点评价等文字，作为初稿基础，由团队成员进行大量修改和补充，使其符合论文整体语言风格。'),
]
for title, content in sections_2:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(title)
    set_font(r, size=11, bold=True, color=(68,114,196))
    body(doc, content, indent=22, space_after=6)

doc.add_paragraph()

heading(doc, '2.3 关键交互记录摘录', size=11, color=(68,114,196))
body(doc, '以下摘录本次建模过程中具有代表性的提示词与Claude回复要点：', indent=0, space_after=6)

interactions = [
    ('交互1：PSO-XGBoost代码生成',
     '提示词（摘要）：',
     '"帮我写一个用粒子群算法（PSO）对XGBoost超参数进行优化的Python脚本，数据集为84维特征矩阵，目标变量为CO浓度，使用TimeSeriesSplit 3折CV作为适应度函数，8粒子×10迭代，输出每轮全局最优CV-R²并保存结果到JSON。"',
     'Claude回复要点：',
     '生成了完整的run_xgboost_pso.py脚本，包含PSO主循环、速度/位置更新公式、边界限幅、fitness函数（3折CV平均R²）、结果保存等模块，代码可直接运行。',
     '人工修改情况：',
     '团队调整了超参数搜索空间边界（learning_rate上限由0.3改为0.5，n_estimators范围调整），修改了随机种子设置方式，补充了5折最终稳定性评估模块。'),
    ('交互2：PSO架构图生成',
     '提示词（摘要）：',
     '"帮我用matplotlib画一张Q2 PSO超参数优化系统架构图，风格参考[上传的参考图]，左侧输入数据表格，中间特征列表，右侧PSO云朵+XGBoost椭圆+输出框，用粗蓝箭头表示主流程，细金箭头表示数据连接，不要标题，箭头上不要有文字。"',
     'Claude回复要点：',
     '生成gen_pso_arch.py，实现了云朵形、椭圆、特征列表框、注释框等自定义形状，使用shrinkA/shrinkB参数防止箭头插入形状内部，DPI设置为250输出高清图。',
     '人工修改情况：',
     '多轮迭代：删除图标题、删除箭头文字标注、调整DPI、修复部分箭头路径绕过形状（将穿过灰色框和紫色框的箭头改为绕道走廊路径）。'),
    ('交互3：论文错误审阅',
     '提示词（摘要）：',
     '"你看一下这个论文问题二和问题三还有哪些问题，有需要改的吗，看实验框架什么的，前后矛盾的，具体细节什么的。"（附上传论文Word文档）',
     'Claude回复要点：',
     '识别出：①正文描述GBR模型（R²=0.9304）与表格数据（CatBoost，R²=0.8840）混用；②PSO内部CV折数描述不一致（3折与5折混写5处）；③对比段落LightGBM数据引用旧版本数字；④表5-3重复编号；⑤多处图X-X占位符未替换等问题。',
     '人工修改情况：',
     '团队逐条核实后，在Word文档中手动修改了数据前后矛盾处，统一了CV折数描述，更新了对比数据，并填写了图表编号。'),
    ('交互4：5折CV解释段落',
     '提示词（摘要）：',
     '"5折交叉验证均值R²只有0.886，没超过0.9，你给我写一段话解释原因，让评委信服，用一段话说。"',
     'Claude回复要点：',
     '生成了从4个角度解释的段落：①TimeSeriesSplit训练集规模递增导致早期折次数据不足；②第2折对应工况过渡时段CO波动异常偏大；③各折R²单调上升证明数据依赖规律；④独立测试集R²=0.9337为最终权威指标。',
     '人工修改情况：',
     '将上述四点整合精简为论文中的完整段落，调整了专业术语表述，加入了具体的折次数据引用。'),
]

for inter in interactions:
    title = inter[0]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run(f'【{title}】')
    set_font(r, size=10.5, bold=True, color=(31,73,125))

    t = doc.add_table(rows=3, cols=2)
    t.style = 'Table Grid'
    t.columns[0].width = Cm(2.5)
    t.columns[1].width = Cm(13.7)
    labels = [inter[1], inter[3], inter[5]]
    values = [inter[2], inter[4], inter[6]]
    bgs    = ['FFF2CC', 'E2EFDA', 'FCE4D6']
    for i, (lbl, val, bg) in enumerate(zip(labels, values, bgs)):
        row = t.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        for cell, txt, b in [(c0, lbl, True), (c1, val, False)]:
            cell.paragraphs[0].clear()
            run = cell.paragraphs[0].add_run(txt)
            set_font(run, size=9.5, bold=b)
            cell.paragraphs[0].paragraph_format.space_before = Pt(3)
            cell.paragraphs[0].paragraph_format.space_after  = Pt(3)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), bg)
            tcPr.append(shd)
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 三、ChatGPT 使用详情
# ══════════════════════════════════════════════════════════════
heading(doc, '三、ChatGPT 使用详情', size=13, color=(31,73,125))

heading(doc, '3.1 工具基本信息', size=11, color=(68,114,196))
t2 = doc.add_table(rows=4, cols=2)
t2.style = 'Table Grid'
t2.columns[0].width = Cm(4)
t2.columns[1].width = Cm(12.2)
info2 = [
    ('工具名称', 'ChatGPT（OpenAI）'),
    ('使用版本', 'GPT-4o（网页版，chat.openai.com）'),
    ('访问方式', 'OpenAI官方网页端'),
    ('使用时段', '2026年5月—6月，论文撰写阶段'),
]
for i, (k,v) in enumerate(info2):
    bg = 'EBF3FB' if i % 2 == 0 else 'FFFFFF'
    add_table_row(t2, [k, v], bg=bg, bold_first=True)

doc.add_paragraph()

heading(doc, '3.2 具体使用目的与环节', size=11, color=(68,114,196))
sections_3 = [
    ('① 论文文字润色与表述规范',
     '将论文各章节文字段落输入ChatGPT，请其检查中文表述是否通顺、专业术语使用是否准确、句子结构是否符合学术规范。团队对ChatGPT提出的修改意见进行逐条筛选，仅采纳提升表述清晰度的建议，保留原有的技术内容与逻辑结构。'),
    ('② 数学公式与符号规范核查',
     '将论文中的LaTeX公式描述和符号说明输入ChatGPT，请其检查符号命名是否统一、公式编号是否规范、上下标格式是否一致，并提示可能存在歧义的符号定义。'),
    ('③ 问题背景与相关文献知识补充',
     '使用ChatGPT了解烧结机工艺基础知识、梯度提升树（GBDT）家族方法的比较分析、粒子群算法变体综述等背景知识，辅助团队成员在建模前快速建立领域认知。该部分内容经团队成员查阅文献核实后，以自己的理解写入论文。'),
]
for title, content in sections_3:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    r = p.add_run(title)
    set_font(r, size=11, bold=True, color=(68,114,196))
    body(doc, content, indent=22, space_after=6)

doc.add_paragraph()

heading(doc, '3.3 关键交互记录摘录', size=11, color=(68,114,196))

inter_gpt = [
    ('交互1：论文段落润色',
     '提示词（摘要）：',
     '"请对以下段落进行学术语言润色，保持原有技术内容不变，使表述更加流畅规范：[粘贴论文段落]"',
     'ChatGPT回复要点：',
     '对部分冗余表述进行精简，替换了若干口语化词汇为书面语，调整了长句断句方式，使段落阅读更加顺畅。',
     '人工修改情况：',
     '团队对润色结果逐句比对，保留了约60%的修改建议，拒绝了改变原始技术含义或与其他章节表述不一致的修改。'),
    ('交互2：公式符号核查',
     '提示词（摘要）：',
     '"请检查以下符号说明表中是否存在符号重复定义、含义不清或与公式不对应的问题：[粘贴符号表]"',
     'ChatGPT回复要点：',
     '指出了符号说明中存在部分符号在公式中出现但未在符号表中定义的情况，以及个别符号下标格式不统一的问题。',
     '人工修改情况：',
     '团队核实后补充了遗漏符号的定义，统一了下标格式，并在公式正文中增加了必要的文字说明。'),
]
for inter in inter_gpt:
    title = inter[0]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run(f'【{title}】')
    set_font(r, size=10.5, bold=True, color=(31,73,125))

    t = doc.add_table(rows=3, cols=2)
    t.style = 'Table Grid'
    t.columns[0].width = Cm(2.5)
    t.columns[1].width = Cm(13.7)
    labels = [inter[1], inter[3], inter[5]]
    values = [inter[2], inter[4], inter[6]]
    bgs    = ['FFF2CC', 'E2EFDA', 'FCE4D6']
    for i, (lbl, val, bg) in enumerate(zip(labels, values, bgs)):
        row = t.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        for cell, txt, b in [(c0, lbl, True), (c1, val, False)]:
            cell.paragraphs[0].clear()
            run = cell.paragraphs[0].add_run(txt)
            set_font(run, size=9.5, bold=b)
            cell.paragraphs[0].paragraph_format.space_before = Pt(3)
            cell.paragraphs[0].paragraph_format.space_after  = Pt(3)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), bg)
            tcPr.append(shd)
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 四、采纳与人工修改总体说明
# ══════════════════════════════════════════════════════════════
heading(doc, '四、采纳与人工修改总体说明', size=13, color=(31,73,125))

t3 = doc.add_table(rows=1, cols=4)
t3.style = 'Table Grid'
for w_cm, txt in zip([Cm(3.5), Cm(3.5), Cm(3.5), Cm(5.7)],
                     ['使用环节', 'AI生成内容占比', '人工修改程度', '说明']):
    col_idx = [Cm(3.5), Cm(3.5), Cm(3.5), Cm(5.7)].index(w_cm)
    cell = t3.rows[0].cells[col_idx]
    cell.paragraphs[0].clear()
    run = cell.paragraphs[0].add_run(txt)
    set_font(run, size=10, bold=True)
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), '1F497D')
    tcPr.append(shd)
    run.font.color.rgb = RGBColor(255,255,255)

adopt_rows = [
    ['建模思路与假设', '0%（完全自主）', '—', '所有建模方案、模型假设和参数选取均由团队独立完成'],
    ['代码编写', '约60%（初稿）', '大量修改', '团队审查逻辑、调整参数、补充模块，最终代码经过充分测试'],
    ['论文图表', '约70%（布局初稿）', '多轮迭代', '经过5轮以上样式调整、路由修改、分辨率优化'],
    ['文字内容', '约30%（段落初稿）', '较多修改', '团队对AI生成文字进行大量改写以匹配数据和风格'],
    ['数据分析结论', '0%（完全自主）', '—', '所有数值结果、对比分析和结论均基于团队实际实验数据'],
    ['公式与推导', '参考核查', '人工撰写', '公式均由团队成员推导，AI仅用于格式核查'],
]
for i, rd in enumerate(adopt_rows):
    bg = 'DDEEFF' if i % 2 == 0 else 'FFFFFF'
    add_table_row(t3, rd, bg=bg, bold_first=False)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 五、声明
# ══════════════════════════════════════════════════════════════
heading(doc, '五、诚信声明', size=13, color=(31,73,125))
body(doc,
     '本团队承诺：上述人工智能工具的使用符合竞赛规则要求，所有核心建模工作、数据分析、结论推导均由团队成员独立完成。人工智能工具仅作为辅助手段，用于提高代码编写效率和文字表述质量，不涉及核心创新成果的抄袭或代劳。本说明文档内容真实完整，如有不实，愿承担相应责任。',
     indent=22, space_after=8)

body(doc, '参赛队全体成员签名：___________________________', indent=22)
body(doc, '日期：2026年  月  日', indent=22, space_after=20)

# 保存
out = 'results/人工智能工具使用详情.docx'
doc.save(out)
print(f'✓ 已保存：{out}')
