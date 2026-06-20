#!/usr/bin/env python3
"""
赛题5 参赛PPT：数字赋能·剪映千年——蔚县剪纸的数字化活化与创新传播
第三届中国研究生"文化中国"两创大赛
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ─── 颜色主题 ───────────────────────────────────────────────
DARK_RED   = RGBColor(0xC0, 0x20, 0x20)   # 中国红（主色）
GOLD       = RGBColor(0xD4, 0xAF, 0x37)   # 金色（点缀）
DARK_GRAY  = RGBColor(0x2C, 0x2C, 0x2C)   # 深灰（正文）
LIGHT_BG   = RGBColor(0xFD, 0xF6, 0xEC)   # 米白（背景）
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
DARK_RED2  = RGBColor(0x8B, 0x00, 0x00)   # 深红

# ─── 工具函数 ────────────────────────────────────────────────

def blank_layout(prs):
    return prs.slide_layouts[6]  # 完全空白

def set_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=DARK_GRAY,
                align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape

def add_para(tf, text, font_size=16, bold=False, color=DARK_GRAY,
             align=PP_ALIGN.LEFT, space_before=Pt(4)):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = space_before
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p

def slide_header(slide, title, subtitle=None):
    """统一顶部标题栏"""
    add_rect(slide, 0, 0, 13.33, 1.2, DARK_RED)
    add_textbox(slide, title, 0.4, 0.15, 10, 0.8,
                font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_textbox(slide, subtitle, 0.4, 0.75, 10, 0.4,
                    font_size=14, color=RGBColor(0xFF,0xDD,0xCC), align=PP_ALIGN.LEFT)
    # 底部装饰条
    add_rect(slide, 0, 7.2, 13.33, 0.3, GOLD)

# ════════════════════════════════════════════════════════════════
# Slide 1 — 封面
# ════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank_layout(prs))
set_bg(s1, LIGHT_BG)

# 左侧大色块
add_rect(s1, 0, 0, 5.5, 7.5, DARK_RED)
# 金色竖线
add_rect(s1, 5.5, 0, 0.08, 7.5, GOLD)

# 左侧文字
add_textbox(s1, "文化中国", 0.4, 0.6, 4.8, 1.0,
            font_size=22, bold=False, color=RGBColor(0xFF,0xCC,0xAA), align=PP_ALIGN.LEFT)
add_textbox(s1, "两创大赛", 0.4, 1.2, 4.8, 0.7,
            font_size=18, color=RGBColor(0xFF,0xCC,0xAA), align=PP_ALIGN.LEFT)
add_rect(s1, 0.4, 2.0, 3.5, 0.05, GOLD)

add_textbox(s1, "数字赋能·剪映千年", 0.3, 2.2, 5.0, 1.1,
            font_size=30, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
add_textbox(s1, "蔚县剪纸的数字化活化与创新传播", 0.3, 3.3, 5.0, 0.8,
            font_size=18, color=RGBColor(0xFF,0xEE,0xDD), align=PP_ALIGN.LEFT)

add_textbox(s1, "赛题 5  开放赛题", 0.4, 4.5, 4.5, 0.5,
            font_size=14, color=GOLD, align=PP_ALIGN.LEFT)
add_textbox(s1, "第三届中国研究生【文化中国】两创大赛", 0.4, 5.0, 4.8, 0.5,
            font_size=12, color=RGBColor(0xFF,0xCC,0xAA), align=PP_ALIGN.LEFT)
add_textbox(s1, "2026年", 0.4, 5.6, 2, 0.5,
            font_size=13, color=RGBColor(0xFF,0xCC,0xAA), align=PP_ALIGN.LEFT)

# 右侧内容
add_textbox(s1, "剪纸", 6.5, 1.0, 6.0, 1.2,
            font_size=72, bold=True, color=RGBColor(0xEE,0xCC,0xAA), align=PP_ALIGN.CENTER)
add_textbox(s1, "千年技艺  数字新生", 5.8, 2.8, 7.0, 0.7,
            font_size=20, bold=True, color=DARK_RED, align=PP_ALIGN.CENTER)

# 右侧要点
for i, (icon, txt) in enumerate([
    ("◆", "国家级非物质文化遗产"),
    ("◆", "AI辅助设计 · 创新转化"),
    ("◆", "AR沉浸体验 · 活化传承"),
    ("◆", "数字档案 · 永久留存"),
]):
    y = 3.8 + i * 0.6
    add_textbox(s1, icon, 6.0, y, 0.5, 0.5, font_size=13, bold=True, color=DARK_RED)
    add_textbox(s1, txt, 6.6, y, 6.2, 0.5, font_size=13, color=DARK_GRAY)

# 底部金条
add_rect(s1, 0, 7.2, 13.33, 0.3, GOLD)
add_textbox(s1, "注：作品中不含学校、个人及指导教师信息", 5.8, 7.2, 7.0, 0.3,
            font_size=9, color=DARK_GRAY, align=PP_ALIGN.RIGHT)

# ════════════════════════════════════════════════════════════════
# Slide 2 — 目录
# ════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank_layout(prs))
set_bg(s2, LIGHT_BG)
slide_header(s2, "目  录", "Content")

items = [
    ("01", "创作背景", "蔚县剪纸的历史渊源与当代困境"),
    ("02", "问题分析", "传承断层与传播局限"),
    ("03", "设计理念", "以数字技术赋能传统非遗"),
    ("04", "技术方案", "AI生成 · AR体验 · 数字档案"),
    ("05", "应用场景", "文旅融合 · 教育传承 · 文创开发"),
    ("06", "社会价值", "文化自信与创新发展"),
]

cols = [(0.4, 3.0), (6.9, 3.0)]
for idx, (num, title, desc) in enumerate(items):
    col = idx % 2
    row = idx // 2
    lx, ly = cols[col]
    y = ly + row * 1.35

    add_rect(s2, lx, y, 0.7, 0.9, DARK_RED)
    add_textbox(s2, num, lx, y+0.05, 0.7, 0.8,
                font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s2, lx+0.7, y, 5.5, 0.9, RGBColor(0xF5,0xEC,0xE0))
    add_textbox(s2, title, lx+0.85, y+0.02, 5.0, 0.42,
                font_size=17, bold=True, color=DARK_RED)
    add_textbox(s2, desc, lx+0.85, y+0.45, 5.0, 0.42,
                font_size=12, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 3 — 创作背景
# ════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank_layout(prs))
set_bg(s3, LIGHT_BG)
slide_header(s3, "01  创作背景", "蔚县剪纸：千年技艺的当代困境")

# 左栏：历史介绍
add_rect(s3, 0.4, 1.4, 5.8, 5.5, RGBColor(0xFF,0xF5,0xEA))
add_textbox(s3, "蔚县剪纸简介", 0.7, 1.5, 5.2, 0.5,
            font_size=16, bold=True, color=DARK_RED)

facts = [
    "• 起源于明代，距今已有500余年历史",
    "• 国家级非物质文化遗产（2006年入选）",
    "• 以【点色彩绘】技法著称，色彩鲜艳独特",
    "• 核心产区：河北省张家口市蔚县",
    "• 年产剪纸超过5000万张，出口40余国",
    "• 题材涵盖戏曲人物、民俗风情、吉祥图案",
]
for i, f in enumerate(facts):
    add_textbox(s3, f, 0.7, 2.1 + i*0.7, 5.2, 0.65, font_size=13, color=DARK_GRAY)

# 右栏：困境
add_rect(s3, 6.6, 1.4, 6.3, 5.5, RGBColor(0xFF,0xEE,0xEE))
add_textbox(s3, "当代传承困境", 6.9, 1.5, 5.8, 0.5,
            font_size=16, bold=True, color=DARK_RED)

problems = [
    ("传承人老龄化", "核心传承人平均年龄超过60岁，年轻接班人严重不足"),
    ("市场认知度低", "大众对蔚县剪纸了解有限，与现代审美脱节"),
    ("技艺记录缺失", "大量独门技法仅靠口耳相传，缺乏系统性数字记录"),
    ("传播渠道单一", "主要依赖线下展销，数字化传播能力薄弱"),
]
for i, (t, d) in enumerate(problems):
    y = 2.1 + i * 1.15
    add_rect(s3, 6.7, y, 0.08, 0.9, DARK_RED)
    add_textbox(s3, t, 6.9, y, 5.7, 0.4, font_size=13, bold=True, color=DARK_RED2)
    add_textbox(s3, d, 6.9, y+0.4, 5.7, 0.5, font_size=12, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 4 — 问题分析
# ════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank_layout(prs))
set_bg(s4, LIGHT_BG)
slide_header(s4, "02  问题分析", "传承断层 · 传播局限 · 创新缺位")

# 三个问题框
problem_data = [
    ("传承断层", DARK_RED,
     ["核心技艺难以量化记录", "学习成本高，周期长达数年", "年轻人从业意愿持续下降", "现有传承人不足200人"]),
    ("传播局限", RGBColor(0xB8,0x5C,0x00),
     ["线下展览覆盖人群有限", "数字内容质量参差不齐", "缺乏互动性与沉浸感", "跨文化传播能力弱"]),
    ("创新缺位", RGBColor(0x1A,0x6B,0x3A),
     ["产品形态与现代生活脱节", "文创开发缺乏系统设计", "与现代科技融合度不足", "品牌认知度低"]),
]
for col, (title, color, points) in enumerate(problem_data):
    x = 0.4 + col * 4.3
    add_rect(s4, x, 1.4, 3.9, 0.6, color)
    add_textbox(s4, title, x, 1.45, 3.9, 0.55,
                font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s4, x, 2.0, 3.9, 4.9, RGBColor(0xF8,0xF4,0xEE))
    for i, pt in enumerate(points):
        add_textbox(s4, f"▸ {pt}", x+0.2, 2.1+i*1.1, 3.5, 1.0, font_size=13, color=DARK_GRAY)

# 底部结论
add_rect(s4, 0.4, 6.8, 12.53, 0.35, RGBColor(0xF0,0xE0,0xD0))
add_textbox(s4, "结论：亟需以现代数字技术为手段，探索蔚县剪纸创造性转化与创新性发展的新路径",
            0.6, 6.82, 12.2, 0.3, font_size=12, bold=True, color=DARK_RED, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
# Slide 5 — 设计理念
# ════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank_layout(prs))
set_bg(s5, LIGHT_BG)
slide_header(s5, "03  设计理念", "以数字技术赋能非遗 · 让传统文化活起来")

# 核心理念图
add_rect(s5, 4.67, 2.8, 4.0, 1.5, DARK_RED)
add_textbox(s5, "数字赋能", 4.67, 2.9, 4.0, 0.6,
            font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_textbox(s5, "蔚县剪纸活化传承", 4.67, 3.5, 4.0, 0.6,
            font_size=14, color=RGBColor(0xFF,0xEE,0xDD), align=PP_ALIGN.CENTER)

principles = [
    (0.5, 2.0, "保护优先", "以数字技术建立完整的技艺档案，\n在数字空间永久保存每一种剪纸技法"),
    (0.5, 4.8, "活化传承", "通过AI学习辅助工具降低技艺入门\n门槛，让更多人参与传承实践"),
    (9.3, 2.0, "创新转化", "结合现代审美进行创意设计，开发\n符合当代市场需求的文创产品"),
    (9.3, 4.8, "广泛传播", "借助AR/VR与社交媒体，打破地域\n限制，实现文化价值的最大传播"),
]
for (x, y, title, desc) in principles:
    add_rect(s5, x, y, 3.5, 1.9, RGBColor(0xFD,0xF0,0xE5))
    add_rect(s5, x, y, 3.5, 0.45, RGBColor(0xE8,0x88,0x50))
    add_textbox(s5, title, x, y+0.02, 3.5, 0.42,
                font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s5, desc, x+0.15, y+0.5, 3.2, 1.3,
                font_size=12, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 6 — 技术方案
# ════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank_layout(prs))
set_bg(s6, LIGHT_BG)
slide_header(s6, "04  技术方案", "三大技术支柱构建数字化传承体系")

tech_cols = [
    ("AI辅助设计系统", DARK_RED, [
        "构建蔚县剪纸图案数据库（5000+样本）",
        "训练深度学习模型识别传统纹样",
        "AI辅助新图案生成，保留传统风格基因",
        "用户交互设计工具，降低创作门槛",
        "输出标准化矢量图，支持工业生产",
    ]),
    ("AR沉浸体验平台", RGBColor(0xB8,0x5C,0x00), [
        "3D扫描建立剪纸作品高精度数字模型",
        "AR识别技术：扫描实物即触发动态展示",
        "虚拟剪纸体验：用户可在线剪出自己作品",
        "非遗传承人录制技艺讲解视频内嵌展示",
        "多语言支持，面向国际传播",
    ]),
    ("数字化档案平台", RGBColor(0x1A,0x6B,0x3A), [
        "对濒危技法进行全流程视频记录",
        "建立传承人口述历史数字档案库",
        "高清图像+3D模型双重保存",
        "开放API供研究机构调用",
        "云端备份，永久保存",
    ]),
]
for col, (title, color, points) in enumerate(tech_cols):
    x = 0.4 + col * 4.3
    add_rect(s6, x, 1.35, 3.9, 0.65, color)
    add_textbox(s6, title, x, 1.38, 3.9, 0.6,
                font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s6, x, 2.0, 3.9, 5.0, RGBColor(0xF8,0xF4,0xEE))
    for i, pt in enumerate(points):
        add_textbox(s6, f"✓  {pt}", x+0.15, 2.1+i*0.9, 3.6, 0.85, font_size=12, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 7 — 创新亮点
# ════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank_layout(prs))
set_bg(s7, LIGHT_BG)
slide_header(s7, "05  创新亮点", "技术创新 × 文化创新 × 模式创新")

highlights = [
    ("技术创新", "首创剪纸纹样AI生成模型",
     "将深度学习与传统美学结合，训练专属于蔚县剪纸风格的生成式AI，"
     "既能辅助传统艺人进行图案设计，又能帮助初学者快速入门，"
     "实现了技艺传播的智能化与大众化。"),
    ("体验创新", "AR技术打造沉浸式非遗体验",
     "用户通过手机扫描剪纸作品，即可看到作品背后的历史故事与制作过程动画，"
     "真正实现「见物见人见生活」，打破传统展陈的单向传播局限。"),
    ("模式创新", "构建保护-传承-产业完整链条",
     "从数字档案保护，到学习传承平台，再到AI辅助文创设计开发，"
     "形成完整的非遗数字化生态，探索可复制、可推广的非遗活化新模式。"),
]
for i, (tag, title, desc) in enumerate(highlights):
    y = 1.4 + i * 1.85
    add_rect(s7, 0.4, y, 1.2, 1.6, DARK_RED)
    add_textbox(s7, tag, 0.4, y+0.5, 1.2, 0.6,
                font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s7, 1.6, y, 11.2, 1.6, RGBColor(0xFD,0xF0,0xE5))
    add_textbox(s7, title, 1.8, y+0.1, 10.8, 0.5,
                font_size=16, bold=True, color=DARK_RED)
    add_textbox(s7, desc, 1.8, y+0.6, 10.8, 0.9,
                font_size=13, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 8 — 应用场景
# ════════════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank_layout(prs))
set_bg(s8, LIGHT_BG)
slide_header(s8, "06  应用场景", "多维度落地 · 全链条赋能")

scenarios = [
    ("🏛", "文旅融合", DARK_RED, [
        "蔚县景区AR导览系统",
        "非遗主题互动体验馆",
        "数字剪纸文创伴手礼",
        "线上非遗文化旅游节",
    ]),
    ("📚", "教育传承", RGBColor(0xB8,0x5C,0x00), [
        "中小学美育课程数字教材",
        "高校非遗研究数据库",
        "在线剪纸学习App",
        "传承人技艺线上公开课",
    ]),
    ("🎨", "文创产业", RGBColor(0x1A,0x6B,0x3A), [
        "AI辅助剪纸图案授权设计",
        "非遗IP开发与品牌孵化",
        "数字藏品（NFT）限量发行",
        "国潮服饰纹样合作授权",
    ]),
    ("🌐", "国际传播", RGBColor(0x2B,0x4C,0x8C), [
        "多语言非遗展示平台",
        "海外文化中心数字展览",
        "与国际博物馆数据共享",
        "跨文化艺术创作交流",
    ]),
]
for col, (icon, title, color, items) in enumerate(scenarios):
    x = 0.4 + col * 3.2
    add_rect(s8, x, 1.35, 2.9, 5.8, RGBColor(0xFD,0xF5,0xEB))
    add_rect(s8, x, 1.35, 2.9, 0.7, color)
    add_textbox(s8, f"{title}", x, 1.38, 2.9, 0.65,
                font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for i, item in enumerate(items):
        add_textbox(s8, f"▸ {item}", x+0.15, 2.15+i*1.1, 2.6, 1.0, font_size=12, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 9 — 社会价值
# ════════════════════════════════════════════════════════════════
s9 = prs.slides.add_slide(blank_layout(prs))
set_bg(s9, LIGHT_BG)
slide_header(s9, "07  社会价值与意义", "文化自信 · 创新发展 · 国际传播")

values = [
    ("坚定文化自信", "蔚县剪纸是中华优秀传统文化的重要载体。通过数字技术的活化传承，"
     "让更多中国人、尤其是年轻一代直观感受传统美学的魅力，"
     "在潜移默化中增强文化认同感与民族自豪感。"),
    ("践行两创精神", "本项目以习近平文化思想为指导，将传统剪纸与AI、AR等现代技术深度融合，"
     "是对中华优秀传统文化「创造性转化、创新性发展」的具体实践，"
     "探索了非遗在数字时代焕发生机的可行路径。"),
    ("服务国家战略", "契合「数字中国」建设、文旅融合发展、乡村振兴等国家重大战略，"
     "助力蔚县打造非遗文化品牌，带动地方文化产业发展，"
     "实现社会效益与经济效益的双重提升。"),
    ("推动文明互鉴", "数字平台突破地域与语言限制，将蔚县剪纸推向国际舞台，"
     "以中华美学感染世界，贡献中国文化软实力，"
     "推动中华文明与世界文明的交流互鉴。"),
]
for i, (title, desc) in enumerate(values):
    col = i % 2
    row = i // 2
    x = 0.4 + col * 6.5
    y = 1.4 + row * 2.7
    add_rect(s9, x, y, 6.0, 2.4, RGBColor(0xFD,0xF0,0xE5))
    add_rect(s9, x, y, 6.0, 0.5, DARK_RED)
    add_textbox(s9, title, x, y+0.03, 6.0, 0.45,
                font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s9, desc, x+0.2, y+0.6, 5.6, 1.7, font_size=12, color=DARK_GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 10 — 结语
# ════════════════════════════════════════════════════════════════
s10 = prs.slides.add_slide(blank_layout(prs))
set_bg(s10, DARK_RED)

add_textbox(s10, "数字赋能·剪映千年", 1.5, 1.2, 10.3, 1.5,
            font_size=42, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_textbox(s10, "蔚县剪纸的数字化活化与创新传播", 1.5, 2.8, 10.3, 0.9,
            font_size=22, color=RGBColor(0xFF,0xEE,0xDD), align=PP_ALIGN.CENTER)

add_rect(s10, 3.5, 3.9, 6.33, 0.06, GOLD)

summary_items = [
    "以AI技术赋能传统技艺，降低传承门槛",
    "以AR平台创新传播方式，提升文化感染力",
    "以数字档案永久留存，守护非遗根脉",
]
for i, item in enumerate(summary_items):
    add_textbox(s10, f"◆  {item}", 2.0, 4.2+i*0.65, 9.33, 0.6,
                font_size=15, color=GOLD, align=PP_ALIGN.CENTER)

add_textbox(s10, "让千年剪纸技艺，在数字时代焕发新生", 1.5, 6.2, 10.3, 0.6,
            font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_rect(s10, 0, 7.2, 13.33, 0.3, GOLD)
add_textbox(s10, "赛题5 · 开放赛题  |  第三届中国研究生【文化中国】两创大赛", 0, 7.22, 13.33, 0.28,
            font_size=11, color=DARK_RED, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
# 保存
# ════════════════════════════════════════════════════════════════
output = "/home/user/math_pro/赛题5-参赛作品展示.pptx"
prs.save(output)
print(f"PPT已生成：{output}")
print(f"共 {len(prs.slides)} 张幻灯片")
