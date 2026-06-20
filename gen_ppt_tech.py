#!/usr/bin/env python3
"""
赛题5 参赛PPT - 科技风版本
深色背景 + 青蓝色科技感 + 图多字少
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ─── 科技风色彩 ──────────────────────────────────────────────
BG        = RGBColor(0x07, 0x0E, 0x1A)   # 极深蓝黑
BG2       = RGBColor(0x0D, 0x1B, 0x2A)   # 稍浅深蓝
CYAN      = RGBColor(0x00, 0xD4, 0xFF)   # 电光青蓝
CYAN2     = RGBColor(0x00, 0x8B, 0xCC)   # 深青蓝
GOLD      = RGBColor(0xFF, 0xD7, 0x00)   # 金黄
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
GRAY      = RGBColor(0xAA, 0xBB, 0xCC)
DIM       = RGBColor(0x55, 0x66, 0x77)
GREEN     = RGBColor(0x00, 0xFF, 0x88)   # 科技绿
ORANGE    = RGBColor(0xFF, 0x88, 0x00)   # 警示橙

def blank(prs):
    return prs.slide_layouts[6]

def bg(slide, color=BG):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def tb(slide, text, l, t, w, h, sz=18, bold=False, color=WHITE,
       align=PP_ALIGN.LEFT, italic=False):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(sz)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return box

def rect(slide, l, t, w, h, fill, line=None, alpha=None):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
    else:
        s.line.fill.background()
    return s

def oval(slide, l, t, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    s = slide.shapes.add_shape(9, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(2)
    else:
        s.line.fill.background()
    return s

def hline(slide, l, t, w, color=CYAN, thickness=1.5):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(0.025))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    return s

def card(slide, l, t, w, h, title, title_color=CYAN, body_lines=None,
         border_color=CYAN, bg_color=RGBColor(0x0D,0x20,0x35)):
    rect(slide, l, t, w, h, bg_color, border_color)
    hline(slide, l, t, w, border_color, 2)
    tb(slide, title, l+0.15, t+0.12, w-0.3, 0.5, sz=15, bold=True, color=title_color)
    if body_lines:
        for i, line in enumerate(body_lines):
            tb(slide, line, l+0.15, t+0.65+i*0.5, w-0.3, 0.48, sz=12, color=GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 1 — 封面
# ════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank(prs))
bg(s1)

# 右侧装饰竖条
for i, x in enumerate([12.8, 12.9, 13.0, 13.1]):
    c = [CYAN2, CYAN, CYAN2, BG2][i]
    rect(s1, x, 0, 0.12, 7.5, c)

# 顶部细线
hline(s1, 0.5, 1.1, 9.0, CYAN)

# 大标题
tb(s1, "数字赋能  剪映千年", 0.5, 1.3, 11.0, 1.8, sz=52, bold=True, color=WHITE)

# 副标题
tb(s1, "蔚县剪纸的数字化活化与创新传播", 0.5, 3.1, 10.0, 0.9,
   sz=24, color=CYAN, bold=False)

# 底线
hline(s1, 0.5, 4.1, 9.0, CYAN)

# 三个关键词标签
for i, (label, val) in enumerate([
    ("AI  辅助设计", CYAN),
    ("AR  沉浸体验", GREEN),
    ("数字档案库", GOLD),
]):
    x = 0.5 + i * 3.5
    rect(s1, x, 4.4, 3.0, 0.7, RGBColor(0x0D,0x25,0x3A), CYAN2)
    tb(s1, label, x, 4.42, 3.0, 0.65, sz=16, bold=True, color=val, align=PP_ALIGN.CENTER)

# 左下信息
tb(s1, "第三届中国研究生文化中国两创大赛  · 赛题5 开放赛题  · 2026", 0.5, 6.7, 11.0, 0.5,
   sz=11, color=DIM)

# 右下装饰数字
tb(s1, "500+", 10.0, 5.3, 3.0, 1.1, sz=60, bold=True,
   color=RGBColor(0x0D,0x30,0x45), align=PP_ALIGN.RIGHT)
tb(s1, "年非遗传承", 10.5, 6.3, 2.5, 0.5, sz=13, color=DIM, align=PP_ALIGN.RIGHT)

# ════════════════════════════════════════════════════════════════
# Slide 2 — 现状：三组数据 + 一句话问题
# ════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank(prs))
bg(s2)

hline(s2, 0, 1.15, 13.33, CYAN)
tb(s2, "面临的困境", 0.5, 0.2, 8.0, 0.9, sz=32, bold=True, color=WHITE)
tb(s2, "传统非遗技艺的传承正在断裂", 0.5, 0.85, 8.0, 0.4, sz=16, color=GRAY)

# 三大数据卡片
data = [
    ("< 200", "人", "全国蔚县剪纸\n核心传承人", ORANGE),
    ("60+",   "岁", "传承人\n平均年龄", RGBColor(0xFF,0x44,0x44)),
    ("5000万","张/年", "年产量却无法\n触达年轻群体", CYAN),
]
for i, (num, unit, desc, color) in enumerate(data):
    x = 0.5 + i * 4.2
    rect(s2, x, 1.5, 3.9, 3.8, RGBColor(0x0D,0x20,0x30), color)
    hline(s2, x, 1.5, 3.9, color, 3)
    tb(s2, num,  x+0.2, 1.7,  3.5, 1.4, sz=56, bold=True, color=color)
    tb(s2, unit, x+0.2, 3.1,  3.5, 0.5, sz=18, color=GRAY)
    tb(s2, desc, x+0.2, 3.65, 3.5, 0.8, sz=13, color=GRAY)

# 底部结论
rect(s2, 0, 5.6, 13.33, 1.0, RGBColor(0x05, 0x15, 0x28))
hline(s2, 0, 5.6, 13.33, CYAN)
tb(s2, "亟需用现代数字技术，让千年剪纸在数字时代重新焕发生命力",
   0.5, 5.72, 12.5, 0.65, sz=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
# Slide 3 — 解决方案全景（三大技术支柱）
# ════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank(prs))
bg(s3)

hline(s3, 0, 1.1, 13.33, CYAN)
tb(s3, "数字化解决方案", 0.5, 0.15, 8.0, 0.9, sz=32, bold=True, color=WHITE)
tb(s3, "三大技术支柱  · 全链条赋能", 0.5, 0.8, 8.0, 0.4, sz=16, color=CYAN)

pillars = [
    ("AI", "辅助设计系统",
     ["图案识别 & 生成", "降低创作门槛", "保留风格基因", "矢量图输出"],
     CYAN, RGBColor(0x00,0x40,0x60)),
    ("AR", "沉浸体验平台",
     ["扫描实物即展示", "虚拟剪纸互动", "技艺讲解内嵌", "多语言支持"],
     GREEN, RGBColor(0x00,0x3A,0x20)),
    ("云", "数字化档案库",
     ["全流程视频记录", "传承人口述存档", "高清3D模型", "开放数据共享"],
     GOLD, RGBColor(0x3A,0x30,0x00)),
]
for i, (icon, title, points, color, bg_c) in enumerate(pillars):
    x = 0.4 + i * 4.3
    rect(s3, x, 1.35, 4.0, 5.8, bg_c, color)
    # 大图标圆形背景
    oval(s3, x+1.25, 1.55, 1.5, 1.5, color)
    tb(s3, icon, x+1.25, 1.6, 1.5, 1.4, sz=52, bold=True,
       color=RGBColor(0x07,0x0E,0x1A), align=PP_ALIGN.CENTER)
    tb(s3, title, x+0.1, 3.2, 3.8, 0.6,
       sz=17, bold=True, color=color, align=PP_ALIGN.CENTER)
    hline(s3, x+0.3, 3.9, 3.4, color)
    for j, pt in enumerate(points):
        tb(s3, f"▶  {pt}", x+0.25, 4.1+j*0.7, 3.5, 0.65, sz=13, color=WHITE)

# ════════════════════════════════════════════════════════════════
# Slide 4 — AI 辅助设计
# ════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank(prs))
bg(s4)

hline(s4, 0, 1.1, 13.33, CYAN)
tb(s4, "AI 辅助设计系统", 0.5, 0.1, 9.0, 0.9, sz=34, bold=True, color=WHITE)
tb(s4, "让机器读懂剪纸的美学密码", 0.5, 0.8, 9.0, 0.4, sz=16, color=CYAN)

# 左侧：流程图
steps = [
    ("INPUT",  "采集5000+\n传统纹样", CYAN),
    ("TRAIN",  "深度学习\n风格建模", GREEN),
    ("GEN",    "AI生成\n新图案", GOLD),
    ("OUTPUT", "矢量输出\n用于生产", CYAN),
]
for i, (tag, label, color) in enumerate(steps):
    y = 1.35 + i * 1.35
    rect(s4, 0.4, y, 1.0, 1.1, color)
    tb(s4, tag, 0.4, y+0.15, 1.0, 0.8, sz=13, bold=True,
       color=BG, align=PP_ALIGN.CENTER)
    tb(s4, label, 1.5, y+0.15, 2.8, 0.8, sz=14, color=WHITE)
    if i < 3:
        tb(s4, "▼", 0.7, y+1.1, 0.5, 0.3, sz=14, color=color, align=PP_ALIGN.CENTER)

# 中间箭头
rect(s4, 4.2, 2.8, 0.5, 1.5, RGBColor(0x0D,0x20,0x30))
tb(s4, "➡", 4.2, 3.1, 0.5, 0.9, sz=28, color=CYAN, align=PP_ALIGN.CENTER)

# 右侧：效果亮点
rect(s4, 5.0, 1.35, 7.9, 5.8, RGBColor(0x0D,0x20,0x35), CYAN)
tb(s4, "核心创新", 5.2, 1.5, 7.5, 0.5, sz=18, bold=True, color=CYAN)
hline(s4, 5.2, 2.05, 7.3, CYAN)

highlights = [
    ("首创", "蔚县剪纸专属AI生成模型，保留点色彩绘风格基因"),
    ("降门槛", "零基础用户可在5分钟内生成正宗剪纸图案"),
    ("规模化", "批量输出标准矢量图，支持文创工厂直接生产"),
    ("可交互", "实时调整色彩、构图，个性化定制专属作品"),
]
for i, (tag, text) in enumerate(highlights):
    y = 2.2 + i * 1.1
    rect(s4, 5.2, y, 1.1, 0.75, CYAN)
    tb(s4, tag, 5.2, y+0.1, 1.1, 0.6, sz=14, bold=True,
       color=BG, align=PP_ALIGN.CENTER)
    tb(s4, text, 6.4, y+0.05, 6.2, 0.7, sz=13, color=GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 5 — AR 沉浸体验
# ════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank(prs))
bg(s5)

hline(s5, 0, 1.1, 13.33, GREEN)
tb(s5, "AR 沉浸体验平台", 0.5, 0.1, 9.0, 0.9, sz=34, bold=True, color=WHITE)
tb(s5, "扫一扫，让剪纸活起来", 0.5, 0.8, 9.0, 0.4, sz=16, color=GREEN)

# 左侧：手机模型（用矩形模拟）
rect(s5, 0.5, 1.3, 3.2, 5.5, RGBColor(0x12,0x12,0x12),
     RGBColor(0x44,0x44,0x44))
rect(s5, 0.65, 1.5, 2.9, 4.9, RGBColor(0x05,0x18,0x28))
# 屏幕内容
tb(s5, "AR SCAN", 0.65, 1.6, 2.9, 0.5, sz=11, bold=True,
   color=GREEN, align=PP_ALIGN.CENTER)
# 模拟扫描框
rect(s5, 1.0, 2.2, 2.2, 2.2, RGBColor(0x05,0x18,0x28),
     RGBColor(0x00,0xFF,0x88))
tb(s5, "[  剪纸识别中  ]", 1.0, 3.1, 2.2, 0.6, sz=11,
   color=GREEN, align=PP_ALIGN.CENTER)
tb(s5, "动态展示触发...", 0.75, 4.6, 2.7, 0.4, sz=10, color=GRAY, align=PP_ALIGN.CENTER)
# 小圆按钮
oval(s5, 1.6, 5.9, 0.6, 0.6, RGBColor(0x33,0x33,0x33), RGBColor(0x66,0x66,0x66))

# 右侧：功能说明
features = [
    ("01", "实物触发", "手机扫描剪纸，即刻呈现动态故事与背景介绍"),
    ("02", "虚拟制作", "屏幕上跟着传承人一步步剪出属于自己的作品"),
    ("03", "技艺传授", "传承人录制讲解视频，扫码随时观看学习"),
    ("04", "多端传播", "一键分享AR体验至社交媒体，扩大文化传播"),
]
for i, (num, title, desc) in enumerate(features):
    y = 1.4 + i * 1.4
    x = 4.2
    rect(s5, x, y, 0.6, 1.1, GREEN)
    tb(s5, num, x, y+0.15, 0.6, 0.8, sz=16, bold=True,
       color=BG, align=PP_ALIGN.CENTER)
    rect(s5, x+0.6, y, 8.2, 1.1, RGBColor(0x05,0x18,0x20), GREEN)
    tb(s5, title, x+0.8, y+0.08, 7.8, 0.45, sz=16, bold=True, color=GREEN)
    tb(s5, desc,  x+0.8, y+0.55, 7.8, 0.45, sz=13, color=GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 6 — 数字档案 + 应用场景（合并）
# ════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank(prs))
bg(s6)

hline(s6, 0, 1.1, 13.33, GOLD)
tb(s6, "数字档案库  ×  应用场景", 0.5, 0.1, 10.0, 0.9, sz=32, bold=True, color=WHITE)
tb(s6, "永久留存技艺基因  · 多维度落地赋能", 0.5, 0.8, 10.0, 0.4, sz=16, color=GOLD)

# 左侧：档案库可视化（数据层次图）
rect(s6, 0.4, 1.3, 5.5, 5.9, RGBColor(0x0D,0x1F,0x10), GOLD)
layers = [
    ("视频层", "全流程录制  1080P+", GOLD, 0.9),
    ("图像层", "高清扫描  精度0.1mm", CYAN, 0.7),
    ("3D层",  "三维建模  还原立体", GREEN, 0.5),
    ("文字层", "口述整理  传承人档案", GRAY, 0.3),
    ("API层",  "开放接口  学术共享", DIM, 0.15),
]
tb(s6, "数字档案结构", 0.6, 1.4, 5.0, 0.5, sz=14, bold=True, color=GOLD)
for i, (name, desc, color, opacity) in enumerate(layers):
    y = 2.05 + i * 0.95
    w = 4.8 * opacity + 0.3
    rgb = color[0], color[1], color[2]
    rect(s6, 0.6, y, w, 0.7, RGBColor(rgb[0]//4, rgb[1]//4, rgb[2]//4), color)
    tb(s6, f"{name}  ·  {desc}", 0.75, y+0.1, w, 0.55, sz=12, color=color)

# 右侧：应用场景4宫格
scenes = [
    ("文旅融合", "景区AR导览\n非遗体验馆\n数字文创产品", CYAN),
    ("教育传承", "中小学美育课\n高校研究数据库\n在线学习平台", GREEN),
    ("文创产业", "AI图案授权\n国潮品牌联名\n数字藏品", GOLD),
    ("国际传播", "多语言平台\n海外文化中心\n国际博物馆合作", RGBColor(0xBB,0x88,0xFF)),
]
for i, (title, desc, color) in enumerate(scenes):
    col = i % 2
    row = i // 2
    x = 6.2 + col * 3.55
    y = 1.3 + row * 2.9
    rect(s6, x, y, 3.3, 2.7, RGBColor(0x08,0x18,0x28), color)
    hline(s6, x, y, 3.3, color, 3)
    tb(s6, title, x+0.15, y+0.1, 3.0, 0.5, sz=16, bold=True, color=color)
    tb(s6, desc, x+0.15, y+0.65, 3.0, 1.8, sz=12, color=GRAY)

# ════════════════════════════════════════════════════════════════
# Slide 7 — 社会价值（4个大图标）
# ════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank(prs))
bg(s7)

hline(s7, 0, 1.1, 13.33, CYAN)
tb(s7, "社会价值与意义", 0.5, 0.1, 9.0, 0.9, sz=34, bold=True, color=WHITE)
tb(s7, "文化自信  ·  创新发展  ·  国家战略  ·  文明互鉴", 0.5, 0.8, 11.0, 0.4,
   sz=16, color=CYAN)

values = [
    ("🔴", "文化自信", "让年轻一代\n亲近传统美学\n增强民族认同", CYAN),
    ("🟡", "两创精神", "传统技艺 × 现代科技\n创造性转化\n创新性发展", GREEN),
    ("🔵", "国家战略", "数字中国 · 文旅融合\n乡村振兴\n产学研一体", GOLD),
    ("🟢", "文明互鉴", "数字平台跨越国界\n中华美学走向世界\n贡献文化软实力", RGBColor(0xBB,0x88,0xFF)),
]
for i, (icon, title, desc, color) in enumerate(values):
    x = 0.4 + i * 3.2
    # 外框
    rect(s7, x, 1.3, 3.0, 5.7, RGBColor(0x0A,0x1A,0x28), color)
    # 顶部色条
    rect(s7, x, 1.3, 3.0, 0.12, color)
    # 大圆
    oval(s7, x+0.75, 1.55, 1.5, 1.5, color)
    tb(s7, title[0], x+0.75, 1.6, 1.5, 1.4, sz=42, bold=True,
       color=BG, align=PP_ALIGN.CENTER)
    tb(s7, title, x+0.1, 3.2, 2.8, 0.55,
       sz=17, bold=True, color=color, align=PP_ALIGN.CENTER)
    hline(s7, x+0.3, 3.85, 2.4, color)
    tb(s7, desc, x+0.1, 4.0, 2.8, 2.8,
       sz=13, color=GRAY, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
# Slide 8 — 结语
# ════════════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank(prs))
bg(s8)

# 背景装饰线条
for i in range(8):
    hline(s8, 0, 0.9*i, 13.33, RGBColor(0x0D,0x1F,0x2A))

# 顶部装饰
hline(s8, 0.5, 1.2, 12.33, CYAN, 3)

# 主标语
tb(s8, "让千年剪纸技艺", 0.5, 1.5, 12.33, 1.5,
   sz=52, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
tb(s8, "在数字时代焕发新生", 0.5, 3.0, 12.33, 1.5,
   sz=52, bold=True, color=CYAN, align=PP_ALIGN.CENTER)

hline(s8, 0.5, 4.6, 12.33, CYAN, 3)

# 三点总结
summary = [
    ("AI", "赋能技艺  降低门槛", CYAN),
    ("AR", "活化展示  触达人心", GREEN),
    ("云端", "永久留存  守护根脉", GOLD),
]
for i, (tag, text, color) in enumerate(summary):
    x = 1.2 + i * 3.7
    oval(s8, x, 5.0, 1.0, 1.0, color)
    tb(s8, tag, x, 5.1, 1.0, 0.8, sz=18, bold=True,
       color=BG, align=PP_ALIGN.CENTER)
    tb(s8, text, x+1.1, 5.2, 2.5, 0.6, sz=14, color=WHITE)

tb(s8, "赛题5 · 开放赛题  |  第三届中国研究生文化中国两创大赛  |  2026",
   0.5, 7.05, 12.33, 0.4, sz=11, color=DIM, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
out = "/home/user/math_pro/赛题5-科技风版.pptx"
prs.save(out)
print(f"生成完成：{out}")
print(f"共 {len(prs.slides)} 张幻灯片")
