#!/usr/bin/env python3
"""
赛题5 PPT v3 — 红金风格 + 图片为主版
保留原红金配色，每页以图片占位为核心，文字极简
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ─── 颜色 ────────────────────────────────────────────────────
RED    = RGBColor(0xC0, 0x20, 0x20)
RED2   = RGBColor(0x8B, 0x00, 0x00)
GOLD   = RGBColor(0xD4, 0xAF, 0x37)
GOLD2  = RGBColor(0xFF, 0xD7, 0x00)
BG     = RGBColor(0xFD, 0xF6, 0xEC)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GRAY   = RGBColor(0x55, 0x55, 0x55)
LGRAY  = RGBColor(0xAA, 0xAA, 0xAA)
DKGRAY = RGBColor(0x2C, 0x2C, 0x2C)

def blank(prs):
    return prs.slide_layouts[6]

def bg(slide, color=BG):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def tb(slide, text, l, t, w, h, sz=16, bold=False, color=DKGRAY,
       align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(sz)
    r.font.bold = bold
    r.font.color.rgb = color
    return box

def rect(slide, l, t, w, h, fill, line=None, lw=1.5):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    return s

def hline(slide, l, t, w, color=GOLD, h=0.025):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()

def imgbox(slide, l, t, w, h, label, sub=""):
    """图片占位框：虚线边框 + 居中说明文字"""
    # 底色
    rect(slide, l, t, w, h, RGBColor(0xF5, 0xEC, 0xE0))
    # 内边框（用偏移稍小的框模拟虚线感）
    inner = slide.shapes.add_shape(1,
        Inches(l+0.06), Inches(t+0.06),
        Inches(w-0.12), Inches(h-0.12))
    inner.fill.background()
    inner.line.color.rgb = RGBColor(0xCC, 0x88, 0x44)
    inner.line.width = Pt(1.0)
    # 图标
    tb(slide, "🖼", l + w/2 - 0.4, t + h/2 - 0.55, 0.8, 0.7,
       sz=28, align=PP_ALIGN.CENTER, color=RGBColor(0xCC,0x88,0x44))
    # 主标签
    tb(slide, label, l, t + h/2 - 0.02, w, 0.45,
       sz=12, bold=True, color=RED, align=PP_ALIGN.CENTER)
    if sub:
        tb(slide, sub, l, t + h/2 + 0.42, w, 0.4,
           sz=10, color=LGRAY, align=PP_ALIGN.CENTER)

def header(slide, title, sub=None):
    rect(slide, 0, 0, 13.33, 1.15, RED)
    hline(slide, 0, 1.15, 13.33, GOLD, 0.04)
    tb(slide, title, 0.45, 0.12, 10.5, 0.75, sz=30, bold=True, color=WHITE)
    if sub:
        tb(slide, sub, 0.45, 0.78, 10.5, 0.38,
           sz=14, color=RGBColor(0xFF,0xDD,0xCC))
    hline(slide, 0, 7.2, 13.33, GOLD, 0.04)

# ════════════════════════════════════════════════════════════════
# Slide 1 — 封面
# ════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank(prs))
bg(s1)

# 左侧红色色块
rect(s1, 0, 0, 5.6, 7.5, RED)
hline(s1, 5.6, 0, 0.06, GOLD, 7.5)

# 左侧文字
tb(s1, "赛题 5  ·  开放赛题", 0.4, 0.55, 4.8, 0.5,
   sz=13, color=RGBColor(0xFF,0xCC,0xAA))
hline(s1, 0.4, 1.15, 4.0, GOLD, 0.04)
tb(s1, "数字赋能·剪映千年", 0.35, 1.3, 5.0, 1.35,
   sz=32, bold=True, color=WHITE)
tb(s1, "蔚县剪纸的数字化活化\n与创新传播", 0.35, 2.7, 5.0, 1.1,
   sz=18, color=RGBColor(0xFF,0xEE,0xDD))
hline(s1, 0.4, 3.95, 4.0, GOLD, 0.04)
tb(s1, "第三届中国研究生\n文化中国两创大赛", 0.4, 4.15, 4.6, 0.9,
   sz=13, color=RGBColor(0xFF,0xCC,0xAA))
tb(s1, "2026", 0.4, 5.2, 2.0, 0.6, sz=13, color=GOLD)

# 右侧：剪纸主图占位（大）
imgbox(s1, 5.85, 0.3, 7.1, 5.1,
       "蔚县剪纸代表作品图", "建议：放一张色彩鲜艳的剪纸全图")

# 右下：三个小标签
for i, (lbl, clr) in enumerate([
    ("AI 辅助设计", RED),
    ("AR 沉浸体验", RED2),
    ("数字档案留存", RED),
]):
    x = 5.9 + i * 2.4
    rect(s1, x, 5.7, 2.15, 0.7, RGBColor(0xF5,0xEC,0xE0), RED)
    tb(s1, lbl, x, 5.72, 2.15, 0.65, sz=13, bold=True, color=RED,
       align=PP_ALIGN.CENTER)

hline(s1, 0, 7.2, 13.33, GOLD, 0.04)

# ════════════════════════════════════════════════════════════════
# Slide 2 — 蔚县剪纸之美（四图展示）
# ════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank(prs))
bg(s2)
header(s2, "蔚县剪纸  ·  千年技艺之美", "国家级非遗 · 500年历史 · 点色彩绘 · 年产5000万张")

# 四张剪纸图，2×2
pics = [
    ("戏曲人物类剪纸",  "如《贵妃醉酒》《霸王别姬》"),
    ("民俗吉祥类剪纸",  "如《莲年有余》《喜上眉梢》"),
    ("花卉动物类剪纸",  "如牡丹、凤凰、蝴蝶等"),
    ("点色彩绘工艺特写", "展示刻刀 + 点色工序"),
]
for i, (lbl, sub) in enumerate(pics):
    col, row = i % 2, i // 2
    x = 0.4 + col * 6.45
    y = 1.35 + row * 2.95
    imgbox(s2, x, y, 6.1, 2.75, lbl, sub)

# ════════════════════════════════════════════════════════════════
# Slide 3 — 传承困境（图+数据）
# ════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank(prs))
bg(s3)
header(s3, "传承困境", "技艺正在消逝 · 亟待数字化保护")

# 左侧：传承人照片
imgbox(s3, 0.4, 1.3, 5.5, 5.7,
       "传承人工作照", "年迈传承人手持刻刀剪纸的纪实照片")

# 右侧：三组数据
stats = [
    ("<200", "人", "全国核心传承人"),
    ("60+",  "岁", "传承人平均年龄"),
    ("断层", "",   "年轻人参与率极低"),
]
for i, (num, unit, desc) in enumerate(stats):
    y = 1.45 + i * 1.85
    rect(s3, 6.2, y, 6.7, 1.6, RGBColor(0xFF,0xF0,0xE5), RED)
    tb(s3, num,  6.4, y+0.08, 2.0, 1.1, sz=48, bold=True, color=RED)
    tb(s3, unit, 8.4, y+0.65, 1.2, 0.6, sz=18, color=DKGRAY)
    tb(s3, desc, 9.6, y+0.55, 3.0, 0.7, sz=16, bold=True, color=DKGRAY)

# ════════════════════════════════════════════════════════════════
# Slide 4 — 解决方案总览（三栏各含图）
# ════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank(prs))
bg(s4)
header(s4, "数字化解决方案", "AI辅助设计  ·  AR沉浸体验  ·  数字档案库")

cols = [
    ("AI 辅助设计", "AI生成剪纸图案效果图\n或软件界面截图",
     ["图案识别与生成", "降低创作门槛", "矢量图输出"]),
    ("AR 沉浸体验", "手机AR扫描剪纸\n触发动态展示的效果图",
     ["扫描实物即展示", "虚拟剪纸互动", "技艺讲解内嵌"]),
    ("数字档案库", "传承人录制视频\n或数字档案界面截图",
     ["全流程视频记录", "高清3D建模", "永久云端保存"]),
]
for i, (title, imglabel, pts) in enumerate(cols):
    x = 0.4 + i * 4.3
    # 顶部图片
    imgbox(s4, x, 1.3, 4.0, 3.0, title, imglabel)
    # 颜色条
    rect(s4, x, 4.35, 4.0, 0.45, RED)
    tb(s4, title, x, 4.37, 4.0, 0.42, sz=15, bold=True,
       color=WHITE, align=PP_ALIGN.CENTER)
    # 文字要点
    for j, pt in enumerate(pts):
        tb(s4, f"▸  {pt}", x+0.15, 4.88+j*0.65, 3.7, 0.6, sz=13, color=DKGRAY)

# ════════════════════════════════════════════════════════════════
# Slide 5 — AI辅助设计（大图 + 流程）
# ════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank(prs))
bg(s5)
header(s5, "AI 辅助设计系统", "让机器学习千年剪纸的美学密码")

# 左大图
imgbox(s5, 0.4, 1.3, 7.5, 5.7,
       "AI生成剪纸图案对比图",
       "左：传统纹样原图  /  右：AI生成新图案\n或：软件操作界面截图")

# 右侧流程
steps = [
    (GOLD,  "INPUT",  "采集5000+传统纹样"),
    (RED,   "TRAIN",  "深度学习风格建模"),
    (GOLD,  "GEN",    "AI生成新图案"),
    (RED,   "OUTPUT", "矢量图 → 文创生产"),
]
for i, (color, tag, desc) in enumerate(steps):
    y = 1.45 + i * 1.45
    rect(s5, 8.2, y, 1.1, 1.0, color)
    tb(s5, tag, 8.2, y+0.18, 1.1, 0.65, sz=12, bold=True,
       color=WHITE, align=PP_ALIGN.CENTER)
    tb(s5, desc, 9.45, y+0.22, 3.7, 0.6, sz=14, color=DKGRAY)
    if i < 3:
        tb(s5, "▼", 8.55, y+1.0, 0.5, 0.4, sz=14, color=color,
           align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
# Slide 6 — AR体验（大图）
# ════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank(prs))
bg(s6)
header(s6, "AR 沉浸体验平台", "扫一扫，让剪纸动起来、活起来")

# 左：手机AR效果图（大）
imgbox(s6, 0.4, 1.3, 6.0, 5.7,
       "AR体验效果图",
       "手机扫描剪纸后屏幕呈现动态效果的截图\n或AR眼镜体验实景图")

# 右：步骤说明 + 小图
steps = [
    ("01  扫描", "手机对准剪纸作品，自动识别"),
    ("02  触发", "作品背后的故事和工艺动态呈现"),
    ("03  互动", "跟着屏幕一步步学剪纸技法"),
    ("04  分享", "一键发布AR体验至社交媒体"),
]
for i, (title, desc) in enumerate(steps):
    y = 1.4 + i * 1.48
    rect(s6, 6.7, y, 6.2, 1.25, RGBColor(0xFF,0xF0,0xE5), RED)
    tb(s6, title, 6.9, y+0.1, 5.8, 0.48, sz=15, bold=True, color=RED)
    tb(s6, desc,  6.9, y+0.6, 5.8, 0.5,  sz=13, color=DKGRAY)

# ════════════════════════════════════════════════════════════════
# Slide 7 — 应用场景（图片网格）
# ════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank(prs))
bg(s7)
header(s7, "应用场景", "文旅融合  ·  教育传承  ·  文创开发  ·  国际传播")

# 4张场景图
scenes = [
    ("文旅融合",   "景区AR导览 / 非遗体验馆实景图"),
    ("教育传承",   "学生课堂学习剪纸 / 在线学习平台截图"),
    ("文创产业",   "印有剪纸纹样的文创产品图"),
    ("国际传播",   "海外展览 / 外国友人体验剪纸图"),
]
for i, (title, sub) in enumerate(scenes):
    col, row = i % 2, i // 2
    x = 0.4 + col * 6.45
    y = 1.3 + row * 2.95
    imgbox(s7, x, y, 6.1, 2.45, title, sub)
    rect(s7, x, y+2.45, 6.1, 0.42, RED)
    tb(s7, title, x, y+2.47, 6.1, 0.38, sz=14, bold=True,
       color=WHITE, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════
# Slide 8 — 结语（大图 + 口号）
# ════════════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank(prs))
bg(s8)

# 顶部红条
rect(s8, 0, 0, 13.33, 1.0, RED)
hline(s8, 0, 1.0, 13.33, GOLD, 0.04)
tb(s8, "数字赋能·剪映千年", 0.5, 0.08, 12.33, 0.82,
   sz=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# 大背景图
imgbox(s8, 0.4, 1.15, 12.53, 4.3,
       "最终视觉大图",
       "建议：一张融合了传统剪纸纹样 + 数字光效/AR界面的合成图\n体现「传统 × 科技」的视觉冲击感")

# 底部三点总结
hline(s8, 0, 5.6, 13.33, GOLD, 0.04)
rect(s8, 0, 5.64, 13.33, 1.56, RGBColor(0xFF,0xF5,0xEA))
for i, txt in enumerate([
    "AI赋能技艺  ·  降低传承门槛",
    "AR活化展示  ·  触达年轻一代",
    "数字档案留存  ·  守护非遗根脉",
]):
    x = 0.5 + i * 4.3
    rect(s8, x, 5.75, 3.9, 0.9, RGBColor(0xFF,0xF0,0xE5), RED)
    tb(s8, txt, x, 5.77, 3.9, 0.86, sz=13, bold=True,
       color=RED, align=PP_ALIGN.CENTER)

hline(s8, 0, 7.2, 13.33, GOLD, 0.04)

# ════════════════════════════════════════════════════════════════
out = "/home/user/math_pro/赛题5-红金图片版.pptx"
prs.save(out)
print(f"生成完成：{out}")
print(f"共 {len(prs.slides)} 张幻灯片")
