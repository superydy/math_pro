#!/usr/bin/env python3
"""
赛题5 PPT v4 — 地质记忆·文明溯源
泥河湾遗址数字化保护与传播
红金风格，图片为主，文字极简
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

RED    = RGBColor(0xC0, 0x20, 0x20)
RED2   = RGBColor(0x8B, 0x00, 0x00)
GOLD   = RGBColor(0xD4, 0xAF, 0x37)
BG     = RGBColor(0xFD, 0xF6, 0xEC)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DKGRAY = RGBColor(0x2C, 0x2C, 0x2C)
LGRAY  = RGBColor(0xAA, 0xAA, 0xAA)
EARTH  = RGBColor(0x8B, 0x60, 0x20)   # 土黄（地质感）
EARTH2 = RGBColor(0xC8, 0x96, 0x40)   # 浅土黄
STONE  = RGBColor(0x6B, 0x5B, 0x45)   # 石灰色

def blank(prs):  return prs.slide_layouts[6]

def bg(slide, color=BG):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def tb(slide, text, l, t, w, h, sz=16, bold=False,
       color=DKGRAY, align=PP_ALIGN.LEFT):
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

def imgbox(slide, l, t, w, h, label, sub=''):
    rect(slide, l, t, w, h, RGBColor(0xF2, 0xEB, 0xDE), EARTH2, 1.0)
    inner = slide.shapes.add_shape(
        1, Inches(l+0.07), Inches(t+0.07), Inches(w-0.14), Inches(h-0.14))
    inner.fill.background()
    inner.line.color.rgb = EARTH2
    inner.line.width = Pt(0.75)
    # icon
    tb(slide, '🖼', l+w/2-0.35, t+h/2-0.6, 0.7, 0.65,
       sz=24, align=PP_ALIGN.CENTER, color=EARTH2)
    tb(slide, label, l+0.1, t+h/2-0.05, w-0.2, 0.45,
       sz=12, bold=True, color=RED, align=PP_ALIGN.CENTER)
    if sub:
        tb(slide, sub, l+0.1, t+h/2+0.4, w-0.2, 0.45,
           sz=10, color=LGRAY, align=PP_ALIGN.CENTER)

def header(slide, title, sub=None):
    rect(slide, 0, 0, 13.33, 1.15, RED)
    hline(slide, 0, 1.15, 13.33, GOLD, 0.04)
    tb(slide, title, 0.45, 0.1, 11.0, 0.75, sz=30, bold=True, color=WHITE)
    if sub:
        tb(slide, sub, 0.45, 0.78, 11.0, 0.38, sz=14,
           color=RGBColor(0xFF, 0xDD, 0xCC))
    hline(slide, 0, 7.2, 13.33, GOLD, 0.04)

# ═══════════════════════════════════════════════════════
# Slide 1 — 封面
# ═══════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank(prs))
bg(s1)

rect(s1, 0, 0, 5.6, 7.5, RED)
hline(s1, 5.6, 0, 0.07, GOLD, 7.5)

# 左侧文字
tb(s1, '赛题 5  ·  开放赛题', 0.4, 0.5, 4.8, 0.45,
   sz=13, color=RGBColor(0xFF,0xCC,0xAA))
hline(s1, 0.4, 1.05, 4.0, GOLD, 0.04)
tb(s1, '地质记忆·文明溯源', 0.35, 1.2, 5.0, 1.2,
   sz=30, bold=True, color=WHITE)
tb(s1, '泥河湾遗址的数字化\n保护与传播', 0.35, 2.5, 5.0, 1.1,
   sz=19, color=RGBColor(0xFF,0xEE,0xDD))
hline(s1, 0.4, 3.75, 4.0, GOLD, 0.04)
tb(s1, '河北 · 张家口 · 阳原县', 0.4, 3.9, 4.5, 0.45,
   sz=13, color=GOLD)
tb(s1, '距今166万年  东亚最早人类活动遗址', 0.4, 4.4, 4.8, 0.45,
   sz=12, color=RGBColor(0xFF,0xCC,0xAA))
tb(s1, '第三届中国研究生文化中国两创大赛  2026', 0.4, 6.6, 4.8, 0.45,
   sz=11, color=RGBColor(0xCC,0x88,0x66))

# 右侧：遗址主图
imgbox(s1, 5.85, 0.25, 7.1, 5.2,
       '泥河湾遗址全景图',
       '建议：遗址发掘现场航拍图\n或地质地层剖面照片')

# 右下：三个标签
tags = ['数字三维复原', 'VR时空穿越', '地质文化科普']
for i, t in enumerate(tags):
    x = 5.9 + i * 2.42
    rect(s1, x, 5.7, 2.2, 0.68, RGBColor(0xF5,0xEC,0xE0), RED)
    tb(s1, t, x, 5.72, 2.2, 0.65, sz=13, bold=True,
       color=RED, align=PP_ALIGN.CENTER)

hline(s1, 0, 7.2, 13.33, GOLD, 0.04)

# ═══════════════════════════════════════════════════════
# Slide 2 — 泥河湾是什么（大图展示）
# ═══════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank(prs))
bg(s2)
header(s2, '泥河湾  ·  中华文明的地质摇篮',
       '166万年前 · 东亚最早人类 · 河北张家口阳原县')

# 左大图
imgbox(s2, 0.4, 1.3, 7.5, 5.7,
       '泥河湾盆地地质地层剖面图',
       '展示泥河湾层的典型剖面\n颜色分层代表不同地质年代')

# 右侧：四组关键事实
facts = [
    ('166万年', '马圈沟遗址石器年代\n东亚迄今最早人类活动证据'),
    ('200+处', '盆地内已发现旧石器\n遗址数量，密度全球罕见'),
    ('古湖泊', '古泥河湾湖孕育早期人类\n温暖湿润的地质环境是关键'),
    ('世界级', '与非洲奥杜威峡谷齐名\n被誉为"东方人类故乡"'),
]
for i, (num, desc) in enumerate(facts):
    y = 1.35 + i * 1.47
    rect(s2, 8.3, y, 4.6, 1.3, RGBColor(0xFF,0xF2,0xE5), RED)
    tb(s2, num, 8.5, y+0.08, 2.0, 0.7, sz=26, bold=True, color=RED)
    tb(s2, desc, 8.5, y+0.72, 4.1, 0.55, sz=12, color=DKGRAY)

# ═══════════════════════════════════════════════════════
# Slide 3 — 地质与文明的关系（核心逻辑）
# ═══════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank(prs))
bg(s3)
header(s3, '地质环境  如何孕育了最早的东亚人类',
       '古气候 · 古湖泊 · 古生态 · 人类选择在这里生存的理由')

# 中间：逻辑链（横向流程）
steps = [
    ('古湖泊\n形成', '泥河湾盆地断陷\n形成大型古湖'),
    ('温暖湿润\n气候', '湖岸水草丰美\n动物资源丰富'),
    ('早期人类\n迁入', '166万年前\n先民在此定居'),
    ('石器文化\n诞生', '打制石器技术\n华北文化源头'),
]
for i, (title, desc) in enumerate(steps):
    x = 0.5 + i * 3.15
    # 方块
    rect(s3, x, 2.3, 2.8, 2.0, RGBColor(0xFF,0xF0,0xE0), RED)
    rect(s3, x, 2.3, 2.8, 0.55, RED)
    tb(s3, title, x, 2.32, 2.8, 0.52, sz=15, bold=True,
       color=WHITE, align=PP_ALIGN.CENTER)
    tb(s3, desc, x+0.15, 2.95, 2.5, 1.2, sz=13, color=DKGRAY)
    # 箭头（除最后一个）
    if i < 3:
        tb(s3, '▶', x+2.8, 3.05, 0.35, 0.9, sz=20,
           color=GOLD, align=PP_ALIGN.CENTER)

# 上方：图片区
imgbox(s3, 0.4, 1.25, 12.53, 0.95,
       '泥河湾古湖泊复原示意图 / 遗址出土石器与化石图',
       '建议横幅图：古湖景观复原画 或 马圈沟遗址出土文物')

# 下方：结论
rect(s3, 0.4, 4.5, 12.53, 1.1, RGBColor(0xF8,0xF0,0xE5))
hline(s3, 0.4, 4.5, 12.53, RED, 0.04)
tb(s3, '结论：泥河湾的地质记录，就是中华文明起源的自然档案',
   0.6, 4.62, 12.1, 0.5, sz=18, bold=True, color=RED, align=PP_ALIGN.CENTER)
tb(s3, '读懂地层，就是读懂祖先选择在这片土地上生息繁衍的理由',
   0.6, 5.15, 12.1, 0.4, sz=14, color=DKGRAY, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════
# Slide 4 — 数字技术方案（三大支柱，各含图）
# ═══════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank(prs))
bg(s4)
header(s4, '数字化解决方案',
       '三维复原 · VR穿越 · 数字博物馆  — 让166万年前的景象重现')

cols = [
    ('三维地质复原',
     '遗址地层3D建模效果图\n或古湖泊数字复原渲染图',
     ['精准复原古地貌景观', '可视化地层年代关系', '展示石器出土空间位置']),
    ('VR时空穿越体验',
     'VR头盔中看到的166万年前\n泥河湾古湖泊景象',
     ['身临其境感受祖先环境', '与虚拟先民互动体验', '科学严谨的场景重建']),
    ('数字博物馆平台',
     '线上博物馆界面截图\n展示文物高清3D模型',
     ['文物高清3D展示', '地质科普交互图文', '面向全球开放访问']),
]
for i, (title, imglabel, pts) in enumerate(cols):
    x = 0.4 + i * 4.3
    imgbox(s4, x, 1.3, 4.0, 3.05, title, imglabel)
    rect(s4, x, 4.38, 4.0, 0.48, RED)
    tb(s4, title, x, 4.4, 4.0, 0.44, sz=15, bold=True,
       color=WHITE, align=PP_ALIGN.CENTER)
    for j, pt in enumerate(pts):
        tb(s4, f'▸  {pt}', x+0.15, 4.95+j*0.68, 3.7, 0.65, sz=13, color=DKGRAY)

# ═══════════════════════════════════════════════════════
# Slide 5 — 复原效果展示（全图页）
# ═══════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank(prs))
bg(s5)
header(s5, '数字复原成果展示',
       '166万年前的泥河湾 · 在数字世界重新可见')

# 左大图：复原主图
imgbox(s5, 0.4, 1.3, 7.8, 5.7,
       '泥河湾古环境数字复原主图',
       '核心图：古湖泊全景+早期人类活动场景复原渲染图\n（建议委托美工或用AI绘图生成）')

# 右上：石器文物图
imgbox(s5, 8.45, 1.3, 4.5, 2.65,
       '马圈沟出土石器文物',
       '高清文物照片\n或3D扫描模型截图')

# 右下：地层对比图
imgbox(s5, 8.45, 4.2, 4.5, 2.8,
       '地质地层与考古层位对应图',
       '将地质年代与人类活动\n在同一图表中可视化展示')

# ═══════════════════════════════════════════════════════
# Slide 6 — 应用场景
# ═══════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank(prs))
bg(s6)
header(s6, '应用场景',
       '科普教育 · 文化旅游 · 学术研究 · 国际传播')

scenes = [
    ('科普教育',
     '中小学生参观遗址博物馆\n或课堂使用VR体验图',
     '青少年通过VR\n亲历166万年前的祖先世界'),
    ('文化旅游',
     '泥河湾遗址公园旅游场景\n游客使用AR导览图',
     '遗址公园AR导览\n让游客读懂脚下的历史'),
    ('学术研究',
     '科研人员使用3D地质\n数据平台工作截图',
     '三维数字平台\n支撑地质与考古跨学科研究'),
    ('国际传播',
     '海外博物馆展览\n或国际学术会议展示图',
     '向世界讲述\n中华文明起源的地质故事'),
]
for i, (title, imglabel, caption) in enumerate(scenes):
    col, row = i % 2, i // 2
    x = 0.4 + col * 6.45
    y = 1.28 + row * 3.0
    imgbox(s6, x, y, 6.1, 2.2, title, imglabel)
    rect(s6, x, y+2.2, 6.1, 0.62, RED)
    tb(s6, title, x+0.1, y+2.22, 2.5, 0.58, sz=14, bold=True, color=WHITE)
    tb(s6, caption, x+2.6, y+2.25, 3.3, 0.55, sz=11, color=RGBColor(0xFF,0xDD,0xCC))

# ═══════════════════════════════════════════════════════
# Slide 7 — 社会价值
# ═══════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank(prs))
bg(s7)
header(s7, '社会价值与意义',
       '文化自信 · 科学传播 · 遗产保护 · 文明互鉴')

values = [
    ('坚定文化自信',
     RED,
     '166万年的人类活动证据\n证明中华文明根脉深厚\n数字化让更多人看见这段历史'),
    ('科学与文化融合',
     EARTH,
     '地质科学与人文历史跨界\n用严谨的科学语言\n讲述生动的文明故事'),
    ('遗产数字保护',
     RED2,
     '自然营力持续侵蚀遗址\n数字技术永久留存\n不可再生的文化遗产信息'),
    ('推动文明互鉴',
     RGBColor(0x6B,0x5B,0x45),
     '与非洲奥杜威峡谷对话\n展示人类共同起源故事\n贡献中国考古学国际视野'),
]
for i, (title, color, desc) in enumerate(values):
    col, row = i % 2, i // 2
    x = 0.4 + col * 6.45
    y = 1.3 + row * 2.8
    rect(s7, x, y, 6.1, 2.55, RGBColor(0xFF,0xF5,0xEA), color)
    hline(s7, x, y, 6.1, color, 0.06)
    tb(s7, title, x+0.2, y+0.15, 5.7, 0.55, sz=18, bold=True, color=color)
    tb(s7, desc,  x+0.2, y+0.8,  5.7, 1.6,  sz=13, color=DKGRAY)

# ═══════════════════════════════════════════════════════
# Slide 8 — 结语
# ═══════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank(prs))
bg(s8)

rect(s8, 0, 0, 13.33, 1.0, RED)
hline(s8, 0, 1.0, 13.33, GOLD, 0.04)
tb(s8, '地质记忆·文明溯源', 0.5, 0.1, 12.33, 0.8,
   sz=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# 大背景图
imgbox(s8, 0.4, 1.1, 12.53, 4.35,
       '泥河湾 · 结语大图',
       '建议：遗址黄昏全景图，或数字复原与真实遗址的对比图\n视觉冲击力最强的一张')

# 底部三点总结
hline(s8, 0, 5.6, 13.33, GOLD, 0.04)
rect(s8, 0, 5.64, 13.33, 1.56, RGBColor(0xFF,0xF5,0xEA))
summaries = [
    '读懂地层  ·  读懂祖先',
    '数字技术  ·  让历史可见',
    '中华文明  ·  根脉永续',
]
for i, txt in enumerate(summaries):
    x = 0.4 + i * 4.3
    rect(s8, x, 5.74, 3.9, 0.92, RGBColor(0xFF,0xEE,0xE0), RED)
    tb(s8, txt, x, 5.76, 3.9, 0.88, sz=14, bold=True,
       color=RED, align=PP_ALIGN.CENTER)

hline(s8, 0, 7.2, 13.33, GOLD, 0.04)

# ═══════════════════════════════════════════════════════
out = '/home/user/math_pro/赛题5-地质文明版.pptx'
prs.save(out)
print(f'生成完成：{out}')
print(f'共 {len(prs.slides)} 张幻灯片')
