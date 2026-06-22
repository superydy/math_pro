#!/usr/bin/env python3
"""
赛题5 PPT v5 — 磅礴大气·历史厚重感
泥河湾遗址 · 深色青铜风格
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ─── 青铜厚重色系 ──────────────────────────────────────────
DARK     = RGBColor(0x10, 0x0A, 0x04)   # 极深暖黑（主背景）
DARK2    = RGBColor(0x1C, 0x12, 0x08)   # 深棕黑（卡片底）
DARK3    = RGBColor(0x28, 0x1C, 0x0E)   # 稍浅（区块底）
BRONZE   = RGBColor(0xC8, 0x94, 0x2A)   # 青铜金（主装饰）
BRONZE2  = RGBColor(0x8B, 0x60, 0x14)   # 深青铜
BRONZE3  = RGBColor(0xE8, 0xB8, 0x50)   # 亮青铜
CREAM    = RGBColor(0xF0, 0xE6, 0xCC)   # 羊皮纸白（主文字）
CREAM2   = RGBColor(0xC8, 0xB8, 0x98)   # 暗奶白（次文字）
RUST     = RGBColor(0x9A, 0x38, 0x18)   # 锈红
STONE    = RGBColor(0x78, 0x68, 0x50)   # 石灰色
STONE2   = RGBColor(0x48, 0x3C, 0x2C)   # 深石

def blank(prs): return prs.slide_layouts[6]

def bg(slide, color=DARK):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def tb(slide, text, l, t, w, h, sz=16, bold=False,
       color=CREAM, align=PP_ALIGN.LEFT):
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

def rect(slide, l, t, w, h, fill, line=None, lw=1.0):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    return s

def hline(slide, l, t, w, color=BRONZE, h=0.022):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()

def imgbox(slide, l, t, w, h, label, sub=''):
    rect(slide, l, t, w, h, DARK2, BRONZE2, 1.0)
    # 内框
    inner = slide.shapes.add_shape(
        1, Inches(l+0.08), Inches(t+0.08), Inches(w-0.16), Inches(h-0.16))
    inner.fill.background()
    inner.line.color.rgb = STONE
    inner.line.width = Pt(0.75)
    tb(slide, '▣', l+w/2-0.35, t+h/2-0.65, 0.7, 0.65,
       sz=22, align=PP_ALIGN.CENTER, color=STONE)
    tb(slide, label, l+0.1, t+h/2-0.08, w-0.2, 0.45,
       sz=12, bold=True, color=BRONZE, align=PP_ALIGN.CENTER)
    if sub:
        tb(slide, sub, l+0.1, t+h/2+0.37, w-0.2, 0.45,
           sz=10, color=STONE, align=PP_ALIGN.CENTER)

def corner_deco(slide, l, t, size=0.3):
    """四角青铜装饰"""
    for dl, dt in [(0,0),(size,0),(0,size),(size,size)]:
        rect(slide, l+dl, t+dt, size*0.08, size*0.55, BRONZE2)
        rect(slide, l+dl, t+dt, size*0.55, size*0.08, BRONZE2)

def header(slide, title, sub=None):
    rect(slide, 0, 0, 13.33, 1.2, DARK2)
    hline(slide, 0, 0, 13.33, BRONZE, 0.035)
    hline(slide, 0, 1.18, 13.33, BRONZE, 0.035)
    tb(slide, title, 0.5, 0.12, 11.0, 0.75, sz=28, bold=True, color=BRONZE3)
    if sub:
        tb(slide, sub, 0.5, 0.78, 11.0, 0.38, sz=13, color=CREAM2)
    hline(slide, 0, 7.2, 13.33, BRONZE, 0.035)

# ════════════════════════════════════════════════════════
# Slide 1 — 封面（磅礴大气）
# ════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank(prs))
bg(s1)

# 顶底青铜横条
hline(s1, 0, 0,    13.33, BRONZE,  0.055)
hline(s1, 0, 0.06, 13.33, BRONZE2, 0.022)
hline(s1, 0, 7.42, 13.33, BRONZE2, 0.022)
hline(s1, 0, 7.45, 13.33, BRONZE,  0.055)

# 左侧竖线装饰
hline(s1, 0.38, 0.08, 0.022, BRONZE, 7.35)
hline(s1, 0.44, 0.08, 0.022, BRONZE2, 7.35)

# 主背景图区（右侧大图）
imgbox(s1, 5.2, 0.18, 7.9, 7.1,
       '泥河湾遗址 · 地层剖面 / 遗址全景',
       '黄昏光线下的遗址发掘现场\n或地质剖面震撼全景图')

# 左侧文字区
rect(s1, 0.5, 0.18, 4.55, 7.1, DARK2)
hline(s1, 0.5, 0.18, 4.55, BRONZE, 0.035)
hline(s1, 0.5, 7.25, 4.55, BRONZE, 0.035)

tb(s1, '赛题五  ·  开放赛题', 0.7, 0.38, 4.0, 0.45,
   sz=12, color=STONE, align=PP_ALIGN.LEFT)
hline(s1, 0.7, 0.92, 3.5, BRONZE, 0.022)

# 主标题
tb(s1, '地质记忆', 0.6, 1.05, 4.2, 1.2,
   sz=48, bold=True, color=BRONZE3, align=PP_ALIGN.LEFT)
tb(s1, '文明溯源', 0.6, 2.22, 4.2, 1.2,
   sz=48, bold=True, color=CREAM, align=PP_ALIGN.LEFT)

hline(s1, 0.7, 3.5, 3.5, BRONZE, 0.022)

tb(s1, '泥河湾遗址的数字化保护与传播', 0.68, 3.65, 4.0, 0.6,
   sz=15, color=CREAM2, align=PP_ALIGN.LEFT)

# 关键数字
tb(s1, '166', 0.6, 4.5, 3.0, 1.5, sz=72, bold=True,
   color=BRONZE, align=PP_ALIGN.LEFT)
tb(s1, '万年', 2.55, 5.35, 1.8, 0.65, sz=20, bold=True,
   color=BRONZE3, align=PP_ALIGN.LEFT)
tb(s1, '东亚最早人类活动遗址', 0.68, 6.1, 4.0, 0.45,
   sz=12, color=CREAM2)
tb(s1, '河北 · 张家口 · 阳原县', 0.68, 6.62, 4.0, 0.45,
   sz=12, color=STONE)

# ════════════════════════════════════════════════════════
# Slide 2 — 何为泥河湾（横版大图）
# ════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank(prs))
bg(s2)
header(s2, '泥河湾  ·  东方人类的摇篮',
       '国家级自然保护区 · 全国重点文物保护单位 · 世界著名旧石器时代遗址群')

# 通栏大图
imgbox(s2, 0.4, 1.28, 12.53, 3.5,
       '泥河湾盆地全景 / 发掘现场震撼大图',
       '建议：用最震撼的一张横版全景照片，体现遗址的宏大尺度')

# 底部四组数据
data = [
    ('166万年', '东亚最早\n人类活动证据'),
    ('200余处', '旧石器遗址\n密度全球罕见'),
    ('古湖泊', '温暖湿润环境\n孕育早期人类'),
    ('东方峡谷', '与非洲奥杜威\n齐名的人类圣地'),
]
for i, (num, desc) in enumerate(data):
    x = 0.4 + i * 3.23
    rect(s2, x, 5.0, 3.0, 2.1, DARK2, BRONZE2)
    hline(s2, x, 5.0, 3.0, BRONZE, 0.04)
    tb(s2, num, x, 5.1, 3.0, 0.8, sz=24, bold=True,
       color=BRONZE3, align=PP_ALIGN.CENTER)
    tb(s2, desc, x, 5.95, 3.0, 0.95, sz=12,
       color=CREAM2, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════
# Slide 3 — 地质与文明（历史叙事）
# ════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank(prs))
bg(s3)
header(s3, '地质环境  ·  文明诞生的摇篮',
       '读懂地层，就是读懂祖先为何选择在此生息繁衍')

# 左侧大图
imgbox(s3, 0.4, 1.3, 6.5, 5.75,
       '古湖泊环境复原图 / 地质地层剖面',
       '泥河湾古湖复原画 或\n地层年代示意图（越古越深）')

# 右侧：时间轴式叙述
milestones = [
    ('300万年前', '泥河湾盆地断陷\n形成大型古湖泊',       BRONZE),
    ('200万年前', '湖岸气候温暖湿润\n动植物资源极为丰富', BRONZE2),
    ('166万年前', '东亚最早先民抵达\n在湖岸打制石器定居', BRONZE3),
    ('数千年积累', '地层封存历史密码\n泥河湾层成地质奇观', STONE),
]
for i, (year, desc, color) in enumerate(milestones):
    y = 1.38 + i * 1.42
    # 时间节点
    rect(s3, 7.2, y+0.28, 0.12, 0.85, color)
    rect(s3, 7.0, y+0.4, 0.55, 0.12, color)
    oval_s = s3.shapes.add_shape(9, Inches(7.25), Inches(y+0.3),
                                  Inches(0.45), Inches(0.45))
    oval_s.fill.solid(); oval_s.fill.fore_color.rgb = color
    oval_s.line.fill.background()
    # 文字
    tb(s3, year, 7.65, y+0.12, 2.8, 0.45, sz=14, bold=True, color=color)
    tb(s3, desc, 7.65, y+0.58, 5.2, 0.75, sz=13, color=CREAM2)
    if i < 3:
        hline(s3, 7.2, y+1.38, 0.12, STONE, 0.3)

# ════════════════════════════════════════════════════════
# Slide 4 — 数字技术方案（三栏图）
# ════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank(prs))
bg(s4)
header(s4, '数字化保护与传播方案',
       '三维复原 · VR时空穿越 · 数字博物馆  让166万年前的世界重现')

cols = [
    ('三维地质复原',
     '遗址地层3D建模\n古湖泊景观数字渲染图',
     ['精准复原古地貌', '可视化地层年代', '石器出土空间定位']),
    ('VR时空穿越',
     'VR头盔中的\n166万年前泥河湾古湖',
     ['沉浸感受祖先环境', '与虚拟先民互动', '科学严谨场景重建']),
    ('数字博物馆',
     '线上博物馆界面\n文物高清3D展示',
     ['文物三维可交互', '地质科普图文并茂', '面向全球免费开放']),
]
for i, (title, imglabel, pts) in enumerate(cols):
    x = 0.4 + i * 4.3
    rect(s4, x, 1.28, 4.0, 6.1, DARK2, BRONZE2, 1.0)
    hline(s4, x, 1.28, 4.0, BRONZE, 0.04)
    imgbox(s4, x+0.08, 1.38, 3.84, 3.0, title, imglabel)
    hline(s4, x, 4.45, 4.0, BRONZE2, 0.022)
    tb(s4, title, x+0.15, 4.55, 3.7, 0.52,
       sz=16, bold=True, color=BRONZE3, align=PP_ALIGN.CENTER)
    hline(s4, x+0.3, 5.12, 3.4, STONE, 0.015)
    for j, pt in enumerate(pts):
        tb(s4, f'◈  {pt}', x+0.2, 5.22+j*0.66, 3.6, 0.62, sz=13, color=CREAM2)

# ════════════════════════════════════════════════════════
# Slide 5 — 复原成果展示（震撼大图页）
# ════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank(prs))
bg(s5)

hline(s5, 0, 0,    13.33, BRONZE,  0.04)
hline(s5, 0, 7.46, 13.33, BRONZE,  0.04)

# 顶部标题（覆盖在图上方）
rect(s5, 0, 0.04, 13.33, 1.0, DARK2)
hline(s5, 0, 1.0, 13.33, BRONZE, 0.025)
tb(s5, '数字复原成果  ·  重见166万年前的东方大地',
   0.5, 0.1, 12.33, 0.82, sz=26, bold=True,
   color=BRONZE3, align=PP_ALIGN.CENTER)

# 左大图（主复原图）
imgbox(s5, 0.3, 1.15, 8.1, 5.9,
       '【核心图】古湖泊 + 早期人类活动\n数字复原全景渲染图',
       '这是整个PPT最重要的一张图\n建议委托专业美工或使用AI绘图生成\n画面：蓝色古湖 + 黄土高地 + 先民剪影')

# 右侧两张辅图
imgbox(s5, 8.65, 1.15, 4.4, 2.8,
       '马圈沟出土石器文物',
       '高清文物照片或3D扫描模型截图')
imgbox(s5, 8.65, 4.2, 4.4, 2.85,
       '地层年代可视化图',
       '将地质分层与人类活动时间线\n对应呈现在同一图中')

# ════════════════════════════════════════════════════════
# Slide 6 — 应用场景（四图网格）
# ════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank(prs))
bg(s6)
header(s6, '应用场景',
       '让地质遗址走进课堂 · 走进景区 · 走向世界')

scenes = [
    ('科普教育',
     '青少年VR体验课堂\n或博物馆互动展示',
     '课堂里的\n百万年时光'),
    ('文化旅游',
     '遗址公园游览\nAR导览手机截图',
     '脚下的历史\n触手可及'),
    ('学术研究',
     '科研人员使用\n3D地质数据平台',
     '跨学科研究\n数据共享平台'),
    ('国际传播',
     '海外展览现场\n或国际学术报告',
     '向世界讲述\n东方文明起源'),
]
for i, (title, imglabel, caption) in enumerate(scenes):
    col, row = i % 2, i // 2
    x = 0.35 + col * 6.5
    y = 1.28 + row * 3.0
    rect(s6, x, y, 6.2, 2.98, DARK2, BRONZE2, 0.8)
    hline(s6, x, y, 6.2, BRONZE, 0.035)
    imgbox(s6, x+0.08, y+0.08, 3.9, 2.75,
           title, imglabel)
    hline(s6, x+4.1, y+0.5, 0.022, STONE, 1.8)
    tb(s6, title, x+4.2, y+0.2, 1.82, 0.5,
       sz=15, bold=True, color=BRONZE3)
    tb(s6, caption, x+4.2, y+0.8, 1.82, 0.9,
       sz=12, color=CREAM2)

# ════════════════════════════════════════════════════════
# Slide 7 — 价值意义（大字排版）
# ════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank(prs))
bg(s7)
header(s7, '价值与意义',
       '文化自信 · 科学精神 · 遗产保护 · 文明互鉴')

# 左：大引言
rect(s7, 0.35, 1.28, 6.0, 5.9, DARK2, BRONZE2)
hline(s7, 0.35, 1.28, 6.0, BRONZE, 0.05)
tb(s7, '"', 0.55, 1.4, 1.0, 1.1, sz=72, bold=True,
   color=BRONZE, align=PP_ALIGN.LEFT)
tb(s7, '读懂\n地层，\n就是\n读懂祖先', 0.55, 2.2, 5.5, 3.5,
   sz=36, bold=True, color=CREAM, align=PP_ALIGN.LEFT)
hline(s7, 0.6, 5.85, 4.5, BRONZE2, 0.022)
tb(s7, '——泥河湾，166万年人类史的地质档案', 0.6, 6.05, 5.5, 0.5,
   sz=12, color=STONE, align=PP_ALIGN.LEFT)

# 右：四个价值点
vals = [
    ('文化自信', '166万年证明中华文明根脉深厚\n数字化让更多人看见这段历史'),
    ('科文融合', '地质科学×人文历史跨界结合\n用科学讲述文明起源的故事'),
    ('遗产保护', '自然侵蚀不可逆\n数字永久留存不可再生遗产信息'),
    ('文明互鉴', '与非洲奥杜威峡谷对话\n展示人类共同起源与多元文明'),
]
for i, (title, desc) in enumerate(vals):
    y = 1.35 + i * 1.45
    rect(s7, 6.65, y, 6.3, 1.3, DARK3, BRONZE2, 0.8)
    hline(s7, 6.65, y, 6.3, BRONZE, 0.035)
    tb(s7, title, 6.85, y+0.1, 5.9, 0.48, sz=16, bold=True, color=BRONZE3)
    tb(s7, desc,  6.85, y+0.62, 5.9, 0.6,  sz=12, color=CREAM2)

# ════════════════════════════════════════════════════════
# Slide 8 — 结语（史诗收尾）
# ════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank(prs))
bg(s8)

hline(s8, 0, 0,    13.33, BRONZE,  0.055)
hline(s8, 0, 0.06, 13.33, BRONZE2, 0.022)
hline(s8, 0, 7.42, 13.33, BRONZE2, 0.022)
hline(s8, 0, 7.45, 13.33, BRONZE,  0.055)

# 全幅背景图
imgbox(s8, 0.35, 0.12, 12.63, 5.25,
       '结语大图  ·  最震撼的一张',
       '建议：泥河湾遗址日落/黄昏全景\n或数字复原图与真实遗址的对比拼接')

# 底部文字压图
rect(s8, 0, 5.5, 13.33, 1.88, DARK2)
hline(s8, 0, 5.5, 13.33, BRONZE, 0.04)

tb(s8, '让166万年前的地质记忆', 0.5, 5.6, 12.33, 0.75,
   sz=30, bold=True, color=BRONZE3, align=PP_ALIGN.CENTER)
tb(s8, '在数字时代重新照亮中华文明的来路', 0.5, 6.32, 12.33, 0.65,
   sz=24, bold=True, color=CREAM, align=PP_ALIGN.CENTER)

hline(s8, 3.5, 7.12, 6.33, BRONZE2, 0.015)
tb(s8, '赛题5 · 开放赛题  |  第三届中国研究生文化中国两创大赛  |  2026',
   0.5, 7.2, 12.33, 0.28, sz=11, color=STONE, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════
out = '/home/user/math_pro/赛题5-磅礴历史版.pptx'
prs.save(out)
print(f'生成完成：{out}')
print(f'共 {len(prs.slides)} 张幻灯片')
