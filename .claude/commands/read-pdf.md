Read and analyze a PDF file, then produce a structured summary or answer questions about it.

## Instructions

The user will provide a PDF file path or describe what they want from the PDF. You should:

1. Read the PDF using the Read tool (supports up to 20 pages per request; for longer PDFs, read in chunks).

2. Depending on what the user asks, provide one of:
   - **Summary**: key points, main conclusions, structure overview
   - **Q&A**: answer specific questions based on the PDF content
   - **Extraction**: pull out specific information (dates, names, data, requirements)
   - **Translation**: summarize in a different language
   - **Writeup / Notes**: convert the PDF into structured markdown notes

3. Always start by telling the user:
   - Total pages detected
   - Document type (academic paper / report / notice / slides / etc.)
   - Main topic in one sentence

## For competition notices / announcements (比赛通知)

Extract and clearly list:
- 比赛名称
- 报名截止时间
- 比赛时间和地点
- 参赛资格要求
- 提交要求
- 奖项设置
- 联系方式

## For academic papers

Extract:
- Research question / objective
- Methodology
- Key findings
- Conclusions
- Limitations

## Usage examples

```
/read-pdf path/to/file.pdf
/read-pdf 帮我总结这个PDF的主要内容 path/to/file.pdf
/read-pdf 这个比赛通知的报名截止日期是什么时候 path/to/notice.pdf
```

$ARGUMENTS
