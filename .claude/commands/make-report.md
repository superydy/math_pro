Generate a professional Word document report (.docx) using python-docx.

## Instructions

The user will describe the report they need. You should:

1. Ask for the following if not provided:
   - Report title and purpose
   - Sections / chapters needed
   - Target reader (teacher / committee / competition judge)
   - Preferred language (Chinese / English)

2. Generate a Python script using `python-docx` to create the report.

3. Run the script to produce a `.docx` file.

4. Save to the current working directory.

## Standard report structure

- Cover page: title, author, date, institution
- Table of contents (manual)
- Introduction
- Main content sections
- Conclusion / Summary
- Appendix (if needed)

## Style guidelines

- Headings: 黑体, level 1 = 16pt blue, level 2 = 13pt dark blue
- Body text: 宋体 11pt, first-line indent for Chinese paragraphs
- Code blocks: Courier New 9pt, grey background
- Professional and clean — no emoji unless requested

## Install if missing

```bash
pip install python-docx
```

$ARGUMENTS
