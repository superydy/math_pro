Generate a PowerPoint presentation using python-pptx based on the user's request.

## Instructions

The user will describe what they want in the presentation. You should:

1. Ask for the following if not provided:
   - Topic / title of the presentation
   - Number of slides (default: 8-10)
   - Target audience (e.g. teacher, conference, competition)
   - Any specific sections or content to include

2. Generate a Python script using `python-pptx` that creates the presentation.

3. Run the script to produce a `.pptx` file.

4. Save the file to the current working directory with a descriptive filename.

## Template structure to follow

- Slide 1: Title slide (title + subtitle)
- Slide 2: Outline / Table of contents
- Slides 3-N: Content slides (heading + bullet points, max 5 bullets per slide)
- Last slide: Summary / Thank you

## Style guidelines

- Use a clean, professional theme (dark blue title bar, white background)
- Title font: bold, 32pt
- Body font: 18pt, with sub-bullets at 16pt
- Keep text concise — no full sentences on slides
- Add slide numbers in footer

## Code pattern to use

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
```

Install if missing:
```bash
pip install python-pptx
```

## Output

After generating, tell the user:
- The filename and location
- How many slides were created
- How to open it (just double-click on Windows)

$ARGUMENTS
