"""Build the standalone speech ('discorso') doc from the presentation's IT notes.

Single source of truth = the speaker notes embedded in pipeline_presentation.pptx,
so the printable speech never drifts from the slides.
"""
import os
from pptx import Presentation
from docx import Document
from docx.shared import Pt, RGBColor

DOCS = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(DOCS, "pipeline_presentation.pptx")
OUT = os.path.join(DOCS, "speech_IT.docx")

prs = Presentation(PPTX)


def slide_title(s):
    for ph in s.placeholders:                       # content layouts: title = idx 0
        if ph.placeholder_format.idx == 0 and ph.has_text_frame and ph.text_frame.text:
            return ph.text_frame.text
    for ph in s.placeholders:                        # cover: first non-empty text
        if ph.has_text_frame and ph.text_frame.text:
            return ph.text_frame.text
    return ""


doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(12)

h = doc.add_heading("Discorso di presentazione", level=0)
doc.add_paragraph(
    "Testo da esporre per ogni slide (le slide sono in inglese, il discorso in "
    "italiano). Una sezione per slide, nell'ordine di proiezione."
).italic = True

for i, s in enumerate(prs.slides, 1):
    title = slide_title(s)
    doc.add_heading(f"Slide {i} — {title}", level=1)
    note = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""
    p = doc.add_paragraph(note.strip())
    p.paragraph_format.space_after = Pt(10)

doc.save(OUT)
print("saved", OUT, "-", len(prs.slides._sldIdLst), "slides")
