from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER
import os

OUT = "q3/audit_plots.pdf"

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

styles = getSampleStyleSheet()

title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=16, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
sub_s = ParagraphStyle("S", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
    textColor=colors.grey, spaceAfter=10)
cap_s = ParagraphStyle("C", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER,
    textColor=colors.grey, spaceAfter=12, leading=12)

story = []

story += [
    Paragraph("Q3 Audit Plots", title_s),
    Paragraph("Bias Audit: Common Voice Dataset", sub_s),
    Spacer(1, 10),
]

plots = [
    ("q3/results/gender_distribution.png", "Figure 1: Gender Distribution"),
    ("q3/results/age_distribution.png", "Figure 2: Age Distribution"),
    ("q3/results/dialect_distribution.png", "Figure 3: Dialect Distribution"),
]

for i, (path, caption) in enumerate(plots):
    if os.path.exists(path):
        img = Image(path, width=16*cm, height=10*cm, kind="proportional")
        story.append(img)
        story.append(Paragraph(caption, cap_s))
        if i < len(plots) - 1:
            story.append(Spacer(1, 15))
    else:
        story.append(Paragraph(f"[Plot not found: {path}]", cap_s))

story.append(Spacer(1, 10))
story.append(Paragraph(
    "Generated from q3/audit.py — Speech Understanding Assignment Q3",
    ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                   textColor=colors.grey, alignment=TA_CENTER)
))

doc.build(story)
print("Generated: " + OUT)
