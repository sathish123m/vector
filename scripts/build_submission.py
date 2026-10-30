#!/usr/bin/env python
"""Build the Week 1-5 submission package: PDF report, Markdown source, oral script."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from submission_content import REPORT, SCRIPT  # noqa: E402

OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Palette and fonts
# ---------------------------------------------------------------------------
NAVY = colors.HexColor("#0F172A")
INK = colors.HexColor("#1E293B")
SLATE = colors.HexColor("#475569")
MUTED = colors.HexColor("#64748B")
TEAL = colors.HexColor("#0D9488")
TEAL_DARK = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#D97706")
RED = colors.HexColor("#B91C1C")
BORDER = colors.HexColor("#E2E8F0")
LIGHT = colors.HexColor("#F8FAFC")
TEAL_BG = colors.HexColor("#F0FDFA")


def register_fonts() -> None:
    import matplotlib

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Body", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Oblique", str(font_dir / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("Body-BoldOblique", str(font_dir / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(font_dir / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold", italic="Body-Oblique", boldItalic="Body-BoldOblique"
    )


# ---------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------
def inline(text: str, mono_size: float = 8.4) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"`(.+?)`",
        lambda match: f'<font name="Mono" size="{mono_size}">{match.group(1)}</font>',
        text,
    )
    return text


# ---------------------------------------------------------------------------
# PDF styles
# ---------------------------------------------------------------------------
def make_styles() -> dict[str, ParagraphStyle]:
    base = dict(alignment=TA_LEFT, allowWidows=0, allowOrphans=0)
    return {
        "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=22, leading=26, textColor=NAVY, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=11.2, leading=15.5, textColor=SLATE, spaceAfter=8),
        "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14.1, textColor=INK, spaceAfter=6.5, **base),
        "h1": ParagraphStyle(
            "h1", fontName="Body-Bold", fontSize=14.5, leading=18, textColor=NAVY,
            spaceBefore=13, spaceAfter=6, keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Body-Bold", fontSize=11.2, leading=14, textColor=TEAL_DARK,
            spaceBefore=9, spaceAfter=3.5, keepWithNext=1,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Body", textColor=INK, fontSize=9.5, leading=13.6, leftIndent=14, bulletIndent=3,
            bulletFontName="Body", bulletFontSize=8, spaceAfter=3, **base,
        ),
        "subbullet": ParagraphStyle(
            "subbullet", fontName="Body", textColor=INK, fontSize=9.3, leading=13.2, leftIndent=28, bulletIndent=17,
            bulletFontName="Body", bulletFontSize=7, spaceAfter=2, **base,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Body-Oblique", fontSize=7.9, leading=10.6, textColor=MUTED,
            spaceBefore=3, spaceAfter=9, **base,
        ),
        "table": ParagraphStyle("table", fontName="Body", fontSize=8.0, leading=10.1, textColor=INK, **base),
        "table_header": ParagraphStyle("table_header", fontName="Body-Bold", fontSize=8.0, leading=10.1, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=7.4, leading=9.3, textColor=INK),
        "callout_title": ParagraphStyle("callout_title", fontName="Body-Bold", fontSize=9.0, leading=12, textColor=TEAL_DARK),
        "callout_body": ParagraphStyle("callout_body", fontName="Body", fontSize=9.2, leading=13.2, textColor=INK),
        "abstract": ParagraphStyle("abstract", fontName="Body", fontSize=9.4, leading=14.0, textColor=INK),
        "label": ParagraphStyle("label", fontName="Body-Bold", fontSize=8.2, leading=10.5, textColor=NAVY),
        "value": ParagraphStyle("value", fontName="Body", fontSize=8.2, leading=10.5, textColor=SLATE),
    }


def para(text: str, style: ParagraphStyle, mono_size: float = 8.4) -> Paragraph:
    return Paragraph(inline(text, mono_size=mono_size), style)


def image_flowable(path: Path, width_in: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    return Image(str(path), width=width_in * inch, height=width_in * height / width * inch)


def table_flowable(headers, rows, widths, styles) -> Table:
    data = [[Paragraph(inline(str(cell), 7.3), styles["table_header"]) for cell in headers]]
    for row in rows:
        data.append([Paragraph(inline(str(cell), 7.3), styles["table"]) for cell in row])
    table = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.25, BORDER),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
            ]
        )
    )
    return table


def meta_table(items, styles) -> Table:
    data = [
        [Paragraph(inline(key), styles["label"]), Paragraph(inline(value), styles["value"])]
        for key, value in items
    ]
    table = Table(data, colWidths=[1.18 * inch, 5.55 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
            ]
        )
    )
    return table


def abstract_box(text: str, styles) -> Table:
    flowables = [
        Paragraph("ABSTRACT", ParagraphStyle("ab_label", fontName="Body-Bold", fontSize=7.8, leading=10, textColor=TEAL)),
        Spacer(1, 2),
        Paragraph(inline(text), styles["abstract"]),
    ]
    table = Table([[flowables]], colWidths=[6.75 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def callout_box(title: str, text: str, styles) -> Table:
    flowables = [
        Paragraph(inline(title), styles["callout_title"]),
        Spacer(1, 2),
        Paragraph(inline(text), styles["callout_body"]),
    ]
    table = Table([[flowables]], colWidths=[6.75 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TEAL_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def code_box(text: str, styles) -> Table:
    table = Table([[Preformatted(text, styles["code"])]], colWidths=[6.75 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("LINEBEFORE", (0, 0), (0, -1), 2.0, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_pdf(styles) -> Path:
    path = OUT / "report.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.8 * inch, 0.62 * inch, letter[0] - 0.8 * inch, 0.62 * inch)
        canvas.setFont("Body", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.8 * inch, 0.45 * inch, "ChurnVec — Week 1–5 milestone report")
        canvas.drawRightString(letter[0] - 0.8 * inch, 0.45 * inch, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.82 * inch,
        title="ChurnVec: A From-Scratch HNSW Vector Index",
        author="ChurnVec project",
        subject="Week 1–5 milestone report",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])

    flowables = []
    for block in REPORT:
        kind = block[0]
        if kind == "title":
            flowables.append(Paragraph(inline(block[1]), styles["title"]))
        elif kind == "subtitle":
            flowables.append(Paragraph(inline(block[1]), styles["subtitle"]))
            flowables.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=1, spaceAfter=8))
        elif kind == "meta":
            flowables.append(meta_table(block[1], styles))
            flowables.append(Spacer(1, 10))
        elif kind == "abstract":
            flowables.append(abstract_box(block[1], styles))
            flowables.append(Spacer(1, 8))
        elif kind == "callout":
            flowables.append(callout_box(block[1], block[2], styles))
            flowables.append(Spacer(1, 8))
        elif kind == "h1":
            flowables.append(Paragraph(inline(block[1]), styles["h1"]))
        elif kind == "h2":
            flowables.append(Paragraph(inline(block[1]), styles["h2"]))
        elif kind == "p":
            flowables.append(Paragraph(inline(block[1]), styles["body"]))
        elif kind == "bullets":
            for item in block[1]:
                if item.startswith("  - "):
                    flowables.append(Paragraph(inline(item[4:]), styles["subbullet"], bulletText="–"))
                else:
                    flowables.append(Paragraph(inline(item), styles["bullet"], bulletText="•"))
            flowables.append(Spacer(1, 3))
        elif kind == "table":
            headers, rows, widths = block[1], block[2], block[3]
            flowables.append(table_flowable(headers, rows, widths, styles))
            flowables.append(Spacer(1, 8))
        elif kind == "image":
            path_, width_in, caption = block[1], block[2], block[3]
            flowables.append(image_flowable(ROOT / path_, width_in))
            if caption:
                flowables.append(Paragraph(inline(caption), styles["caption"]))
            else:
                flowables.append(Spacer(1, 6))
        elif kind == "figure_note":
            flowables.append(Paragraph(inline(block[1]), styles["caption"]))
        elif kind == "code":
            flowables.append(code_box(block[1], styles))
            flowables.append(Spacer(1, 8))
        elif kind == "pagebreak":
            flowables.append(PageBreak())
        else:
            raise ValueError(f"unknown report block: {kind}")

    doc.build(flowables)
    return path


# ---------------------------------------------------------------------------
# Markdown source
# ---------------------------------------------------------------------------
def build_markdown() -> Path:
    path = OUT / "report.md"
    lines: list[str] = []
    for block in REPORT:
        kind = block[0]
        if kind == "title":
            lines.extend([f"# {block[1]}", ""])
        elif kind == "subtitle":
            lines.extend([f"*{block[1]}*", ""])
        elif kind == "meta":
            for key, value in block[1]:
                lines.append(f"- **{key}:** {value}")
            lines.append("")
        elif kind == "abstract":
            lines.extend(["## Abstract", "", block[1], ""])
        elif kind == "callout":
            lines.extend([f"> **{block[1]}** — {block[2]}", ""])
        elif kind == "h1":
            lines.extend([f"## {block[1]}", ""])
        elif kind == "h2":
            lines.extend([f"### {block[1]}", ""])
        elif kind == "p":
            lines.extend([block[1], ""])
        elif kind == "bullets":
            for item in block[1]:
                if item.startswith("  - "):
                    lines.append(f"  - {item[4:]}")
                else:
                    lines.append(f"- {item}")
            lines.append("")
        elif kind == "table":
            headers, rows = block[1], block[2]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            lines.append("")
        elif kind == "image":
            asset = block[1].replace("submission/", "")
            caption = block[3]
            lines.extend([f"![{caption}]({asset})", ""])
            if caption:
                lines.extend([f"*{caption}*", ""])
        elif kind == "figure_note":
            lines.extend([f"*{block[1]}*", ""])
        elif kind == "code":
            lines.extend(["```text", block[1], "```", ""])
        elif kind == "pagebreak":
            lines.extend(["---", ""])
        else:
            raise ValueError(f"unknown report block: {kind}")
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def build_script() -> Path:
    path = OUT / "oral_script.md"
    path.write_text(SCRIPT)
    return path


def build_readme() -> Path:
    path = OUT / "README.md"
    path.write_text(
        """# Submission package — Week 1–5

| File | Purpose |
|---|---|
| `report.pdf` | Self-contained report for submission (no title page, per request) |
| `report.md` | Editable Markdown source of the same report |
| `oral_script.md` | Timed 10-minute script, delivery notes, and Q&A preparation |
| `assets/` | Figures used in the report |

Regenerate everything with:

```bash
.venv/bin/python scripts/make_submission_assets.py
.venv/bin/python scripts/build_submission.py
```

The report and script are generated from `scripts/submission_content.py`. Measured
numbers come from `reports/phase1/e1_sift100k/result.json` and the committed
run logs; the 28 October rerun is referenced as a reproducibility check, not
as the primary evidence.
"""
    )
    return path


def main() -> None:
    register_fonts()
    styles = make_styles()
    pdf = build_pdf(styles)
    md = build_markdown()
    script = build_script()
    readme = build_readme()
    for path in (pdf, md, script, readme):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
