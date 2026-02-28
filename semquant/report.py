from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import pagesizes
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
                return f"{v:.3e}"
            return f"{v:.4g}"
        return str(v)
    except Exception:
        return str(v)


def build_pdf_report(
    pdf_path: str | Path,
    title: str,
    summary: Dict[str, Any],
    key_figures: List[str | Path],
) -> None:
    """Create a compact PDF report with summary metrics and selected figures."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=pagesizes.letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    # Summary table
    rows = [["Metric", "Value"]]
    for k in sorted(summary.keys()):
        rows.append([k, _fmt(summary[k])])

    table = Table(rows, colWidths=[3.2 * inch, 2.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    # Figures
    story.append(Paragraph("Key figures", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    for fig in key_figures:
        p = Path(fig)
        if not p.exists():
            continue
        try:
            img = RLImage(str(p))
            img.drawHeight = 3.0 * inch
            img.drawWidth = 5.8 * inch
            story.append(img)
            story.append(Spacer(1, 0.15 * inch))
        except Exception:
            # Ignore bad images
            continue

    doc.build(story)
