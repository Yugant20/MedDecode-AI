# pdf_utils.py
import io
from typing import Optional, List, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _draw_paragraph(
    c: canvas.Canvas,
    x: int,
    y: int,
    text: str,
    *,
    font_name: str = "Helvetica",
    font_size: int = 10,
    width: int = 510,
    leading: int = 12,
    bottom_margin: int = 70
) -> int:
    """
    Draws word-wrapped text and returns the new y position.
    Automatically creates a new page if y goes below bottom_margin.
    """
    c.setFont(font_name, font_size)
    words = (text or "").split()
    if not words:
        return y - leading

    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font_name, font_size) <= width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
            if y < bottom_margin:
                c.showPage()
                y = letter[1] - 50
                c.setFont(font_name, font_size)

    if line:
        c.drawString(x, y, line)
        y -= leading
        if y < bottom_margin:
            c.showPage()
            y = letter[1] - 50
            c.setFont(font_name, font_size)

    return y


def _draw_bullets(
    c: canvas.Canvas,
    x: int,
    y: int,
    items: List[str],
    *,
    width: int = 500,
    max_items: int = 10
) -> int:
    """
    Draws bullets with wrapping. Returns new y.
    """
    for item in (items or [])[:max_items]:
        y = _draw_paragraph(c, x, y, f"• {item}", width=width)
        y -= 2
    return y


def build_radiology_summary_pdf_bytes(
    *,
    app_title: str = "MedDecode AI — Radiology Summary",
    report_date: Optional[str],
    modality: str,
    body_part: Optional[str],
    impression: Optional[str],
    findings: Optional[str],
    recommendations: Optional[str],
    urgency_flags: List[str],
    explanation: Dict[str, Any],
) -> bytes:
    """
    Returns PDF bytes.
    explanation keys expected:
      - simple_summary (str)
      - key_points (list[str])
      - questions_for_doctor (list[str])
      - red_flags (list[str])
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 50

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, app_title)
    y -= 24

    c.setFont("Helvetica", 10)
    meta = f"Date: {report_date or 'Unknown'}   Modality: {modality}   Body part: {body_part or 'Unknown'}"
    c.drawString(50, y, meta)
    y -= 18

    # Impression
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Impression (Most Important)")
    y -= 16
    y = _draw_paragraph(c, 50, y, impression or "Not found in report.")

    y -= 8
    # Flags
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Flags / Attention")
    y -= 16
    flags_text = ", ".join(urgency_flags) if urgency_flags else "None detected"
    y = _draw_paragraph(c, 50, y, flags_text)

    y -= 8
    # Simple summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Simple Summary")
    y -= 16
    y = _draw_paragraph(c, 50, y, explanation.get("simple_summary", ""))

    y -= 8
    # Key points
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Key Points")
    y -= 16
    y = _draw_bullets(c, 60, y, explanation.get("key_points", []), max_items=8)

    y -= 8
    # Questions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Questions to Ask Your Doctor")
    y -= 16
    y = _draw_bullets(c, 60, y, explanation.get("questions_for_doctor", []), max_items=10)

    y -= 8
    # Recommendations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Recommendations (from report)")
    y -= 16
    y = _draw_paragraph(c, 50, y, recommendations or "None listed.")

    y -= 8
    # Findings
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Findings (details)")
    y -= 16
    y = _draw_paragraph(c, 50, y, findings or "Not found in report.")

    y -= 10
    # Red flags (general)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "General Red Flags (not diagnosis)")
    y -= 16
    y = _draw_bullets(c, 60, y, explanation.get("red_flags", []), max_items=6)

    # Disclaimer
    disclaimer = (
        "Disclaimer: This summary is informational only and not medical advice. "
        "It does not diagnose conditions. Always consult a qualified clinician."
    )
    c.setFont("Helvetica", 8)
    _draw_paragraph(c, 50, 60, disclaimer, font_size=8, leading=10)

    c.save()
    buf.seek(0)
    return buf.read()
