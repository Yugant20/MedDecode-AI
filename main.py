import os
import uuid
import base64
import traceback
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from google import genai
import fitz  # PyMuPDF

from PIL import Image
import io

# =========================
# APP SETUP
# =========================
app = FastAPI(title="MedDecode AI", version="14.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://yugant20.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STORAGE (DEV: in-memory)
# =========================
REPORTS: Dict[str, Dict[str, Any]] = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUT_DIR = os.path.join(BASE_DIR, "generated")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")  # put logo.png here

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

LOGO_PATH_PRIMARY = os.path.join(ASSETS_DIR, "logo.png")
LOGO_PATH_FALLBACK = os.path.join(BASE_DIR, "logo.png")


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        app.state.client = genai.Client(api_key=api_key)
    else:
        app.state.client = None
        print("⚠️ GEMINI_API_KEY is missing. /process will fail until you set it.")

    # Use pro for best accuracy:
    # PowerShell: $env:GEMINI_MODEL="gemini-2.5-pro"
    app.state.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Render knobs
    app.state.max_pages = int(os.getenv("MD_MAX_PAGES", "3"))
    app.state.render_zoom = float(os.getenv("MD_RENDER_ZOOM", "3.8"))  # higher = better fracture detail
    app.state.max_total_images = int(os.getenv("MD_MAX_IMAGES", "10"))  # allow more crops

    if os.path.exists(LOGO_PATH_PRIMARY):
        print(f"✅ Logo found: {LOGO_PATH_PRIMARY}")
    elif os.path.exists(LOGO_PATH_FALLBACK):
        print(f"✅ Logo found: {LOGO_PATH_FALLBACK}")
    else:
        print("⚠️ Logo NOT found. Put logo at:")
        print(f"   - {LOGO_PATH_PRIMARY}")
        print(f"   - {LOGO_PATH_FALLBACK}")


# =========================
# HELPERS
# =========================
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def require_client():
    if not getattr(app.state, "client", None):
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY not set.\n"
                "Set it in Windows Environment Variables (User variables) and restart VS Code/terminal."
            ),
        )


def save_upload(file: UploadFile, report_id: str) -> str:
    if file.content_type != "application/pdf" and (file.filename and not file.filename.lower().endswith(".pdf")):
        raise HTTPException(400, "Please upload a PDF file.")

    path = os.path.join(UPLOAD_DIR, f"{report_id}.pdf")
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path


def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    return doc.page_count


def render_pdf_pages_to_png_bytes(pdf_path: str, max_pages: int, zoom: float) -> List[bytes]:
    doc = fitz.open(pdf_path)
    if doc.page_count < 1:
        raise ValueError("PDF has no pages.")
    pages_to_render = min(doc.page_count, max_pages)
    mat = fitz.Matrix(zoom, zoom)
    out: List[bytes] = []
    for i in range(pages_to_render):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(pix.tobytes("png"))
    return out


def png_bytes_to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def make_crops_from_page(png_bytes: bytes) -> List[bytes]:
    """
    For hand X-rays, the fracture is often subtle.
    We send the full image + several zoomed crops so Gemini can see cortex lines.
    """
    img = png_bytes_to_pil(png_bytes)
    w, h = img.size

    crops: List[Image.Image] = []

    # Full image
    crops.append(img)

    # Fingers area (top ~70%)
    crops.append(img.crop((0, 0, w, int(h * 0.72))))

    # Central focus (phalanges/metacarpals)
    crops.append(img.crop((int(w * 0.10), int(h * 0.03), int(w * 0.90), int(h * 0.78))))

    # Left finger region
    crops.append(img.crop((int(w * 0.03), int(h * 0.03), int(w * 0.58), int(h * 0.78))))

    # Right finger region
    crops.append(img.crop((int(w * 0.42), int(h * 0.03), int(w * 0.97), int(h * 0.78))))

    # Light upscale to help tiny details
    out_bytes: List[bytes] = []
    for c in crops:
        cw, ch = c.size
        scale = 1.35
        c2 = c.resize((int(cw * scale), int(ch * scale)))
        out_bytes.append(pil_to_png_bytes(c2))

    return out_bytes


def _img_part_from_png(png_bytes: bytes) -> Dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(png_bytes).decode("utf-8"),
        }
    }


# =========================
# GEMINI
# =========================
def gemini_pass_1_identify(all_images: List[bytes], page_count: int) -> str:
    """
    Pass 1: modality/anatomy/view/limits only (no diagnosis).
    """
    require_client()
    client = app.state.client
    model = app.state.model

    prompt = f"""
You are a radiology assistant.

You will be shown images from a PDF (full images + zoomed crops).
Your job is ONLY to identify:
- modality
- body part/anatomy
- likely view/plane
- limitations

PDF page count: {page_count}

Return exactly this format:

Modality guess: ...
Body part/anatomy: ...
Laterality: left/right/uncertain
View/plane: ...
Limitations: ...
"""

    parts = [_img_part_from_png(p) for p in all_images[: app.state.max_total_images]]
    parts.append({"text": prompt})

    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config={"temperature": 0.1, "max_output_tokens": 260},
    )
    if not getattr(resp, "text", None):
        raise RuntimeError("Gemini pass-1 returned empty response.")
    return resp.text.strip()


def gemini_pass_2_radiology_report(all_images: List[bytes], context: str, page_count: int) -> str:
    """
    Pass 2: Radiology-style report, possible diagnosis, not definitive.
    """
    require_client()
    client = app.state.client
    model = app.state.model

    single_view = "Yes" if page_count <= 1 else "No"

    prompt = f"""
You are an expert radiology assistant.

Task: Write a professional radiology-style SUMMARY based on the provided images.
Be direct and clinically useful, but do NOT give a definitive diagnosis.
Use "Possible / Suspected / Likely" wording.

CONTEXT:
{context}

Single view provided: {single_view}

OUTPUT FORMAT (exactly, with headings):

IMPRESSION:
<1–3 lines. Most likely diagnosis/possibilities with location. Use "Possible/Suspected".>

Confidence: <Low / Medium / High>

FINDINGS:
• <3–8 bullets describing visible signs: fracture line, cortical disruption, angulation, alignment, joint involvement, swelling>

WHAT THIS MEANS:
<2–5 short lines for a patient>

LIMITATIONS:
• <1–3 bullets: single view, image quality, no history, subtle fractures>

RECOMMENDED NEXT STEP:
• <2–5 bullets: additional views (AP/lateral/oblique), radiology read, ortho/ER if severe pain/deformity, immobilization guidance>

RULES:
- If you see fracture cues (lucent line, cortical break, step-off, angulation, displaced fragment), state:
  "Suspected/possible fracture at <finger + bone + segment>".
- NEVER say "no fracture" or "fracture ruled out".
  If not obvious, say: "No obvious displaced fracture; occult/non-displaced fracture cannot be excluded."
- Do NOT invent patient age/history/symptoms.
- If laterality is unclear, say "laterality uncertain".
- If you provide fewer than 10 lines, you failed. Always include all sections.
"""

    parts = [_img_part_from_png(p) for p in all_images[: app.state.max_total_images]]
    parts.append({"text": prompt})

    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config={"temperature": 0.2, "max_output_tokens": 1200},
    )
    if not getattr(resp, "text", None):
        raise RuntimeError("Gemini pass-2 returned empty response.")
    return resp.text.strip()


def safety_guard_text(summary: str) -> str:
    """
    Remove dangerous absolute claims but keep useful 'possible diagnosis' text.
    """
    s = summary
    replacements = [
        ("No fracture", "No obvious displaced fracture seen; an occult/non-displaced fracture cannot be excluded"),
        ("no fracture", "no obvious displaced fracture seen; an occult/non-displaced fracture cannot be excluded"),
        ("fracture ruled out", "a fracture cannot be fully excluded"),
        ("ruled out", "cannot be fully excluded"),
        ("definitely", "likely"),
        ("certainly", "likely"),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    return s


# =========================
# HOSPITAL STYLE PDF (LOGO + WATERMARK)
# =========================
def create_pdf(report_id: str, raw_text: str, filename: str) -> str:
    import re
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader

    path = os.path.join(OUT_DIR, f"{report_id}_summary.pdf")

    # Clean markdown
    text = raw_text.replace("**", "")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

    # Extract generated timestamp if present
    generated = ""
    m = re.search(r"Generated:\s*([0-9T:\.\-Z]+)", text)
    if m:
        generated = m.group(1)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtle_style = ParagraphStyle(
        name="Subtle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#4B5563"),
    )
    label_style = ParagraphStyle(
        name="Label",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#111827"),
    )
    body_style = ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        name="Bullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=6,
    )

    # Load logo
    logo_reader = None
    if os.path.exists(LOGO_PATH_PRIMARY):
        logo_reader = ImageReader(LOGO_PATH_PRIMARY)
    elif os.path.exists(LOGO_PATH_FALLBACK):
        logo_reader = ImageReader(LOGO_PATH_FALLBACK)

    def draw_header_footer(canvas, doc):
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors

        w, h = letter
        canvas.saveState()

        # WATERMARK
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.setFont("Helvetica-Bold", 60)
        canvas.setFillAlpha(0.10)
        canvas.translate(w / 2, h / 2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, "CONFIDENTIAL")
        canvas.restoreState()

        # TOP BAR
        canvas.setFillColor(colors.HexColor("#0B4F6C"))
        canvas.rect(0, h - 60, w, 60, fill=1, stroke=0)

        # LOGO
        x0 = 36
        y0 = h - 52
        logo_size = 34
        if logo_reader:
            canvas.drawImage(logo_reader, x0, y0, width=logo_size, height=logo_size, mask="auto")
        else:
            canvas.setFillColor(colors.white)
            canvas.roundRect(x0, y0, logo_size, logo_size, 6, fill=0, stroke=1)

        # BRAND
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x0 + logo_size + 10, h - 34, "MedDecode AI")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x0 + logo_size + 10, h - 50, "Summary Report")

        # PAGE
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(w - 36, 26, f"Page {doc.page}")

        # FOOTER LINE
        canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
        canvas.setLineWidth(1)
        canvas.line(36, 40, w - 36, 40)

        # DISCLAIMER
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.setFont("Helvetica", 8.5)
        canvas.drawCentredString(
            w / 2,
            16,
            "AI-generated summary for informational use only. Confirm with a clinician."
        )

        canvas.restoreState()

    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=88,
        bottomMargin=62,
    )

    story = []
    story.append(Spacer(1, 6))
    story.append(Paragraph("Summary", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB")))
    story.append(Spacer(1, 10))

    meta_data = [
        [Paragraph("Report ID:", label_style), Paragraph(report_id, subtle_style),
         Paragraph("Generated (UTC):", label_style), Paragraph(generated or "—", subtle_style)],
        [Paragraph("Filename:", label_style), Paragraph(filename or "—", subtle_style),
         Paragraph("Reviewer:", label_style), Paragraph("MedDecode AI (Automated)", subtle_style)],
    ]
    meta_table = Table(meta_data, colWidths=[80, 200, 105, 135])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E5E7EB")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Render body preserving structure
    for ln in raw_text.splitlines():
        ln = ln.rstrip()

        if not ln.strip():
            story.append(Spacer(1, 8))
            continue

        stripped = ln.lstrip()
        if stripped.startswith(("•", "-", "*")):
            clean = stripped[1:].strip()
            story.append(Paragraph(clean, bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(ln, body_style))

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    return path


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": getattr(app.state, "model", None),
        "has_key": bool(getattr(app.state, "client", None)),
    }


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    rid = str(uuid.uuid4())
    path = save_upload(file, rid)
    REPORTS[rid] = {"file": path, "status": "uploaded", "filename": file.filename or f"{rid}.pdf"}
    return {"report_id": rid, "filename": REPORTS[rid]["filename"], "status": "uploaded"}


@app.post("/process/{rid}")
def process(rid: str):
    if rid not in REPORTS:
        raise HTTPException(404, "Report not found")

    require_client()
    pdf_path = REPORTS[rid]["file"]
    filename = REPORTS[rid].get("filename", "")

    try:
        page_count = get_pdf_page_count(pdf_path)

        # Render pages
        pages_png = render_pdf_pages_to_png_bytes(
            pdf_path,
            max_pages=app.state.max_pages,
            zoom=app.state.render_zoom,
        )

        # Full + crops for each page
        all_images: List[bytes] = []
        for p in pages_png:
            all_images.extend(make_crops_from_page(p))

        all_images = all_images[: app.state.max_total_images]

        context = gemini_pass_1_identify(all_images, page_count)
        summary = gemini_pass_2_radiology_report(all_images, context, page_count)
        summary = safety_guard_text(summary)

    except HTTPException:
        raise
    except Exception as e:
        print("=== PROCESS ERROR ===")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {type(e).__name__}: {str(e)}",
        )

    final_text = f"""Report ID: {rid}
Filename: {filename}
Generated: {now_iso()}

{summary}
"""

    pdf_out = create_pdf(rid, final_text, filename)

    REPORTS[rid]["status"] = "processed"
    REPORTS[rid]["pdf"] = pdf_out
    REPORTS[rid]["images_sent"] = len(all_images)
    REPORTS[rid]["page_count"] = page_count
    REPORTS[rid]["model"] = app.state.model

    return {
        "report_id": rid,
        "status": "processed",
        "pdf_url": f"/pdf/{rid}",
        "images_sent": len(all_images),
        "page_count_in_pdf": page_count,
        "model": app.state.model,
    }


@app.get("/pdf/{rid}")
def download(rid: str):
    if rid not in REPORTS:
        raise HTTPException(404, "Not found")

    if REPORTS[rid].get("status") != "processed":
        raise HTTPException(400, "Report not processed yet. Call /process/{rid} first.")

    pdf_path = REPORTS[rid].get("pdf")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(404, "Generated PDF not found.")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"MedDecodeAI_{rid}.pdf",
    )
