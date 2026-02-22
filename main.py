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
app = FastAPI(title="MedDecode AI", version="16.0")

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
# TOKEN SETTINGS (ROBUST)
# =========================
PASS1_MAX_TOKENS = 260
PASS2_MAX_TOKENS = 5000
LAB_MAX_TOKENS = 5000

LAB_CONTINUE_MAX_TRIES = 4
RAD_CONTINUE_MAX_TRIES = 3


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

    # PowerShell: $env:GEMINI_MODEL="gemini-2.5-pro"
    app.state.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # Render knobs
    app.state.max_pages = int(os.getenv("MD_MAX_PAGES", "3"))
    app.state.render_zoom = float(os.getenv("MD_RENDER_ZOOM", "3.8"))
    app.state.max_total_images = int(os.getenv("MD_MAX_IMAGES", "10"))

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
            detail="Server not configured (missing AI key). Please try again later.",
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


def extract_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
    """
    Lab reports usually contain selectable text.
    If scanned image-only PDF, this may return empty (OCR would be needed).
    """
    doc = fitz.open(pdf_path)
    pages = min(doc.page_count, max_pages)
    chunks: List[str] = []
    for i in range(pages):
        t = doc.load_page(i).get_text("text") or ""
        t = t.strip()
        if t:
            chunks.append(t)
    return "\n\n".join(chunks).strip()


def detect_doc_type_from_text(extracted_text: str) -> str:
    """
    Returns: "lab" or "radiology"
    Strong LAB detection so CBC/biochem PDFs do not go through radiology prompts.
    """
    t = (extracted_text or "").lower()

    lab_keywords = [
        "complete blood count", "cbc", "hemoglobin", "haemoglobin",
        "wbc", "rbc", "platelet", "plt", "hematocrit", "mcv", "mch", "mchc", "rdw",
        "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",
        "reference range", "units", "result", "laboratory", "lab report", "pathology",
        "peripheral smear", "anisocytosis", "poikilocytosis",
        "serum", "ferritin", "iron", "tibc", "vitamin b12", "folate", "creatinine",
        "sgot", "sgpt", "alt", "ast", "bilirubin", "glucose", "hba1c",
        "reporting laboratory", "specimen", "method", "flag", "high", "low",
    ]

    rad_keywords = [
        "x-ray", "xray", "radiograph", "ct", "mri", "ultrasound",
        "ap view", "pa view", "lateral", "oblique", "fracture", "dislocation",
        "cortical", "joint space", "soft tissue swelling",
    ]

    lab_hits = sum(1 for k in lab_keywords if k in t)
    rad_hits = sum(1 for k in rad_keywords if k in t)

    # If there's meaningful extracted text and ANY lab signal with no rad signal => lab
    if len(t) > 120 and lab_hits >= 1 and rad_hits == 0:
        return "lab"

    if lab_hits >= 2 and lab_hits >= rad_hits:
        return "lab"

    return "radiology"


def clean_ai_error(e: Exception):
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        raise HTTPException(status_code=429, detail="AI limit reached. Please come back later.")
    raise HTTPException(status_code=500, detail="AI processing failed. Please try again later.")


def png_bytes_to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def make_crops_from_page(png_bytes: bytes) -> List[bytes]:
    """
    Full image + several crops. Good for subtle fractures.
    """
    img = png_bytes_to_pil(png_bytes)
    w, h = img.size

    crops: List[Image.Image] = []
    crops.append(img)
    crops.append(img.crop((0, 0, w, int(h * 0.72))))
    crops.append(img.crop((int(w * 0.10), int(h * 0.03), int(w * 0.90), int(h * 0.78))))
    crops.append(img.crop((int(w * 0.03), int(h * 0.03), int(w * 0.58), int(h * 0.78))))
    crops.append(img.crop((int(w * 0.42), int(h * 0.03), int(w * 0.97), int(h * 0.78))))

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


def _has_all_radiology_sections(text: str) -> bool:
    t = (text or "").upper()
    required = [
        "IMPRESSION:",
        "CONFIDENCE:",
        "FINDINGS:",
        "WHAT THIS MEANS:",
        "LIMITATIONS:",
        "RECOMMENDED NEXT STEP:",
    ]
    return all(r in t for r in required)


# =========================
# GEMINI
# =========================
def gemini_pass_1_identify(all_images: List[bytes], page_count: int) -> str:
    """
    Pass 1: modality/anatomy/view/limits only (PROMPT UNCHANGED)
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

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config={"temperature": 0.1, "max_output_tokens": PASS1_MAX_TOKENS},
        )
    except Exception as e:
        clean_ai_error(e)

    if not getattr(resp, "text", None):
        raise RuntimeError("Gemini pass-1 returned empty response.")
    return resp.text.strip()


def gemini_pass_2_radiology_report(all_images: List[bytes], context: str, page_count: int) -> str:
    """
    Pass 2: Radiology-style report. PROMPT TEXT UNCHANGED (only adds END marker line).
    Auto-continues if truncated.
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

End your response with a new line containing exactly: <<<END>>>
""".strip()

    parts = [_img_part_from_png(p) for p in all_images[: app.state.max_total_images]]
    parts.append({"text": prompt})

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config={"temperature": 0.2, "max_output_tokens": PASS2_MAX_TOKENS},
        )
    except Exception as e:
        clean_ai_error(e)

    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini pass-2 returned empty response.")

    tries = 0
    while ("<<<END>>>" not in text or not _has_all_radiology_sections(text)) and tries < RAD_CONTINUE_MAX_TRIES:
        tries += 1
        cont_prompt = """
Continue EXACTLY from where you left off.
Do NOT repeat anything already written.
Finish any cut-off word/sentence and ensure all required headings exist:
IMPRESSION, Confidence, FINDINGS, WHAT THIS MEANS, LIMITATIONS, RECOMMENDED NEXT STEP.
End with a new line containing exactly: <<<END>>>
""".strip()

        try:
            resp2 = client.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": parts},
                    {"role": "model", "parts": [{"text": text}]},
                    {"role": "user", "parts": [{"text": cont_prompt}]},
                ],
                config={"temperature": 0.2, "max_output_tokens": PASS2_MAX_TOKENS},
            )
        except Exception as e:
            clean_ai_error(e)

        text2 = (getattr(resp2, "text", "") or "").strip()
        if not text2:
            break
        text = (text + "\n" + text2).strip()

    text = text.replace("<<<END>>>", "").strip()
    if not text:
        raise RuntimeError("Gemini pass-2 returned empty response.")
    return text


def _has_all_lab_sections(text: str) -> bool:
    t = (text or "").upper()
    required = [
        "IMPRESSION:",
        "KEY ABNORMALITIES:",
        "WHAT THIS MAY SUGGEST:",
        "WHAT TO CONFIRM:",
        "NEXT STEPS:",
        "LIMITATIONS:",
    ]
    return all(r in t for r in required)


def gemini_lab_summary(extracted_text: str) -> str:
    """
    LAB summary: text-based, forces completeness.
    Does NOT touch radiology prompts.
    """
    require_client()
    client = app.state.client
    model = app.state.model

    base_prompt = f"""
You are a clinical lab report summarizer.

Input: Text from a lab report (CBC/biochem panels).
Goal: Produce a patient-friendly but clinically useful summary.
Do NOT refuse. Do NOT say you cannot generate.
Do NOT invent values that are not present.
If values/ranges are missing, say "not shown".

OUTPUT FORMAT (exact headings):

IMPRESSION:
<1–3 lines: overall pattern (use "possible/suggestive", not definitive).>

KEY ABNORMALITIES:
• <3–12 bullets: each bullet should include test name + value + unit + flag if present, otherwise say "not shown".>

WHAT THIS MAY SUGGEST:
<2–6 short lines (not definitive).>

WHAT TO CONFIRM:
• <2–6 bullets (e.g., ferritin/iron studies, repeat CBC, peripheral smear, etc.)>

NEXT STEPS:
• <2–6 bullets (follow-up and urgency guidance).>

LIMITATIONS:
• <1–4 bullets (single report, missing history, reference ranges vary).>

IMPORTANT:
- Do NOT stop mid-sentence or mid-bullet.
- Always include ALL headings above.
- End your response with a new line containing exactly: <<<END>>>

LAB REPORT TEXT:
{extracted_text[:14000]}
""".strip()

    # 1) First call
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": base_prompt}]}],
            # 5000 is fine; 3000 also works. Keep 5000 for “never cut”.
            config={"temperature": 0.2, "max_output_tokens": 5000},
        )
    except Exception as e:
        clean_ai_error(e)

    text = (getattr(resp, "text", "") or "").strip()

    # 2) Continue until END marker + headings exist
    tries = 0
    while tries < 6 and (("<<<END>>>" not in text) or (not _has_all_lab_sections(text))):
        tries += 1
        cont_prompt = """
Continue EXACTLY from where you left off.
Do NOT repeat any previous lines.
Finish any cut-off word/sentence.
If any heading is missing, add it and fill it.
End with a new line containing exactly: <<<END>>>
""".strip()

        try:
            resp2 = client.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": [{"text": base_prompt}]},
                    {"role": "model", "parts": [{"text": text}]},
                    {"role": "user", "parts": [{"text": cont_prompt}]},
                ],
                config={"temperature": 0.2, "max_output_tokens": 2000},
            )
        except Exception as e:
            clean_ai_error(e)

        text2 = (getattr(resp2, "text", "") or "").strip()
        if not text2:
            break
        text = (text + "\n" + text2).strip()

    # 3) Last-resort “finish missing parts only” if still incomplete
    if ("<<<END>>>" not in text) or (not _has_all_lab_sections(text)):
        final_prompt = """
You MUST complete the response.
Do NOT repeat anything.
Only add what is missing (finish the cut sentence and any missing headings/lines).
End with a new line containing exactly: <<<END>>>
""".strip()

        try:
            resp3 = client.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": [{"text": base_prompt}]},
                    {"role": "model", "parts": [{"text": text}]},
                    {"role": "user", "parts": [{"text": final_prompt}]},
                ],
                config={"temperature": 0.1, "max_output_tokens": 1500},
            )
            text3 = (getattr(resp3, "text", "") or "").strip()
            if text3:
                text = (text + "\n" + text3).strip()
        except Exception as e:
            clean_ai_error(e)

    # strip marker
    text = text.replace("<<<END>>>", "").strip()

    if not text:
        raise RuntimeError("Gemini lab summary returned empty response.")
    return text


def safety_guard_text(summary: str) -> str:
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
    text = raw_text.replace("**", "").replace("\t", " ")
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

    # Render body preserving structure (handles "FINDINGS: • item" too)
    for ln in raw_text.splitlines():
        ln = ln.rstrip()

        if not ln.strip():
            story.append(Spacer(1, 8))
            continue

        # Inline bullet on same line as heading: "FINDINGS: • ..."
        if "•" in ln and not ln.lstrip().startswith("•"):
            before, after = ln.split("•", 1)
            before = before.strip()
            after = after.strip()
            if before:
                story.append(Paragraph(before, body_style))
            if after:
                story.append(Paragraph(after, bullet_style, bulletText="•"))
            continue

        stripped = ln.lstrip()
        if stripped.startswith(("•", "-", "*")):
            clean = stripped[1:].strip()
            if clean:
                story.append(Paragraph(clean, bullet_style, bulletText="•"))
            else:
                story.append(Spacer(1, 4))
            continue

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

        # Detect LAB vs RADIOLOGY from PDF TEXT first
        extracted_text = extract_pdf_text(pdf_path, max_pages=6)
        doc_type = detect_doc_type_from_text(extracted_text)

        if doc_type == "lab":
            if not extracted_text:
                raise HTTPException(
                    status_code=400,
                    detail="This looks like a lab report, but text could not be extracted (likely scanned). Upload a text-based PDF (selectable text).",
                )
            summary = gemini_lab_summary(extracted_text)
            images_sent = 0
        else:
            pages_png = render_pdf_pages_to_png_bytes(
                pdf_path,
                max_pages=app.state.max_pages,
                zoom=app.state.render_zoom,
            )

            all_images: List[bytes] = []
            for p in pages_png:
                all_images.extend(make_crops_from_page(p))

            all_images = all_images[: app.state.max_total_images]
            images_sent = len(all_images)

            context = gemini_pass_1_identify(all_images, page_count)
            summary = gemini_pass_2_radiology_report(all_images, context, page_count)
            summary = safety_guard_text(summary)

    except HTTPException:
        raise
    except Exception as e:
        print("=== PROCESS ERROR ===")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise HTTPException(status_code=429, detail="AI limit reached. Please come back later.")
        raise HTTPException(status_code=500, detail="Processing failed. Please try again later.")

    final_text = f"""Report ID: {rid}
Filename: {filename}
Generated: {now_iso()}

{summary}
"""

    pdf_out = create_pdf(rid, final_text, filename)

    REPORTS[rid]["status"] = "processed"
    REPORTS[rid]["pdf"] = pdf_out
    REPORTS[rid]["page_count"] = page_count
    REPORTS[rid]["model"] = app.state.model
    REPORTS[rid]["doc_type"] = doc_type
    REPORTS[rid]["images_sent"] = images_sent
    REPORTS[rid]["summary_len"] = len(summary)

    return {
        "report_id": rid,
        "status": "processed",
        "pdf_url": f"/pdf/{rid}",
        "page_count_in_pdf": page_count,
        "model": app.state.model,
        "doc_type": doc_type,
        "images_sent": images_sent,
        "summary_len": REPORTS[rid]["summary_len"],
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
