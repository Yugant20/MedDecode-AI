import os
import re
import io
import uuid
import json
import base64
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from google import genai
import fitz  # PyMuPDF
from PIL import Image

# =========================
# APP SETUP
# =========================
app = FastAPI(title="MedDecode AI", version="26.0-LAB-ROBUST-JSON-FALLBACK")

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
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

LOGO_PATH_PRIMARY = os.path.join(ASSETS_DIR, "logo.png")
LOGO_PATH_FALLBACK = os.path.join(BASE_DIR, "logo.png")

# =========================
# TOKEN SETTINGS
# =========================
PASS1_MAX_TOKENS = 260
PASS2_MAX_TOKENS = 5000

LAB_JSON_MAX_TOKENS = 1400
LAB_SUMMARY_MAX_TOKENS = 1800
LAB_TEXT_SLICE = 9000

LAB_CONTINUE_MAX_TRIES = 8
RAD_CONTINUE_MAX_TRIES = 3

DEBUG_MODE = os.getenv("MD_DEBUG", "0").strip().lower() in ("1", "true", "yes")


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

    app.state.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # Radiology render knobs
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

    print(f"🧪 MD_DEBUG={int(DEBUG_MODE)} | model={app.state.model} | version={app.version}")


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


def dbg_kv(label: str, value: Any):
    if DEBUG_MODE:
        print(f"=== DEBUG {label}: {value} ===")


def dbg_text(label: str, text: str, n: int = 900):
    if not DEBUG_MODE:
        return
    t = (text or "").replace("\r", "")
    print(f"\n=== DEBUG {label} (first {n} chars) ===")
    print(t[:n])
    print("=== END DEBUG ===\n")


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


def extract_pdf_text(pdf_path: str, max_pages: int = 8) -> str:
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
    t = (extracted_text or "").lower().strip()
    if len(t) < 40:
        return "radiology"

    lab_keywords = [
        "cbc", "complete blood count", "haemoglobin", "hemoglobin", "wbc", "rbc",
        "platelet", "plt", "mcv", "mch", "mchc", "rdw", "reference range", "units",
        "hematocrit", "haematocrit", "neutrophil", "lymphocyte", "monocyte",
        "eosinophil", "basophil", "ferritin", "iron", "tibc", "glucose", "hba1c",
        "creatinine", "alt", "ast", "bilirubin", "results", "result", "flag", "high", "low",
    ]
    rad_keywords = [
        "ct", "x-ray", "xray", "mri", "ultrasound", "fracture", "nodule", "spiculated",
        "lymphadenopathy", "mediastinal", "hilar", "pet/ct", "ebus", "biopsy",
        "carcinoma", "metast", "staging",
    ]

    lab_hits = sum(1 for k in lab_keywords if k in t)
    rad_hits = sum(1 for k in rad_keywords if k in t)

    if lab_hits >= 1 and lab_hits >= rad_hits:
        return "lab"
    return "radiology"


def clean_ai_error(e: Exception):
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        raise HTTPException(status_code=429, detail="AI limit reached. Please come back later.")
    raise HTTPException(status_code=500, detail=f"AI processing failed: {msg}")


def png_bytes_to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def make_crops_from_page(png_bytes: bytes) -> List[bytes]:
    img = png_bytes_to_pil(png_bytes)
    w, h = img.size
    crops = [
        img,
        img.crop((0, 0, w, int(h * 0.72))),
        img.crop((int(w * 0.10), int(h * 0.03), int(w * 0.90), int(h * 0.78))),
        img.crop((int(w * 0.03), int(h * 0.03), int(w * 0.58), int(h * 0.78))),
        img.crop((int(w * 0.42), int(h * 0.03), int(w * 0.97), int(h * 0.78))),
    ]
    out: List[bytes] = []
    for c in crops:
        cw, ch = c.size
        c2 = c.resize((int(cw * 1.35), int(ch * 1.35)))
        out.append(pil_to_png_bytes(c2))
    return out


def _img_part_from_png(png_bytes: bytes) -> Dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(png_bytes).decode("utf-8"),
        }
    }


# =========================
# GENAI RESPONSE TEXT (ROBUST)
# =========================
def genai_text(resp) -> str:
    # 1) Prefer resp.text
    try:
        t = getattr(resp, "text", None)
        if t and str(t).strip():
            return str(t).strip()
    except Exception:
        pass

    # 2) dict-like
    try:
        if isinstance(resp, dict):
            t = resp.get("text")
            if t and str(t).strip():
                return str(t).strip()
    except Exception:
        pass

    # 3) candidates -> content -> parts
    out: List[str] = []
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates is None and isinstance(resp, dict):
            candidates = resp.get("candidates", [])
        candidates = candidates or []

        for c in candidates:
            content = getattr(c, "content", None)
            if content is None and isinstance(c, dict):
                content = c.get("content")

            parts = getattr(content, "parts", None) if content is not None else None
            if parts is None and isinstance(content, dict):
                parts = content.get("parts")
            parts = parts or []

            for p in parts:
                if isinstance(p, dict):
                    pt = p.get("text")
                else:
                    pt = getattr(p, "text", None)
                if pt and str(pt).strip():
                    out.append(str(pt).strip())
    except Exception:
        pass

    return "\n".join(out).strip()


# =========================
# LAB JSON PARSE HELPERS
# =========================
def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.replace("```json", "").replace("```JSON", "").replace("```", "")
    return s.strip()


def extract_first_json_object(s: str) -> Optional[dict]:
    s = strip_code_fences(s)
    if not s:
        return None

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start: i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
    return None


def _has_all_lab_sections(text: str) -> bool:
    t = (text or "")
    patterns = [
        r"IMPRESSION\s*:",
        r"KEY\s+ABNORMALITIES\s*:",
        r"WHAT\s+THIS\s+MAY\s+SUGGEST\s*:",
        r"WHAT\s+TO\s+CONFIRM\s*:",
        r"NEXT\s+STEPS\s*:",
        r"LIMITATIONS\s*:",
    ]
    return all(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def finalize_lab_summary(text: str) -> str:
    t = (text or "").strip()
    # If model accidentally used FINDINGS heading, normalize it
    t = re.sub(r"\bFINDINGS\s*:", "KEY ABNORMALITIES:", t, flags=re.IGNORECASE)

    required = [
        "IMPRESSION:",
        "KEY ABNORMALITIES:",
        "WHAT THIS MAY SUGGEST:",
        "WHAT TO CONFIRM:",
        "NEXT STEPS:",
        "LIMITATIONS:",
    ]

    def has_heading(h: str) -> bool:
        hh = h.replace(":", "").strip()
        pattern = r"\b" + r"\s+".join(map(re.escape, hh.split())) + r"\s*:"
        return bool(re.search(pattern, t, flags=re.IGNORECASE))

    if not t:
        return (
            "IMPRESSION:\nUnable to generate a complete summary from the provided report text.\n\n"
            "KEY ABNORMALITIES:\n• Not shown.\n\n"
            "WHAT THIS MAY SUGGEST:\nNot shown.\n\n"
            "WHAT TO CONFIRM:\n• Not shown.\n\n"
            "NEXT STEPS:\n• Not shown.\n\n"
            "LIMITATIONS:\n• Model returned empty output.\n• Confirm using the original report and a clinician."
        )

    for h in required:
        if not has_heading(h):
            if h == "IMPRESSION:":
                t += "\n\nIMPRESSION:\nNot shown."
            elif h == "KEY ABNORMALITIES:":
                t += "\n\nKEY ABNORMALITIES:\n• Not shown."
            elif h == "WHAT THIS MAY SUGGEST:":
                t += "\n\nWHAT THIS MAY SUGGEST:\nNot shown."
            elif h == "WHAT TO CONFIRM:":
                t += "\n\nWHAT TO CONFIRM:\n• Not shown."
            elif h == "NEXT STEPS:":
                t += "\n\nNEXT STEPS:\n• Not shown."
            elif h == "LIMITATIONS:":
                t += (
                    "\n\nLIMITATIONS:\n"
                    "• Output may be incomplete.\n"
                    "• Confirm using the original report and a clinician.\n"
                    "• Reference ranges vary by lab."
                )

    return t.strip()


# =========================
# LAB: JSON EXTRACT (ROBUST + RETRY) + FALLBACK
# =========================
def lab_extract_json_robust(extracted_text: str) -> Optional[dict]:
    require_client()
    client = app.state.client
    model = app.state.model

    lab_text = (extracted_text or "").strip()
    if not lab_text:
        return None

    lab_text = lab_text[:LAB_TEXT_SLICE]

    prompt = f"""
Return ONLY a single JSON object. No markdown. No commentary.

Schema:
{{
  "impression": "string",
  "abnormalities": [
    {{"test":"string","value":"string","unit":"string","flag":"High|Low|Abnormal|Normal|Not shown","range":"string"}}
  ],
  "what_this_may_suggest": "string",
  "what_to_confirm": ["string"],
  "next_steps": ["string"],
  "limitations": ["string"]
}}

Rules:
- Do NOT invent values.
- If missing or unreadable, use "Not shown".
- Max abnormalities: 12
- End immediately after the final }}.

LAB REPORT:
{lab_text}
""".strip()

    last_raw = ""
    for attempt in range(1, 4):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"temperature": 0.0, "max_output_tokens": LAB_JSON_MAX_TOKENS},
            )
        except Exception as e:
            clean_ai_error(e)

        last_raw = genai_text(resp)
        dbg_text(f"LAB_JSON_RAW_attempt_{attempt}", last_raw, n=700)

        data = extract_first_json_object(last_raw or "")
        if isinstance(data, dict):
            return data

        prompt = "ONLY JSON. " + prompt

    dbg_text("LAB_JSON_FAILED_RAW", last_raw, n=1200)
    return None


def lab_summary_from_json(data: dict) -> str:
    require_client()
    client = app.state.client
    model = app.state.model

    prompt = f"""
Write a complete lab summary using EXACT headings.

IMPRESSION:
<1-3 lines>

KEY ABNORMALITIES:
• <3-12 bullets>

WHAT THIS MAY SUGGEST:
<2-6 short lines>

WHAT TO CONFIRM:
• <2-6 bullets>

NEXT STEPS:
• <2-6 bullets>

LIMITATIONS:
• <1-4 bullets>

Rules:
- Do NOT invent values.
- If "Not shown", keep it.
- End with <<<END>>> on its own line.

JSON:
{json.dumps(data, ensure_ascii=False)}
""".strip()

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.2, "max_output_tokens": LAB_SUMMARY_MAX_TOKENS, "stop_sequences": ["<<<END>>>"]},
        )
    except Exception as e:
        clean_ai_error(e)

    text = genai_text(resp)
    dbg_text("LAB_SUMMARY_FROM_JSON_RAW", text, n=900)

    if not (text or "").strip():
        raise RuntimeError("Lab summary-from-JSON returned empty text")

    text = (text or "").replace("<<<END>>>", "").strip()
    return finalize_lab_summary(text)


def lab_summary_plain_fallback(extracted_text: str) -> str:
    require_client()
    client = app.state.client
    model = app.state.model

    lab_text = (extracted_text or "").strip()[:LAB_TEXT_SLICE]

    prompt = f"""
Summarize this lab report with EXACT headings.

IMPRESSION:
KEY ABNORMALITIES:
WHAT THIS MAY SUGGEST:
WHAT TO CONFIRM:
NEXT STEPS:
LIMITATIONS:

Rules:
- Do NOT invent values.
- If missing, write "Not shown".
- End with <<<END>>> on its own line.

LAB REPORT:
{lab_text}
""".strip()

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.2, "max_output_tokens": LAB_SUMMARY_MAX_TOKENS, "stop_sequences": ["<<<END>>>"]},
        )
    except Exception as e:
        clean_ai_error(e)

    text = genai_text(resp)
    dbg_text("LAB_SUMMARY_PLAIN_RAW", text, n=900)

    if not (text or "").strip():
        raise RuntimeError("Lab plain fallback returned empty text")

    text = (text or "").replace("<<<END>>>", "").strip()
    return finalize_lab_summary(text)


def lab_summary_text_pipeline(extracted_text: str) -> str:
    data = lab_extract_json_robust(extracted_text)
    if data:
        return lab_summary_from_json(data)
    return lab_summary_plain_fallback(extracted_text)


# =========================
# GEMINI - LAB (IMAGE FALLBACK)
# =========================
def gemini_lab_summary_from_images(all_images: List[bytes]) -> str:
    require_client()
    client = app.state.client
    model = app.state.model

    prompt = """
You are a clinical LAB report summarizer (from images).

Use ONLY these headings in order:
IMPRESSION:
KEY ABNORMALITIES:
WHAT THIS MAY SUGGEST:
WHAT TO CONFIRM:
NEXT STEPS:
LIMITATIONS:

Rules:
- Do NOT invent values.
- If unreadable, write "Not shown".
- End with <<<END>>> on its own line.
""".strip()

    parts = [_img_part_from_png(p) for p in all_images[: app.state.max_total_images]]
    parts.append({"text": prompt})

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config={"temperature": 0.1, "max_output_tokens": LAB_SUMMARY_MAX_TOKENS, "stop_sequences": ["<<<END>>>"]},
        )
    except Exception as e:
        clean_ai_error(e)

    text = genai_text(resp)
    dbg_text("LAB_SUMMARY_IMAGE_RAW", text, n=900)

    if not (text or "").strip():
        raise RuntimeError("Gemini returned empty text for LAB summary (image).")

    text = (text or "").replace("<<<END>>>", "").strip()
    return finalize_lab_summary(text)


# =========================
# GEMINI - RADIOLOGY
# =========================
def gemini_pass_1_identify(all_images: List[bytes], page_count: int) -> str:
    require_client()
    client = app.state.client
    model = app.state.model

    prompt = f"""
You are a radiology assistant.

You will be shown images from a PDF (full images + zoomed crops).
Identify:
- modality
- anatomy
- view/plane
- limitations

PDF page count: {page_count}

Format:
Modality guess: ...
Body part/anatomy: ...
Laterality: left/right/uncertain
View/plane: ...
Limitations: ...
""".strip()

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

    text = genai_text(resp).strip()
    if not text:
        raise RuntimeError("Gemini pass-1 returned empty response.")
    return text


def gemini_pass_2_radiology_report(all_images: List[bytes], context: str, page_count: int) -> str:
    require_client()
    client = app.state.client
    model = app.state.model

    single_view = "Yes" if page_count <= 1 else "No"

    prompt = f"""
You are an expert radiology assistant.

Task: Write a professional radiology-style SUMMARY.
Be direct but do NOT give a definitive diagnosis.

CONTEXT:
{context}

Single view provided: {single_view}

OUTPUT FORMAT:
IMPRESSION:
Confidence:
FINDINGS:
WHAT THIS MEANS:
LIMITATIONS:
RECOMMENDED NEXT STEP:

End with <<<END>>>.
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

    text = genai_text(resp).strip()
    if not text:
        raise RuntimeError("Gemini pass-2 returned empty response.")

    tries = 0
    while ("<<<END>>>" not in text) and tries < RAD_CONTINUE_MAX_TRIES:
        tries += 1
        cont_prompt = "Continue from where you left off. End with <<<END>>>."
        try:
            resp2 = client.models.generate_content(
                model=model,
                contents=[
                    {"role": "model", "parts": [{"text": text}]},
                    {"role": "user", "parts": [{"text": cont_prompt}]},
                ],
                config={"temperature": 0.2, "max_output_tokens": PASS2_MAX_TOKENS},
            )
        except Exception as e:
            clean_ai_error(e)

        text2 = genai_text(resp2).strip()
        if not text2:
            break
        text = (text + "\n" + text2).strip()

    return text.replace("<<<END>>>", "").strip()


def safety_guard_text(summary: str) -> str:
    s = summary or ""
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
# HOSPITAL STYLE PDF
# =========================
def create_pdf(report_id: str, raw_text: str, filename: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader

    path = os.path.join(OUT_DIR, f"{report_id}_summary.pdf")

    text = raw_text.replace("**", "").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

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

        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.setFont("Helvetica-Bold", 60)
        canvas.setFillAlpha(0.10)
        canvas.translate(w / 2, h / 2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, "CONFIDENTIAL")
        canvas.restoreState()

        canvas.setFillColor(colors.HexColor("#0B4F6C"))
        canvas.rect(0, h - 60, w, 60, fill=1, stroke=0)

        x0 = 36
        y0 = h - 52
        logo_size = 34
        if logo_reader:
            canvas.drawImage(logo_reader, x0, y0, width=logo_size, height=logo_size, mask="auto")
        else:
            canvas.setFillColor(colors.white)
            canvas.roundRect(x0, y0, logo_size, logo_size, 6, fill=0, stroke=1)

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x0 + logo_size + 10, h - 34, "MedDecode AI")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x0 + logo_size + 10, h - 50, "Summary Report")

        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(w - 36, 26, f"Page {doc.page}")

        canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
        canvas.setLineWidth(1)
        canvas.line(36, 40, w - 36, 40)

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

    for ln in raw_text.splitlines():
        ln = ln.rstrip()

        if not ln.strip():
            story.append(Spacer(1, 8))
            continue

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
        "app_version": app.version,
        "debug": int(DEBUG_MODE),
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
        extracted_text = extract_pdf_text(pdf_path, max_pages=8)

        dbg_kv("EXTRACTED_TEXT_LEN", len(extracted_text or ""))
        dbg_text("EXTRACTED_TEXT_HEAD", (extracted_text or "")[:1200], n=1200)

        doc_type = detect_doc_type_from_text(extracted_text)

        tlow = (extracted_text or "").lower()
        force_lab_terms = ["cbc", "hemoglobin", "haemoglobin", "platelet", "wbc", "rbc", "mcv", "mch", "rdw", "reference range"]
        if any(k in tlow for k in force_lab_terms):
            doc_type = "lab"

        images_sent = 0

        # LAB FLOW
        if doc_type == "lab":
            if extracted_text and len(extracted_text.strip()) >= 120:
                summary = lab_summary_text_pipeline(extracted_text)
                summary = finalize_lab_summary(summary)
                images_sent = 0
            else:
                pages_png = render_pdf_pages_to_png_bytes(pdf_path, max_pages=3, zoom=3.2)
                all_images = pages_png[: app.state.max_total_images]
                images_sent = len(all_images)
                summary = gemini_lab_summary_from_images(all_images)
                summary = finalize_lab_summary(summary)

        # RADIOLOGY FLOW
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

            # You can keep your full radiology prompts here if you want;
            # leaving minimal versions in this file.
            context = gemini_pass_1_identify(all_images, page_count)
            summary = gemini_pass_2_radiology_report(all_images, context, page_count)
            summary = safety_guard_text(summary)

    except HTTPException:
        raise
    except Exception as e:
        print("=== PROCESS ERROR ===")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        if DEBUG_MODE:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
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
        "app_version": app.version,
        "debug": int(DEBUG_MODE),
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
