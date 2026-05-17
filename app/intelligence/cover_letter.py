"""
intelligence/cover_letter.py
-----------------------------
Cover Letter Generation — AI-crafted, JD-targeted cover letters.

generate_cover_letter()     → JSON response (for display in UI)
generate_cover_letter_pdf() → bytes (PDF file for download)

PDF format matches professional sample:
  - Name + contact header with divider line
  - Date, recipient, salutation
  - Body paragraphs with justified text
  - Bullet points for key selling points (bold label + body)
  - Professional sign-off

Same OpenRouter pattern: gpt-oss-120b → llama-3.3-70b → openrouter/free
"""

import os
import io
import json
import re
import requests
from typing import Optional
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]

# ─── Cover Letter System Prompt ──────────────────────────────────────────────

COVER_LETTER_SYSTEM_PROMPT = """You are an expert career coach and professional writer with 15+ years crafting compelling cover letters that land interviews at top tech and product companies in India and globally.

Your task: Write a powerful, personalized cover letter for the candidate applying to the specific job. Return ONLY a valid JSON object. No preamble, no explanation, no markdown fences.

Use this EXACT schema:
{
  "subject_line": "Application for Senior Data Engineer — [Your Name]",
  "opening_paragraph": "Hook paragraph — 2-3 sentences. Specific role + company name + ONE compelling reason excited about THIS company. Reference something real from the JD — their mission, tech stack, stage of growth.",
  "body_paragraph": "Match paragraph — 3-4 sentences. Top 2-3 directly relevant experiences using the same language as the JD. Include at least one metric.",
  "bullets": [
    {
      "label": "Product-Led Mindset",
      "text": "I do not just build reports; I build portals. At IVY, I deployed an AI-powered artist analytics dashboard that aggregated data for 483 artists, allowing teams to compare performance in real-time."
    },
    {
      "label": "Engineering Rigor",
      "text": "My background as a Software Engineer at LTIMindtree means my SQL skills and pipeline optimizations are grounded in clean, scalable code."
    }
  ],
  "value_paragraph": "Value-add paragraph — 2-3 sentences. Something unique the candidate brings that was not explicitly asked. A unique perspective, domain knowledge, side project, or complementary skill. Make the reader curious.",
  "closing_paragraph": "Close — 2 sentences. Enthusiastic, specific CTA. Express readiness for a call.",
  "sign_off": "Best regards,",
  "key_selling_points": [
    "4 years of Python + SQL experience directly matching their data pipeline requirements",
    "Built real-time ETL system processing 2M events/day"
  ],
  "tone_used": "professional",
  "word_count": 285,
  "personalization_score": 82
}

STRICT rules for content:
- Use exact company name from JD in opening_paragraph
- Use exact role title from JD in subject_line
- bullets: 2-4 bullets. Each has a short bold label (2-4 words) and 1-3 sentence text
- NEVER use: I am writing to apply, To Whom It May Concern, I am a hard worker, passion for technology, team player, go-getter, synergy, leverage
- DO use: Specific company name, specific role title, specific JD skills/tools by name
- word_count: Approximate total word count of the full letter
- personalization_score: 0-100 how tailored to the specific JD (100 = every line is JD-specific)
- Return ONLY the JSON. No other text, no markdown."""


# ─── API helpers ─────────────────────────────────────────────────────────────

def _get_api_key(index: int = 0) -> str:
    key1 = os.getenv("OPENROUTER_API_KEY", "")
    key2 = os.getenv("OPENROUTER_API_KEY_2", "")
    if index % 2 == 1 and key2:
        return key2
    return key1


def _build_cl_user_message(resume_text: str, jd_text: str, tone: str,
                            highlights: str, candidate_name: str) -> str:
    hl = f"\n\nCandidate wants to highlight: {highlights}" if highlights else ""
    nm = f"\nCandidate name: {candidate_name}" if candidate_name else ""
    return f"""Write a {tone} cover letter for this candidate applying to the job below.{hl}{nm}

--- RESUME START ---
{resume_text.strip()[:3500]}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{jd_text.strip()[:2500]}
--- JOB DESCRIPTION END ---

Return the JSON cover letter."""


def _call_openrouter(model: str, user_message: str, api_key: str) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HireBot Cover Letter",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": COVER_LETTER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.5,
        "max_tokens":  1800,
    }
    try:
        response = requests.post(
            OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()
        data     = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$",           "", raw_text)
        return json.loads(raw_text)
    except requests.exceptions.Timeout:
        print(f"[CoverLetter] Timeout: {model}"); return None
    except requests.exceptions.HTTPError as e:
        print(f"[CoverLetter] HTTP {e.response.status_code}: {model}"); return None
    except json.JSONDecodeError as e:
        print(f"[CoverLetter] JSON parse error ({model}): {e}"); return None
    except Exception as e:
        print(f"[CoverLetter] Error ({model}): {e}"); return None


def _validate(result: dict) -> bool:
    required = {"subject_line", "opening_paragraph", "body_paragraph",
                "closing_paragraph", "sign_off"}
    return (
        required.issubset(result.keys())
        and len(result.get("opening_paragraph", "")) > 50
    )


# ─── Main generation ─────────────────────────────────────────────────────────

def generate_cover_letter(
    resume_text:    str,
    jd_text:        str,
    tone:           str  = "professional",
    highlights:     str  = "",
    candidate_name: str  = "",
    key_index:      int  = 0,
) -> dict:
    """
    Generate a targeted cover letter for a specific JD.
    Returns JSON dict with all letter parts + metadata.
    """
    if not resume_text or len(resume_text.strip()) < 100:
        raise ValueError("Resume text is too short.")
    if not jd_text or len(jd_text.strip()) < 50:
        raise ValueError("Job description is too short.")

    valid_tones = {"professional", "enthusiastic", "concise"}
    tone = tone.lower() if tone.lower() in valid_tones else "professional"

    api_key = _get_api_key(key_index)
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    user_message = _build_cl_user_message(
        resume_text, jd_text, tone, highlights, candidate_name
    )

    for model in FREE_MODELS:
        print(f"[CoverLetter] Trying model: {model}")
        result = _call_openrouter(model, user_message, api_key)
        if result and _validate(result):
            print(f"[CoverLetter] Success: {model}")
            result["_model_used"] = model
            # Build a flat cover_letter_text for copy-all
            parts = [
                result.get("opening_paragraph", ""),
                result.get("body_paragraph", ""),
            ]
            for b in result.get("bullets", []):
                parts.append(f"• {b.get('label','')}: {b.get('text','')}")
            parts.append(result.get("value_paragraph", ""))
            parts.append(result.get("closing_paragraph", ""))
            result["cover_letter_text"] = "\n\n".join(p for p in parts if p)
            if "word_count" not in result:
                result["word_count"] = len(result["cover_letter_text"].split())
            return result
        print(f"[CoverLetter] Failed: {model}, trying next...")

    raise RuntimeError(
        "All OpenRouter free models failed for cover letter generation. "
        "Rate limit (200 req/day) likely hit. Try again later."
    )


# ─── PDF Generation ──────────────────────────────────────────────────────────

# PDF colour palette (matches Zahid's sample: clean black/grey professional)
_DARK   = colors.HexColor("#111827")
_MID    = colors.HexColor("#374151")
_LIGHT  = colors.HexColor("#6B7280")
_LINE   = colors.HexColor("#D1D5DB")
_BLUE   = colors.HexColor("#1D4ED8")


def _pdf_styles() -> dict:
    return {
        "name": ParagraphStyle(
            "name", fontName="Helvetica-Bold", fontSize=17,
            textColor=_DARK, spaceAfter=3, leading=22,
        ),
        "contact": ParagraphStyle(
            "contact", fontName="Helvetica", fontSize=9,
            textColor=_LIGHT, spaceAfter=0, leading=14,
        ),
        "date": ParagraphStyle(
            "date", fontName="Helvetica", fontSize=10,
            textColor=_MID, spaceBefore=12, spaceAfter=3,
        ),
        "recipient_bold": ParagraphStyle(
            "recipient_bold", fontName="Helvetica-Bold", fontSize=10,
            textColor=_DARK, spaceAfter=1,
        ),
        "recipient": ParagraphStyle(
            "recipient", fontName="Helvetica", fontSize=10,
            textColor=_MID, spaceAfter=1,
        ),
        "salute": ParagraphStyle(
            "salute", fontName="Helvetica-Bold", fontSize=10,
            textColor=_DARK, spaceBefore=16, spaceAfter=12,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=_MID, leading=17, spaceAfter=11,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=10,
            textColor=_MID, leading=16, spaceAfter=7,
            leftIndent=14, alignment=TA_JUSTIFY,
        ),
        "bullet_head": ParagraphStyle(
            "bullet_head", fontName="Helvetica-Bold", fontSize=10,
            textColor=_DARK, leading=16, spaceAfter=2, leftIndent=0,
        ),
        "sign": ParagraphStyle(
            "sign", fontName="Helvetica", fontSize=10,
            textColor=_MID, spaceBefore=16, spaceAfter=4,
        ),
        "signname": ParagraphStyle(
            "signname", fontName="Helvetica-Bold", fontSize=10,
            textColor=_DARK,
        ),
    }


def _escape_xml(text: str) -> str:
    """Escape special XML chars for ReportLab Paragraph."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def generate_cover_letter_pdf(
    cl_data:        dict,
    candidate_name: str  = "",
    contact_line:   str  = "",
    letter_date:    str  = "",
    recipient_name: str  = "Hiring Team",
    company_name:   str  = "",
) -> bytes:
    """
    Convert a cover letter JSON dict (from generate_cover_letter()) into a
    properly formatted PDF (bytes). Matches the Zahid Shaikh sample format.

    Args:
        cl_data:        Output of generate_cover_letter()
        candidate_name: Full name for header (extracted from resume/profile)
        contact_line:   "email | phone | location" string
        letter_date:    "May 14, 2026" — defaults to today
        recipient_name: Who to address ("Hiring Team", "Sarah Chen")
        company_name:   Company to address letter to

    Returns:
        PDF as bytes (use in FastAPI Response or write to file)
    """
    buf    = io.BytesIO()
    styles = _pdf_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    if not letter_date:
        letter_date = date.today().strftime("%B %d, %Y")

    story = []

    # ── Header: Name ─────────────────────────────────────────────────────────
    name_display = candidate_name or "Applicant"
    story.append(Paragraph(_escape_xml(name_display), styles["name"]))

    if contact_line:
        story.append(Paragraph(_escape_xml(contact_line), styles["contact"]))

    # Divider line
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_LINE, spaceAfter=10))

    # ── Date ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(_escape_xml(letter_date), styles["date"]))
    story.append(Spacer(1, 6))

    # ── Recipient ─────────────────────────────────────────────────────────────
    story.append(Paragraph(_escape_xml(recipient_name), styles["recipient_bold"]))
    if company_name:
        story.append(Paragraph(_escape_xml(company_name), styles["recipient_bold"]))

    # ── Salutation ────────────────────────────────────────────────────────────
    salute_to = company_name or recipient_name or "Hiring Team"
    story.append(Paragraph(
        _escape_xml(f"Dear {salute_to} Hiring Team,"),
        styles["salute"]
    ))

    # ── Opening paragraph ─────────────────────────────────────────────────────
    opening = cl_data.get("opening_paragraph", "")
    if opening:
        story.append(Paragraph(_escape_xml(opening), styles["body"]))

    # ── Body paragraph ────────────────────────────────────────────────────────
    body = cl_data.get("body_paragraph", "")
    if body:
        story.append(Paragraph(_escape_xml(body), styles["body"]))

    # ── Why I'm a strong fit heading (if bullets exist) ───────────────────────
    bullets = cl_data.get("bullets", [])
    if bullets:
        story.append(Paragraph("Why I am a strong fit for this role:", styles["body"]))
        for b in bullets:
            label = b.get("label", "")
            text  = b.get("text", "")
            if label and text:
                # Bold label inline + body text
                combined = f"<b>{_escape_xml(label)}:</b> {_escape_xml(text)}"
                story.append(Paragraph(f"• {combined}", styles["bullet"]))
            elif text:
                story.append(Paragraph(f"• {_escape_xml(text)}", styles["bullet"]))
        story.append(Spacer(1, 4))

    # ── Value-add paragraph ───────────────────────────────────────────────────
    value = cl_data.get("value_paragraph", "")
    if value:
        story.append(Paragraph(_escape_xml(value), styles["body"]))

    # ── Closing paragraph ─────────────────────────────────────────────────────
    closing = cl_data.get("closing_paragraph", "")
    if closing:
        story.append(Paragraph(_escape_xml(closing), styles["body"]))

    # ── Sign-off ──────────────────────────────────────────────────────────────
    sign_off = cl_data.get("sign_off", "Best regards,")
    story.append(Paragraph(_escape_xml(sign_off), styles["sign"]))
    story.append(Paragraph(_escape_xml(name_display), styles["signname"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()