"""
api.py — HireBot FastAPI server v0.5.0

Endpoints:
  GET  /                        → Serves frontend HTML
  POST /analyze                 → Resume parsing (Step 1)
  POST /analyze/intelligence    → LLM career intelligence (Step 2)
  POST /analyze/match           → JD matching via text paste (Step 3)
  POST /analyze/match/url       → JD matching via URL scrape (Step 3b)
  POST /analyze/optimize        → AI resume optimization (Step 4)
  POST /analyze/optimize/jd     → JD-targeted optimization (Step 4b)
  POST /analyze/batch-match     → Batch match: up to 10 JDs, shortlist + rank (Step 5a)
  POST /analyze/cover-letter    → AI cover letter generation (Step 5b JSON)
  POST /analyze/cover-letter/pdf → AI cover letter as downloadable PDF

Run:
  cd E:/projects/HireBot/app
  uvicorn api:app --reload --port 8000
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response

from parser.extract_text import extract_resume_text, extract_pdf_links, extract_professional_links
from parser.parse_resume import parse_basic_details, compute_total_experience_years
from intelligence.analyze import analyze_intelligence
from intelligence.match import analyze_match, fetch_jd_from_url
from intelligence.optimize import analyze_optimize
from intelligence.batch_match import analyze_batch_match
from intelligence.cover_letter import generate_cover_letter, generate_cover_letter_pdf

app = FastAPI(title="Hirebot API", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_SIZE_MB = 10


# ── Step 0: Serve frontend ────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/hirebot.html")


# ── Step 1: Resume parsing ────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported format: {ext}. Use PDF or DOCX.")
    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, detail="File too large. Max 10MB.")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content); tmp_path = tmp.name
    try:
        resume_text   = extract_resume_text(tmp_path)
        basic_details = parse_basic_details(resume_text)
        if not basic_details.get("total_experience"):
            basic_details["total_experience"] = compute_total_experience_years(
                basic_details.get("experience", [])
            )
        professional_links = {}
        if ext == ".pdf":
            links = extract_pdf_links(tmp_path)
            professional_links = extract_professional_links(links)
        return JSONResponse(content={
            **basic_details, **professional_links, "resume_text": resume_text
        })
    finally:
        os.unlink(tmp_path)


# ── Step 2: LLM Intelligence ──────────────────────────────────────────────────

@app.post("/analyze/intelligence")
async def resume_intelligence(request: Request):
    body        = await request.json()
    resume_text = body.get("resume_text", "").strip()
    if not resume_text:
        raise HTTPException(400, detail="'resume_text' is required.")
    try:
        return JSONResponse(content=analyze_intelligence(resume_text))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Intelligence error: {str(e)}")


# ── Step 3: Single JD Match ───────────────────────────────────────────────────

@app.post("/analyze/match")
async def match_resume_jd(request: Request):
    body        = await request.json()
    resume_text = body.get("resume_text", "").strip()
    jd_text     = body.get("jd_text", "").strip()
    if not resume_text: raise HTTPException(400, detail="'resume_text' is required.")
    if not jd_text:     raise HTTPException(400, detail="'jd_text' is required.")
    try:
        return JSONResponse(content=analyze_match(resume_text, jd_text))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Match error: {str(e)}")


@app.post("/analyze/match/url")
async def match_resume_jd_url(request: Request):
    body        = await request.json()
    resume_text = body.get("resume_text", "").strip()
    jd_url      = body.get("jd_url", "").strip()
    if not resume_text: raise HTTPException(400, detail="'resume_text' is required.")
    if not jd_url:      raise HTTPException(400, detail="'jd_url' is required.")
    if not jd_url.startswith("http"):
        raise HTTPException(400, detail="'jd_url' must start with http/https.")
    jd_text = fetch_jd_from_url(jd_url)
    if not jd_text or len(jd_text) < 50:
        raise HTTPException(
            422, detail="Could not extract JD from URL. Try pasting the JD text directly."
        )
    try:
        result = analyze_match(resume_text, jd_text)
        result["jd_text_preview"] = jd_text[:300] + "..." if len(jd_text) > 300 else jd_text
        result["jd_url"] = jd_url
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Match error: {str(e)}")


# ── Step 4: Optimization ──────────────────────────────────────────────────────

@app.post("/analyze/optimize")
async def optimize_resume(request: Request):
    body        = await request.json()
    resume_text = body.get("resume_text", "").strip()
    if not resume_text:
        raise HTTPException(400, detail="'resume_text' is required.")
    try:
        return JSONResponse(content=analyze_optimize(resume_text))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Optimize error: {str(e)}")


@app.post("/analyze/optimize/jd")
async def optimize_resume_jd(request: Request):
    body        = await request.json()
    resume_text = body.get("resume_text", "").strip()
    jd_text     = body.get("jd_text", "").strip()
    if not resume_text: raise HTTPException(400, detail="'resume_text' is required.")
    if not jd_text:     raise HTTPException(400, detail="'jd_text' is required for JD-targeted optimization.")
    try:
        return JSONResponse(content=analyze_optimize(resume_text, jd_text))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Optimize error: {str(e)}")


# ── Step 5a: Batch JD Match ───────────────────────────────────────────────────

@app.post("/analyze/batch-match")
async def batch_match_resume(request: Request):
    """
    Batch match a resume against 2-10 JDs simultaneously.

    Input:  { "resume_text": "...", "jd_list": ["JD1", "JD2", ...] }
    Output: { jobs[], shortlisted[], ranked[], universal_tips{}, batch_summary{} }
    """
    body        = await request.json()
    resume_text = body.get("resume_text", "").strip()
    jd_list     = body.get("jd_list", [])

    if not resume_text:
        raise HTTPException(400, detail="'resume_text' is required.")
    if not jd_list or not isinstance(jd_list, list):
        raise HTTPException(400, detail="'jd_list' must be a non-empty array of JD strings.")
    if len(jd_list) < 2:
        raise HTTPException(400, detail="'jd_list' must contain at least 2 job descriptions.")
    if len(jd_list) > 10:
        raise HTTPException(400, detail="'jd_list' cannot exceed 10 JDs per batch.")

    try:
        return JSONResponse(content=analyze_batch_match(resume_text, jd_list))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Batch match error: {str(e)}")


# ── Step 5b: Cover Letter JSON ────────────────────────────────────────────────

@app.post("/analyze/cover-letter")
async def cover_letter(request: Request):
    """
    Generate a targeted cover letter (JSON, for display in UI).

    Input:
        {
          "resume_text": "...",
          "jd_text": "...",
          "tone": "professional" | "enthusiastic" | "concise",
          "highlights": "...",       (optional)
          "candidate_name": "..."    (optional)
        }
    """
    body           = await request.json()
    resume_text    = body.get("resume_text", "").strip()
    jd_text        = body.get("jd_text", "").strip()
    tone           = body.get("tone", "professional").strip()
    highlights     = body.get("highlights", "").strip()
    candidate_name = body.get("candidate_name", "").strip()

    if not resume_text:
        raise HTTPException(400, detail="'resume_text' is required.")
    if not jd_text:
        raise HTTPException(400, detail="'jd_text' is required.")

    valid_tones = {"professional", "enthusiastic", "concise"}
    if tone not in valid_tones:
        tone = "professional"

    try:
        return JSONResponse(content=generate_cover_letter(
            resume_text    = resume_text,
            jd_text        = jd_text,
            tone           = tone,
            highlights     = highlights,
            candidate_name = candidate_name,
        ))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Cover letter error: {str(e)}")


# ── Step 5b: Cover Letter PDF Download ────────────────────────────────────────

@app.post("/analyze/cover-letter/pdf")
async def cover_letter_pdf(request: Request):
    """
    Generate a cover letter AND return it as a downloadable PDF.

    Input:
        {
          "resume_text": "...",
          "jd_text": "...",
          "tone": "professional",
          "highlights": "...",
          "candidate_name": "Zahid Shaikh",
          "contact_line": "zahid@email.com | +91-8286092787 | Mumbai, India",
          "letter_date": "May 14, 2026",      (optional, defaults to today)
          "recipient_name": "Hiring Team",     (optional)
          "company_name": "Yugabyte"           (optional)
        }

    Returns: PDF file (application/pdf)
    """
    body           = await request.json()
    resume_text    = body.get("resume_text", "").strip()
    jd_text        = body.get("jd_text", "").strip()
    tone           = body.get("tone", "professional").strip()
    highlights     = body.get("highlights", "").strip()
    candidate_name = body.get("candidate_name", "").strip()
    contact_line   = body.get("contact_line", "").strip()
    letter_date    = body.get("letter_date", "").strip()
    recipient_name = body.get("recipient_name", "Hiring Team").strip()
    company_name   = body.get("company_name", "").strip()

    if not resume_text:
        raise HTTPException(400, detail="'resume_text' is required.")
    if not jd_text:
        raise HTTPException(400, detail="'jd_text' is required.")

    valid_tones = {"professional", "enthusiastic", "concise"}
    if tone not in valid_tones:
        tone = "professional"

    try:
        # Step 1: Generate letter content via LLM
        cl_data = generate_cover_letter(
            resume_text    = resume_text,
            jd_text        = jd_text,
            tone           = tone,
            highlights     = highlights,
            candidate_name = candidate_name,
        )

        # Step 2: Render to PDF
        pdf_bytes = generate_cover_letter_pdf(
            cl_data        = cl_data,
            candidate_name = candidate_name,
            contact_line   = contact_line,
            letter_date    = letter_date,
            recipient_name = recipient_name,
            company_name   = company_name,
        )

        # Step 3: Return as downloadable file
        safe_name = (candidate_name or "Cover_Letter").replace(" ", "_")
        safe_co   = (company_name or "Application").replace(" ", "_")
        filename  = f"{safe_name}_{safe_co}_Cover_Letter.pdf"

        return Response(
            content      = pdf_bytes,
            media_type   = "application/pdf",
            headers      = {"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Cover letter PDF error: {str(e)}")