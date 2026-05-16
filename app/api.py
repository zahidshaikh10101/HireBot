"""
api.py
------
HireBot FastAPI server.

Endpoints:
  GET  /                     → Serves frontend HTML
  POST /analyze              → Resume parsing (name, email, skills, experience...)
  POST /analyze/intelligence → LLM career intelligence (strengths, score, fit tags...)

Run:
  cd E:/projects/HireBot/app
  uvicorn api:app --reload --port 8000
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from parser.extract_text import extract_resume_text, extract_pdf_links, extract_professional_links
from parser.parse_resume import parse_basic_details, compute_total_experience_years
from intelligence.analyze import analyze_intelligence

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Hirebot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_SIZE_MB = 10


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/hirebot.html")


@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    """
    Step 1: Parse resume → extract structured profile.

    Input:  multipart/form-data, field 'file', accepts .pdf / .docx
    Output: JSON with name, email, phone, location, education, experience,
            skills, github, linkedin, portfolio, resume_text
    """
    # 1. Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported format: {ext}. Use PDF or DOCX.")

    # 2. Validate size
    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, detail="File too large. Max 10MB.")

    # 3. Write to temp file (parser needs a real path)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 4. Extract text + parse structured fields
        resume_text = extract_resume_text(tmp_path)
        basic_details = parse_basic_details(resume_text)

        # 5. Compute experience if regex didn't catch it
        if not basic_details.get("total_experience"):
            basic_details["total_experience"] = compute_total_experience_years(
                basic_details.get("experience", [])
            )

        # 6. Extract hyperlinks (PDF only)
        professional_links = {}
        if ext == ".pdf":
            links = extract_pdf_links(tmp_path)
            professional_links = extract_professional_links(links)

        # 7. Merge and return — resume_text is included so frontend
        #    can pass it directly to /analyze/intelligence
        result = {
            **basic_details,
            **professional_links,
            "resume_text": resume_text,
        }
        return JSONResponse(content=result)

    finally:
        os.unlink(tmp_path)    # always clean up


@app.post("/analyze/intelligence")
async def resume_intelligence(request: Request):
    """
    Step 2: LLM career intelligence on full resume text.

    Input:  JSON body { "resume_text": "..." }
    Output: JSON with strengths, weaknesses, career_domain,
            seniority_level, career_fit_tags, profile_score, _model_used
    """
    body = await request.json()
    resume_text = body.get("resume_text", "").strip()

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="'resume_text' is required and cannot be empty."
        )

    try:
        result = analyze_intelligence(resume_text)
        return JSONResponse(content=result)

    except ValueError as e:
        # API key missing or resume text too short
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        # All free models failed / rate limited
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intelligence error: {str(e)}")