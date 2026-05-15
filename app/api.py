import os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Import YOUR existing parser modules ──
from parser.extract_text import extract_resume_text, extract_pdf_links, extract_professional_links
from parser.parse_resume import parse_basic_details, compute_total_experience_years

app = FastAPI(title="Hirebot API", version="1.0.0")

# ── CORS: allow the frontend to call this API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_SIZE_MB = 10

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):

    # 1. Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported format: {ext}. Use PDF or DOCX.")

    # 2. Validate file size
    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, detail="File too large. Max 10MB.")

    # 3. Save to temp file (your parser needs a file path)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 4. Call YOUR existing functions — unchanged
        resume_text = extract_resume_text(tmp_path)
        basic_details = parse_basic_details(resume_text)

        # 5. Compute total experience if not already extracted
        if not basic_details.get("total_experience"):
            basic_details["total_experience"] = compute_total_experience_years(
                basic_details.get("experience", [])
            )

        # 6. Extract links (PDF only)
        professional_links = {}
        if ext == ".pdf":
            links = extract_pdf_links(tmp_path)
            professional_links = extract_professional_links(links)

        # 7. Merge and return
        result = {**basic_details, **professional_links}
        return JSONResponse(content=result)

    finally:
        os.unlink(tmp_path)   # always clean up temp file

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/hirebot.html")