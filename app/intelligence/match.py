"""
intelligence/match.py
---------------------
Step 3 — Resume vs Job Description Matching.

Given resume text + JD text (or scraped from URL), returns:
  match_score, section_scores, matched_keywords, missing_keywords,
  transferable_skills, red_flags, ats_jd_score, tailoring_tips,
  recommendation, recommendation_reason, _model_used

Same OpenRouter pattern: gpt-oss-120b → llama-3.3-70b → openrouter/free
"""

import os
import json
import re
import requests
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]

MATCH_SYSTEM_PROMPT = """You are a senior technical recruiter and ATS specialist with 15+ years evaluating candidate-job fit across tech, data, and engineering roles in India and global markets.

Your task: Given a resume and a job description, return ONLY a valid JSON object. No preamble, no explanation, no markdown fences. Just raw JSON.

Use this EXACT schema:
{
  "match_score": 74,
  "section_scores": {
    "skills": 80,
    "experience": 70,
    "education": 65,
    "culture_and_soft": 75
  },
  "matched_keywords": ["Python", "FastAPI", "SQL", "Data Pipeline"],
  "missing_keywords": ["Spark", "Kubernetes", "dbt", "Airflow"],
  "transferable_skills": ["Strong SQL maps to their data warehouse work", "REST API experience covers their internal tooling needs"],
  "red_flags": ["No mention of cloud platforms despite JD requiring AWS", "Only 1 year experience vs 3+ required"],
  "ats_jd_score": 62,
  "tailoring_tips": [
    "Add AWS/GCP certifications or projects prominently",
    "Quantify data pipeline throughput — e.g. 'processed 5M rows/day' — to match their scale",
    "Use exact JD terms: 'data orchestration' instead of 'workflow automation'"
  ],
  "recommendation": "strong_consider",
  "recommendation_reason": "The candidate covers ~75% of technical requirements. Strong Python and SQL are exact matches. The primary gap is cloud infrastructure, which could be bridged quickly. Worth a first-round call."
}

Rules:
- match_score: 0-100. Skills 40%, experience 35%, education 15%, culture/soft 10%.
- section_scores: 0-100 for each of the 4 dimensions.
- matched_keywords: exact or equivalent skills/tools from JD found in resume. Max 15 items.
- missing_keywords: important JD requirements NOT in resume. Hard requirements first. Max 12 items.
- transferable_skills: 2-5 sentences where resume skill partially satisfies JD requirement. Be specific.
- red_flags: 0-4 items. Hard dealbreakers or major gaps only. Not minor quibbles.
- ats_jd_score: 0-100 how well this resume would pass ATS for THIS JD — keyword overlap, exact phrases, formatting.
- tailoring_tips: 3-5 specific actionable changes. Cite exact resume + JD terms.
- recommendation: exactly one of "hire" / "strong_consider" / "maybe" / "pass"
- recommendation_reason: 2-4 sentences. Balanced, specific, honest. Not generic boilerplate.
- Return ONLY the JSON. No other text, no markdown."""


def _build_match_message(resume_text: str, jd_text: str) -> str:
    return f"""Analyze this candidate against the job description and return the JSON match report.

--- RESUME START ---
{resume_text.strip()[:4000]}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{jd_text.strip()[:3000]}
--- JOB DESCRIPTION END ---"""


def _call_openrouter_match(model: str, resume_text: str, jd_text: str) -> Optional[dict]:
    api_key = os.getenv("OPENROUTER_API_KEY", "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")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HireBot JD Match",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": MATCH_SYSTEM_PROMPT},
            {"role": "user",   "content": _build_match_message(resume_text, jd_text)},
        ],
        "temperature": 0.15,
        "max_tokens": 1500,
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=90
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return json.loads(raw_text)

    except requests.exceptions.Timeout:
        print(f"[Match] Timeout: {model}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[Match] HTTP {e.response.status_code}: {model}")
        return None
    except json.JSONDecodeError as e:
        print(f"[Match] JSON parse error: {model}: {e}")
        return None
    except Exception as e:
        print(f"[Match] Error: {model}: {e}")
        return None


def _validate_match_result(result: dict) -> bool:
    required = {
        "match_score", "section_scores", "matched_keywords",
        "missing_keywords", "ats_jd_score", "tailoring_tips",
        "recommendation", "recommendation_reason"
    }
    return required.issubset(result.keys())


def fetch_jd_from_url(url: str) -> str:
    """Fetch and extract plain text from a job posting URL."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="ignore")

        # Strip script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]

    except Exception as e:
        print(f"[Match] JD URL fetch failed: {e}")
        return ""


def analyze_match(resume_text: str, jd_text: str) -> dict:
    """
    Match a resume against a JD. Returns structured match report.

    Raises:
        ValueError: Missing key / inputs too short
        RuntimeError: All models failed
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "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")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-..."
        )

    if not resume_text or len(resume_text.strip()) < 100:
        raise ValueError("Resume text is too short to analyze.")

    if not jd_text or len(jd_text.strip()) < 50:
        raise ValueError("Job description is too short. Paste at least 50 characters.")

    for model in FREE_MODELS:
        print(f"[Match] Trying model: {model}")
        result = _call_openrouter_match(model, resume_text, jd_text)
        if result and _validate_match_result(result):
            print(f"[Match] Success: {model}")
            result["_model_used"] = model
            return result
        print(f"[Match] Failed: {model}, trying next...")

    raise RuntimeError(
        "All OpenRouter free models failed for JD matching. "
        "Rate limit (200 req/day) likely hit. Try again later."
    )