"""
intelligence/analyze.py
-----------------------
Resume Intelligence using OpenRouter (free models).
Analyzes full resume text and returns structured career insights.

Free models used (in priority order):
  1. openai/gpt-oss-120b:free  — best quality, 120B params, native structured output
  2. meta-llama/llama-3.3-70b-instruct:free — reliable fallback, GPT-4 class
  3. openrouter/free           — random free model (last resort)

OpenRouter is OpenAI-compatible — base_url swap is all it takes.
"""

import os
import json
import re
import requests
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ⚠️  Key is read at call-time (inside functions), NOT here at module level.
# Reading os.getenv() here would freeze an empty string if this module gets
# imported before load_dotenv() runs — easy to do by accident in test scripts.

FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]

# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior career intelligence analyst and compensation specialist with 15+ years across tech, data, and engineering hiring in India and global markets.

Your task: Analyze a resume and return ONLY a valid JSON object. No preamble, no explanation, no markdown fences. Just raw JSON.

Use this exact schema:
{
  "strengths": [
    "Specific strength backed by evidence from resume — quote exact project/metric/tool (2-3 sentences)"
  ],
  "weaknesses": [
    "Honest, constructive gap with specific explanation — not generic advice"
  ],
  "career_domain": "Most accurate primary domain e.g. Data Engineering & BI / ML Engineering / Backend Engineering",
  "seniority_level": "One of: Junior / Mid-Level / Senior / Lead / Principal",
  "career_fit_tags": ["Role title 1", "Role title 2"],
  "profile_score": 75,
  "ats_score": 68,
  "salary": {
    "current_estimate": "₹X–Y LPA",
    "target_range": "₹X–Y LPA",
    "market_percentile": "Top X%",
    "basis": "One sentence explaining what drives this estimate — years of experience, tech stack, company tier, location, domain"
  }
}

Rules:
- strengths: exactly 3–5 items. Cite specific resume evidence (project name, metric, tool). Never generic.
- weaknesses: exactly 2–4 items. Specific and constructive, not boilerplate.
- career_domain: single string, most accurate primary domain.
- seniority_level: based on years, scope, autonomy, and complexity of work — not just title.
- career_fit_tags: 3–6 role titles this person is genuinely a fit for.
- profile_score: 0–100 holistic score. Factors: clarity, achievement depth, skill relevance, presentation, impact evidence.
- ats_score: 0–100. How well this resume would pass ATS systems. Factors: keyword density for their domain, formatting clarity, measurable outcomes, standard section headers, no tables/columns/images. Be honest — most resumes score 55–75.
- salary.current_estimate: what they are LIKELY earning NOW based on role, company tier, years of experience, location, and Indian market data. Use ₹ LPA format.
- salary.target_range: realistic NEXT role salary they should target, accounting for their seniority jump and market demand for their skills.
- salary.market_percentile: where they sit in the market for their role+city+experience e.g. "Top 30%" or "Top 15%".
- salary.basis: one sentence max. Specific — mention tech stack, company tier, city, or experience factor that anchors the number.
- Return ONLY the JSON. Absolutely no other text, no markdown."""


def _build_user_message(resume_text: str) -> str:
    return f"""Analyze this resume and return the JSON intelligence report:

--- RESUME START ---
{resume_text.strip()}
--- RESUME END ---"""


# ── Core API call ────────────────────────────────────────────────────────────

def _call_openrouter(model: str, resume_text: str) -> Optional[dict]:
    """
    Makes a single API call to OpenRouter with the given model.
    Returns parsed dict on success, None on failure.
    Key is read fresh here so it always picks up the loaded .env value.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HireBot Intelligence",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(resume_text)},
        ],
        "temperature": 0.2,   # Lower = more deterministic salary/score numbers
        "max_tokens": 1200,   # Slightly more for the richer salary object
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        raw_text = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if model wraps JSON anyway
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        return json.loads(raw_text)

    except requests.exceptions.Timeout:
        print(f"[Intelligence] Timeout with model: {model}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[Intelligence] HTTP error with model {model}: {e.response.status_code}")
        return None
    except json.JSONDecodeError as e:
        print(f"[Intelligence] JSON parse error with model {model}: {e}")
        return None
    except Exception as e:
        print(f"[Intelligence] Unexpected error with model {model}: {e}")
        return None


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_result(result: dict) -> bool:
    """Basic schema validation — all required keys must be present."""
    required = {
        "strengths", "weaknesses", "career_domain",
        "seniority_level", "career_fit_tags", "profile_score",
        "ats_score", "salary"
    }
    return required.issubset(result.keys())


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_intelligence(resume_text: str) -> dict:
    """
    Analyze resume text and return career intelligence.
    Tries free models in order, falls back gracefully.

    Returns dict with keys:
        strengths, weaknesses, career_domain, seniority_level,
        career_fit_tags, profile_score, ats_score, salary

    Raises:
        ValueError: If API key is missing or text too short
        RuntimeError: If all models fail
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set.\n"
            "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-...\n"
            "Get a free key at: https://openrouter.ai/keys"
        )

    if not resume_text or len(resume_text.strip()) < 100:
        raise ValueError("Resume text is too short to analyze.")

    for model in FREE_MODELS:
        print(f"[Intelligence] Trying model: {model}")
        result = _call_openrouter(model, resume_text)

        if result and _validate_result(result):
            print(f"[Intelligence] Success with model: {model}")
            result["_model_used"] = model
            return result
        else:
            print(f"[Intelligence] Failed or invalid from: {model}, trying next...")

    raise RuntimeError(
        "All OpenRouter free models failed. "
        "Rate limit (200 req/day) likely hit. Try again later."
    )