"""
intelligence/optimize.py
------------------------
Step 4 — AI Resume Optimization.

Given full resume text (+ optional JD text for targeted rewrites), returns:
  - bullet_rewrites[]      each with: original, rewritten, improvement_reason, impact_score
  - summary_rewrite        { original, rewritten, improvement_reason }
  - skills_to_add[]        skills the market expects for this domain/seniority not on resume
  - headline_suggestion    one punchy professional headline string
  - overall_improvement    score delta estimate (e.g. "+12 ATS points expected")
  - quick_wins[]           3-5 immediate non-rewrite changes (formatting, section order, etc.)
  - _model_used

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

OPTIMIZE_SYSTEM_PROMPT = """You are an elite resume writer and career coach with 15+ years optimizing resumes for top-tier tech, data, and engineering roles in India and global markets. You specialize in ATS optimization, impact-driven bullet writing, and turning duty-lists into achievement stories.

Your task: Analyze the resume and return ONLY a valid JSON object. No preamble, no explanation, no markdown fences. Just raw JSON.

Use this EXACT schema:
{
  "bullet_rewrites": [
    {
      "original": "Exact original bullet or responsibility from the resume",
      "rewritten": "Strong rewrite: Action Verb + Context + Metric + Business Impact",
      "improvement_reason": "One sentence: what was weak and what the rewrite fixes",
      "impact_score": 8
    }
  ],
  "summary_rewrite": {
    "original": "Original summary/objective from resume, or 'No summary found' if absent",
    "rewritten": "Punchy 2-3 sentence professional summary: who you are + what you bring + what you target",
    "improvement_reason": "One sentence explaining what changed"
  },
  "headline_suggestion": "Senior Data Engineer | Python · Spark · AWS | 4 YOE building real-time pipelines at scale",
  "skills_to_add": [
    {"skill": "Apache Airflow", "reason": "Standard orchestration tool for Data Engineers — expected by 80% of JDs in this domain"},
    {"skill": "dbt", "reason": "Fast-growing data transformation layer — major signal of modern data stack experience"}
  ],
  "quick_wins": [
    "Move Skills section above Experience — ATS ranks keyword-dense sections higher when seen first",
    "Add a 2-3 line professional summary at the top — currently missing",
    "Quantify the internship bullets — even rough numbers like '50+ records' beat zero metrics",
    "Replace 'Responsible for' with action verbs like 'Built', 'Automated', 'Reduced'"
  ],
  "overall_improvement": "+14 ATS points expected after applying rewrites and adding missing keywords"
}

Rules:
- bullet_rewrites: Pick the 4-7 weakest bullets from the resume. Prioritize duty-lists with no metrics, passive voice, and generic descriptions. Skip bullets that are already strong.
- Each rewrite MUST follow: Strong Action Verb (past tense for past roles) + What you did + Scale/Context + Measurable Result.
- If no metric exists in original, invent a realistic estimate and flag it with "(est.)" in the rewrite.
- impact_score: 1-10. How much stronger the rewrite is vs original. 7+ means significant improvement.
- summary_rewrite: If no summary exists, write one from scratch based on their profile. Mark original as "No professional summary found".
- headline_suggestion: LinkedIn-style one-liner. Role title | Top 3 skills | Key differentiator.
- skills_to_add: 3-6 skills the market expects for their domain+seniority but are absent from resume. Each needs a specific reason.
- quick_wins: 3-5 non-rewrite structural/formatting changes. Concrete and actionable.
- overall_improvement: Honest estimate of ATS score improvement if all rewrites + quick wins applied.
- Return ONLY the JSON. No other text, no markdown."""


OPTIMIZE_JD_SYSTEM_PROMPT = """You are an elite resume writer specializing in tailoring resumes to specific job descriptions for maximum ATS and human reviewer impact.

Your task: Analyze the resume against the provided job description and return ONLY a valid JSON object with targeted rewrites. No preamble, no explanation, no markdown fences. Just raw JSON.

Use this EXACT schema:
{
  "bullet_rewrites": [
    {
      "original": "Exact original bullet from the resume",
      "rewritten": "JD-targeted rewrite using JD keywords: Action Verb + Context + Metric + JD-aligned Impact",
      "improvement_reason": "One sentence: which JD requirement this now addresses + what was fixed",
      "impact_score": 9,
      "jd_keywords_added": ["keyword1", "keyword2"]
    }
  ],
  "summary_rewrite": {
    "original": "Original summary or 'No summary found'",
    "rewritten": "JD-targeted summary: who you are + exact JD skills you have + why you fit this role",
    "improvement_reason": "One sentence"
  },
  "headline_suggestion": "Role title from JD | Top 3 JD-matching skills | Key differentiator",
  "skills_to_add": [
    {"skill": "Skill from JD not on resume", "reason": "Required/preferred in JD — add to skills section immediately"}
  ],
  "quick_wins": [
    "Specific action to close a JD gap without rewriting experience bullets"
  ],
  "overall_improvement": "+X ATS points expected vs this JD after applying all changes"
}

Rules:
- bullet_rewrites: Focus on 4-7 bullets that, once rewritten with JD keywords, will dramatically improve match. 
- Each rewrite must weave in exact JD terminology and requirements naturally — not keyword stuffing.
- jd_keywords_added: List exact JD terms added to the rewrite (for UI highlighting).
- impact_score: 1-10 vs the JD specifically. 9-10 = closes a critical JD gap.
- summary_rewrite: Must mirror the JD's language and explicitly address their top 2-3 requirements.
- skills_to_add: Only skills that appear in the JD that are missing from the resume.
- quick_wins: Fast wins specifically tied to JD gaps (section reorder, certifications to mention, etc.)
- Return ONLY the JSON. No other text, no markdown."""


def _build_optimize_message(resume_text: str, jd_text: str = "") -> str:
    if jd_text:
        return f"""Optimize this resume specifically for the job description below. Return the JSON optimization report.

--- RESUME START ---
{resume_text.strip()[:4000]}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{jd_text.strip()[:2500]}
--- JOB DESCRIPTION END ---"""
    else:
        return f"""Optimize this resume for maximum impact and ATS compatibility. Return the JSON optimization report.

--- RESUME START ---
{resume_text.strip()[:5000]}
--- RESUME END ---"""


def _call_openrouter_optimize(model: str, resume_text: str, jd_text: str = "") -> Optional[dict]:
    api_key = os.getenv("OPENROUTER_API_KEY", "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")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    system_prompt = OPTIMIZE_JD_SYSTEM_PROMPT if jd_text else OPTIMIZE_SYSTEM_PROMPT

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HireBot Resume Optimizer",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": _build_optimize_message(resume_text, jd_text)},
        ],
        "temperature": 0.4,   # Slightly higher — creative rewrites need some variance
        "max_tokens": 2000,   # Rewrites are verbose — need more tokens
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return json.loads(raw_text)

    except requests.exceptions.Timeout:
        print(f"[Optimize] Timeout: {model}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[Optimize] HTTP {e.response.status_code}: {model}")
        return None
    except json.JSONDecodeError as e:
        print(f"[Optimize] JSON parse error: {model}: {e}")
        return None
    except Exception as e:
        print(f"[Optimize] Error: {model}: {e}")
        return None


def _validate_optimize_result(result: dict) -> bool:
    required = {
        "bullet_rewrites", "summary_rewrite", "headline_suggestion",
        "skills_to_add", "quick_wins", "overall_improvement"
    }
    return required.issubset(result.keys()) and len(result.get("bullet_rewrites", [])) > 0


def analyze_optimize(resume_text: str, jd_text: str = "") -> dict:
    """
    Optimize a resume. If jd_text provided, rewrites are JD-targeted.

    Returns:
        dict with bullet_rewrites, summary_rewrite, headline_suggestion,
        skills_to_add, quick_wins, overall_improvement, _model_used, _jd_mode

    Raises:
        ValueError: Missing key / text too short
        RuntimeError: All models failed
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    if not resume_text or len(resume_text.strip()) < 100:
        raise ValueError("Resume text is too short to optimize.")

    for model in FREE_MODELS:
        print(f"[Optimize] Trying model: {model}")
        result = _call_openrouter_optimize(model, resume_text, jd_text)
        if result and _validate_optimize_result(result):
            print(f"[Optimize] Success: {model}")
            result["_model_used"] = model
            result["_jd_mode"] = bool(jd_text)
            return result
        print(f"[Optimize] Failed: {model}, trying next...")

    raise RuntimeError(
        "All OpenRouter free models failed for optimization. "
        "Rate limit (200 req/day) likely hit. Try again later."
    )