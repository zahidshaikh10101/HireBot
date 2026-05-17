"""
intelligence/batch_match.py
---------------------------
Step 5a — Batch Resume vs Multiple JDs.

Given resume text + list of JD texts (up to 10), runs match in parallel using
concurrent.futures. Supports dual API keys for 2x throughput on free tier.

Per-job result now includes:
  apply_verdict   — "APPLY" / "BORDERLINE" / "SKIP"
  verdict_reason  — why this verdict was given
  score_boosters  — specific things to add/change to push score above threshold

Returns:
  jobs[]           — each job with full match result
  shortlisted[]    — non-skip jobs sorted by match_score desc
  ranked[]         — compact ranking table with verdicts
  universal_tips{} — detailed cross-job resume changes
  batch_summary{}  — stats
  _models_used[]
"""

import os
import json
import re
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]

# Score thresholds
APPLY_THRESHOLD      = 65   # >= 65: APPLY — strong enough, go for it
BORDERLINE_THRESHOLD = 50   # 50-64: BORDERLINE — apply after targeted changes
# < 50: SKIP — too big a gap right now

# ─── Prompts ────────────────────────────────────────────────────────────────

MATCH_SYSTEM_PROMPT = """You are a senior technical recruiter and ATS specialist with 15+ years evaluating candidate-job fit across tech, data, and engineering roles in India and global markets.

Your task: Given a resume and a job description, return ONLY a valid JSON object. No preamble, no explanation, no markdown fences. Just raw JSON.

Use this EXACT schema:
{
  "job_title_guess": "Senior Data Engineer",
  "company_guess": "Unknown or extracted company name",
  "match_score": 74,
  "section_scores": {
    "skills": 80,
    "experience": 70,
    "education": 65,
    "culture_and_soft": 75
  },
  "matched_keywords": ["Python", "FastAPI", "SQL"],
  "missing_keywords": ["Spark", "Kubernetes", "dbt"],
  "transferable_skills": ["Strong SQL maps to their data warehouse work"],
  "red_flags": ["Only 1 year experience vs 3+ required"],
  "ats_jd_score": 62,
  "tailoring_tips": [
    "Add AWS/GCP certifications prominently",
    "Quantify data pipeline throughput"
  ],
  "score_boosters": [
    {
      "action": "Add Apache Spark to your Skills section and mention it in your pipeline experience bullet",
      "score_delta": "+8 pts",
      "effort": "low",
      "why": "Spark appears 6 times in this JD as a hard requirement — adding it bridges the biggest single gap"
    },
    {
      "action": "Rewrite your data pipeline bullet to mention scale: e.g. processed 5M events/day using Kafka + Python",
      "score_delta": "+5 pts",
      "effort": "low",
      "why": "JD requires high-throughput systems experience; your current bullet has no scale indicators"
    }
  ],
  "recommendation": "strong_consider",
  "recommendation_reason": "The candidate covers 75% of technical requirements. Worth a first-round call."
}

Rules:
- job_title_guess: Infer role title from JD. Be specific.
- company_guess: Extract company name from JD if visible, else Not specified.
- match_score: 0-100. Skills 40%, experience 35%, education 15%, culture/soft 10%.
- section_scores: 0-100 for each dimension.
- matched_keywords: exact/equivalent JD skills found in resume. Max 12 items.
- missing_keywords: important JD requirements NOT in resume. Max 10 items.
- transferable_skills: 1-4 sentences where resume skill partially satisfies JD.
- red_flags: 0-3 hard dealbreakers only.
- ats_jd_score: 0-100 ATS keyword overlap for THIS JD.
- tailoring_tips: 2-4 specific actionable changes.
- score_boosters: 2-4 SPECIFIC changes that would most increase match_score. Each must have:
    * action: Exact step (name the specific skill/bullet/section — not generic)
    * score_delta: Estimated score increase like +8 pts
    * effort: low (resume edit only) / medium (small project or cert) / high (new skill to learn)
    * why: One sentence tying this directly to a specific JD requirement
- recommendation: exactly one of hire / strong_consider / maybe / pass
- recommendation_reason: 2-3 sentences. Specific, honest.
- Return ONLY the JSON. No other text, no markdown."""


UNIVERSAL_TIPS_SYSTEM_PROMPT = """You are a senior resume strategist and career coach with 15+ years helping tech professionals optimize job applications. You specialize in identifying cross-job patterns and giving hyper-specific, actionable advice — not generic career platitudes.

Your task: Given a resume and match results across multiple job descriptions, generate DETAILED, SPECIFIC cross-job resume improvements that would raise scores across the MOST jobs simultaneously.

Return ONLY a valid JSON object. No preamble, no explanation, no markdown fences.

Use this EXACT schema:
{
  "universal_tips": [
    {
      "change": "Add Apache Airflow to your Skills section under Orchestration Tools",
      "detail": "Airflow is present in 7 of 9 JDs you matched against — making it your single highest-ROI addition. Add it to your skills section AND reference it in one experience bullet: e.g. Migrated 12 manual cron jobs to Apache Airflow DAGs, reducing failure rate by 40%. Even if you have only used it in a side project, add it with a GitHub link. Pair it with Prefect or Luigi if you have any exposure.",
      "impact": "Closes the most common gap across 7 of your 9 target JDs",
      "priority": "critical",
      "effort": "low",
      "affected_jobs": ["Senior Data Engineer at Flipkart", "Data Platform Engineer at Swiggy"],
      "before_example": "Skills: Python, SQL, Pandas, FastAPI",
      "after_example": "Skills: Python, SQL, Pandas, FastAPI, Apache Airflow, Prefect"
    }
  ],
  "skill_gap_frequency": [
    {
      "skill": "Apache Airflow",
      "missing_in_n_jobs": 7,
      "total_jobs": 9,
      "frequency_pct": 78,
      "urgency": "critical",
      "how_to_add": "Add to Skills section under Orchestration. Reference in 1 experience bullet. Side project on GitHub counts.",
      "market_signal": "Standard orchestration tool in modern data stacks — expected by default in 2025"
    }
  ],
  "resume_section_audit": {
    "summary": "Your summary is generic — it does not mention your domain, your stack, or your level. Rewrite to mirror the JD language your target roles use.",
    "experience_bullets": "6 of your 12 experience bullets start with Responsible for or Worked on — passive language that ATS systems and recruiters penalize. Rewrite to start with strong action verbs plus metrics.",
    "skills_section": "Your skills list is unstructured — group into categories like Languages, Databases, Cloud, Tools for faster ATS parsing and recruiter scanning.",
    "education": "Your degree is a strong signal — note whether to move it above or below Experience based on years of experience.",
    "missing_sections": ["Certifications even free ones count", "Projects section with GitHub links", "Publications or open-source contributions if any"]
  },
  "ats_optimization": {
    "current_ats_avg": 58,
    "target_ats": 75,
    "quick_fixes": [
      "Use exact JD terminology — say ETL pipelines not data workflows — ATS matches exact strings",
      "Add a dedicated Technical Skills section if it does not exist",
      "Spell out acronyms once like Machine Learning ML — some ATS systems do not match abbreviations"
    ]
  },
  "overall_readiness": "You are a genuine fit for 4 of 9 roles right now. With 3-4 targeted resume changes you would qualify for 7 of 9. Your Python plus SQL depth is a strong market signal.",
  "recommended_focus": "Prioritize mid-level Data Engineering roles at product companies. Avoid pure DevOps roles until you build more cloud infrastructure depth.",
  "salary_positioning": "Based on your profile position yourself in the 12-18 LPA range. Your MTech plus engineering pedigree justifies the higher end if you add cloud exposure."
}

Rules:
- universal_tips: 5-8 changes. Each must:
  * Be SPECIFIC — name the exact skill, section, bullet, or JD term
  * Include before/after examples showing the exact text change
  * Have a detail field of 3-5 sentences explaining HOW to make this change
  * Only recommend changes that benefit 2 or more of the shortlisted jobs
  * effort: low under 30 min resume edit / medium small project or cert / high new skill weeks
- skill_gap_frequency: Top 6-8 skills sorted by missing_in_n_jobs desc. Each must include:
  * frequency_pct as missing_in_n_jobs divided by total_jobs times 100 rounded
  * urgency: critical above 60 percent / high 40-60 percent / medium 20-40 percent
  * how_to_add: Exactly HOW to add this to the resume which section what to write
  * market_signal: Why this skill matters in the current job market
- resume_section_audit: Honest audit of each major resume section. Be specific and critical.
  * missing_sections: List sections absent but would help for these roles
- ats_optimization: Current avg ATS score across jobs, target, and 3 specific fixes
- overall_readiness: 2-3 sentences. Specific numbers X of Y jobs. Honest.
- recommended_focus: Which job types to prioritize and which to avoid.
- salary_positioning: Salary range advice based on their profile and target market.
- Return ONLY the JSON. No other text, no markdown."""


# ─── Core LLM call ──────────────────────────────────────────────────────────

def _get_api_key(index: int = 0) -> str:
    key1 = os.getenv("OPENROUTER_API_KEY", "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")
    key2 = os.getenv("OPENROUTER_API_KEY_2", "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")
    if index % 2 == 1 and key2:
        return key2
    return key1


def _call_openrouter(model: str, system_prompt: str, user_message: str,
                     api_key: str, temperature: float = 0.15,
                     max_tokens: int = 1800) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HireBot Batch Match",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
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
        print(f"[BatchMatch] Timeout: {model}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[BatchMatch] HTTP {e.response.status_code}: {model}")
        return None
    except json.JSONDecodeError as e:
        print(f"[BatchMatch] JSON parse error ({model}): {e}")
        return None
    except Exception as e:
        print(f"[BatchMatch] Error ({model}): {e}")
        return None


def _compute_apply_verdict(result: dict) -> dict:
    """Add apply_verdict and verdict_reason based on score + red flags."""
    score     = result.get("match_score", 0)
    rec       = result.get("recommendation", "pass")
    red_flags = result.get("red_flags", [])

    if score >= APPLY_THRESHOLD and rec in ("hire", "strong_consider"):
        verdict = "APPLY"
        reason  = (
            f"Strong fit at {score}% — your profile covers the core requirements. "
            "Apply now and tailor your resume with the Score Boosters below."
        )
    elif score >= BORDERLINE_THRESHOLD or rec == "maybe":
        verdict = "BORDERLINE"
        reason  = (
            f"Partial fit at {score}% — close but not there yet. "
            "Apply after implementing the Score Boosters; they could push you to 65%+."
        )
    else:
        verdict = "SKIP"
        reason  = (
            f"Low fit at {score}% — too many core requirements are missing right now. "
            "Use the Score Boosters to build toward this role type, then revisit in 3–6 months."
        )

    # Hard red flags cap APPLY down to BORDERLINE
    if verdict == "APPLY" and len(red_flags) >= 2:
        verdict = "BORDERLINE"
        reason  = (
            f"Good score ({score}%) but {len(red_flags)} hard gaps flagged. "
            "Worth applying, but address the red flags directly in your cover letter."
        )

    result["apply_verdict"] = verdict
    result["verdict_reason"] = reason
    return result


def _match_single_job(job_index: int, resume_text: str, jd_text: str) -> dict:
    """Match one JD against the resume. Called in thread pool."""
    api_key  = _get_api_key(job_index)
    user_msg = f"""Analyze this candidate against the job description and return the JSON match report.

--- RESUME START ---
{resume_text.strip()[:3500]}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{jd_text.strip()[:2500]}
--- JOB DESCRIPTION END ---"""

    for model in FREE_MODELS:
        print(f"[BatchMatch] Job {job_index+1} trying {model}")
        result = _call_openrouter(model, MATCH_SYSTEM_PROMPT, user_msg, api_key)
        if result and _validate_match(result):
            print(f"[BatchMatch] Job {job_index+1} OK: {model}")
            result["_model_used"] = model
            result["_job_index"]  = job_index
            result = _compute_apply_verdict(result)
            return result
        print(f"[BatchMatch] Job {job_index+1} failed {model}, trying next...")

    return {
        "_job_index":    job_index,
        "_error":        "All models failed for this job.",
        "match_score":   0,
        "recommendation":"pass",
        "apply_verdict": "SKIP",
        "verdict_reason":"Could not analyze this job — API error.",
        "job_title_guess": f"Job {job_index+1}",
        "company_guess": "Unknown",
        "score_boosters": [],
    }


def _validate_match(result: dict) -> bool:
    required = {"match_score", "section_scores", "matched_keywords",
                "missing_keywords", "ats_jd_score", "recommendation",
                "recommendation_reason"}
    return required.issubset(result.keys())


def _generate_universal_tips(resume_text: str, shortlisted_results: list,
                              total_jobs: int) -> dict:
    api_key = _get_api_key(0)

    job_summaries = []
    for r in shortlisted_results:
        job_summaries.append({
            "job":             f"{r.get('job_title_guess','?')} at {r.get('company_guess','?')}",
            "match_score":     r.get("match_score", 0),
            "ats_jd_score":    r.get("ats_jd_score", 0),
            "missing_keywords":r.get("missing_keywords", []),
            "matched_keywords":r.get("matched_keywords", [])[:6],
            "red_flags":       r.get("red_flags", []),
            "tailoring_tips":  r.get("tailoring_tips", []),
            "score_boosters":  [b.get("action", "") for b in r.get("score_boosters", [])],
            "section_scores":  r.get("section_scores", {}),
            "apply_verdict":   r.get("apply_verdict", "?"),
        })

    avg_ats = round(
        sum(j["ats_jd_score"] for j in job_summaries) / len(job_summaries), 0
    ) if job_summaries else 0

    user_msg = f"""Analyze the resume against these {len(job_summaries)} shortlisted job match results (out of {total_jobs} total) and return DETAILED, SPECIFIC universal resume improvement recommendations.

Current average ATS score across shortlisted jobs: {avg_ats}%

--- RESUME START ---
{resume_text.strip()[:3000]}
--- RESUME END ---

--- SHORTLISTED JOB MATCH RESULTS ---
{json.dumps(job_summaries, indent=2)}
--- END RESULTS ---

Be hyper-specific. Name exact skills, exact sections, exact bullet rewrites. Include before/after examples. Do not give generic advice — say WHICH bullets and HOW to change them."""

    for model in FREE_MODELS:
        print(f"[BatchMatch] Universal tips trying {model}")
        result = _call_openrouter(
            model, UNIVERSAL_TIPS_SYSTEM_PROMPT, user_msg, api_key,
            temperature=0.2, max_tokens=2500
        )
        if result and "universal_tips" in result:
            print(f"[BatchMatch] Universal tips OK: {model}")
            result["_model_used"] = model
            return result
        print(f"[BatchMatch] Universal tips failed {model}")

    return {
        "universal_tips":      [],
        "skill_gap_frequency": [],
        "resume_section_audit":{},
        "ats_optimization":    {},
        "overall_readiness":   "Could not generate universal tips — rate limit hit.",
        "recommended_focus":   "Please try again later.",
        "_error":              "All models failed for universal tips.",
    }


# ─── Main Entry ─────────────────────────────────────────────────────────────

def analyze_batch_match(resume_text: str, jd_list: list) -> dict:
    if not resume_text or len(resume_text.strip()) < 100:
        raise ValueError("Resume text is too short.")
    if not jd_list or len(jd_list) < 2:
        raise ValueError("Please provide at least 2 job descriptions for batch matching.")
    if len(jd_list) > 10:
        raise ValueError("Maximum 10 JDs per batch.")

    jd_list = [j.strip() for j in jd_list if j and len(j.strip()) >= 50]
    if len(jd_list) < 2:
        raise ValueError("At least 2 non-empty JDs (50+ chars each) are required.")

    print(f"[BatchMatch] Starting parallel match: {len(jd_list)} JDs")

    results_by_index = {}
    max_workers = min(len(jd_list), 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_match_single_job, i, resume_text, jd): i
            for i, jd in enumerate(jd_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_index[idx] = future.result()
            except Exception as e:
                results_by_index[idx] = {
                    "_job_index":    idx,
                    "_error":        str(e),
                    "match_score":   0,
                    "recommendation":"pass",
                    "apply_verdict": "SKIP",
                    "verdict_reason":f"Processing error: {str(e)}",
                    "job_title_guess": f"Job {idx+1}",
                    "company_guess": "Unknown",
                    "score_boosters": [],
                }

    all_jobs = []
    for i in range(len(jd_list)):
        job = results_by_index.get(i, {"_job_index": i, "_error": "No result"})
        job["job_number"] = i + 1
        job["jd_preview"] = (
            jd_list[i][:200] + "..." if len(jd_list[i]) > 200 else jd_list[i]
        )
        all_jobs.append(job)

    shortlisted = sorted(
        [j for j in all_jobs
         if j.get("recommendation") != "pass" and not j.get("_error")],
        key=lambda x: x.get("match_score", 0),
        reverse=True,
    )

    ranked = []
    for rank, job in enumerate(shortlisted):
        entry = {
            "rank":             rank + 1,
            "job_number":       job["job_number"],
            "job_title_guess":  job.get("job_title_guess", f"Job {job['job_number']}"),
            "company_guess":    job.get("company_guess", "Unknown"),
            "match_score":      job.get("match_score", 0),
            "recommendation":   job.get("recommendation", "maybe"),
            "apply_verdict":    job.get("apply_verdict", "SKIP"),
            "verdict_reason":   job.get("verdict_reason", ""),
            "ats_jd_score":     job.get("ats_jd_score", 0),
            "top_matched":      job.get("matched_keywords", [])[:5],
            "top_missing":      job.get("missing_keywords", [])[:5],
            "score_boosters":   job.get("score_boosters", []),
            "recommendation_reason": job.get("recommendation_reason", ""),
        }
        if rank > 0:
            entry["score_above_next"] = (
                shortlisted[rank - 1].get("match_score", 0) - job.get("match_score", 0)
            )
        ranked.append(entry)

    universal_tips_result = (
        _generate_universal_tips(resume_text, shortlisted, len(jd_list))
        if shortlisted else {
            "universal_tips":      [],
            "skill_gap_frequency": [],
            "resume_section_audit":{"summary": "No jobs were shortlisted."},
            "ats_optimization":    {},
            "overall_readiness":   "None of your jobs were shortlisted.",
            "recommended_focus":   "Consider applying to more entry/mid-level roles.",
        }
    )

    scored_jobs = [j for j in all_jobs if not j.get("_error")]
    scores      = [j.get("match_score", 0) for j in scored_jobs]

    batch_summary = {
        "total_jobs":       len(jd_list),
        "shortlisted_count":len(shortlisted),
        "passed_count":     len(jd_list) - len(shortlisted),
        "avg_match_score":  round(sum(scores) / len(scores), 1) if scores else 0,
        "top_match_score":  max(scores) if scores else 0,
        "verdict_breakdown": {
            "APPLY":      len([j for j in all_jobs if j.get("apply_verdict") == "APPLY"]),
            "BORDERLINE": len([j for j in all_jobs if j.get("apply_verdict") == "BORDERLINE"]),
            "SKIP":       len([j for j in all_jobs if j.get("apply_verdict") == "SKIP"]),
        },
        "recommendation_breakdown": {
            "hire":            len([j for j in all_jobs if j.get("recommendation") == "hire"]),
            "strong_consider": len([j for j in all_jobs if j.get("recommendation") == "strong_consider"]),
            "maybe":           len([j for j in all_jobs if j.get("recommendation") == "maybe"]),
            "pass":            len([j for j in all_jobs if j.get("recommendation") == "pass"]),
        },
    }

    return {
        "jobs":          all_jobs,
        "shortlisted":   shortlisted,
        "ranked":        ranked,
        "universal_tips":universal_tips_result,
        "batch_summary": batch_summary,
        "_models_used":  list({
            j.get("_model_used", "unknown")
            for j in all_jobs if j.get("_model_used")
        }),
    }