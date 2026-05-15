import re
import datetime

# spaCy is optional — if the model isn't installed, NER-based extractors
# (name, location) will gracefully return None instead of crashing
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


# ════════════════════════════════════════════════
#  BASIC CONTACT EXTRACTORS
# ════════════════════════════════════════════════

def extract_email(text):
    """
    Finds the first email address in the text using a standard RFC-style regex.
    Returns the email string or None if not found.
    """
    pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    match = re.search(pattern, text)
    return match.group() if match else None


def extract_phone(text):
    """
    Extracts an Indian mobile number from text.
    Handles optional +91 country code with space or hyphen separator.
    Indian mobile numbers start with digits 6–9 followed by 9 more digits.
    Returns the matched string or None.
    """
    pattern = r'(\+91[-\s]?)?[6-9]\d{9}'
    match = re.search(pattern, text)
    return match.group() if match else None


def extract_name(text):
    """
    Uses spaCy Named Entity Recognition (NER) to detect a PERSON entity
    from the first 1000 characters — the top of the resume where the
    candidate's name almost always appears.

    Falls back to None if:
    - spaCy model failed to load (nlp is None)
    - No PERSON entity is detected in that section
    """
    # Guard: if spaCy didn't load, skip NER entirely
    if nlp is None:
        return None

    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def extract_location(text):
    """
    Uses spaCy NER to find the first geopolitical entity (GPE — cities,
    countries, states) or general location (LOC) in the first 1000 chars.

    Falls back to None if spaCy isn't available or no location is found.
    GPE covers: "Mumbai", "India", "Maharashtra"
    LOC covers: broader geographic regions
    """
    # Guard: if spaCy didn't load, skip NER entirely
    if nlp is None:
        return None

    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text
    return None


# ════════════════════════════════════════════════
#  EXPERIENCE YEAR EXTRACTOR (regex fallback)
# ════════════════════════════════════════════════

def extract_total_experience(text):
    """
    Regex-based fallback to extract total years of experience when
    a candidate explicitly writes it (e.g. "5+ years of experience").

    NOTE: compute_total_experience_years() calculated from date ranges
    is more reliable. This is only used when that returns 0 or None.

    Matches patterns like:
      "3 years of experience"
      "5+ years of experience"
    Returns an int or None.
    """
    pattern = r'(\d+)\+?\s+years?\s+of\s+experience'
    match = re.search(pattern, text.lower())
    if match:
        return int(match.group(1))
    return None


# ════════════════════════════════════════════════
#  SECTION DETECTION HELPERS
# ════════════════════════════════════════════════

def _find_section_header(text, headings, start_pos=0):
    """
    Scans the text line by line (starting from start_pos) looking for a
    line that begins with one of the given heading strings.

    Matching rules:
    - Case-insensitive
    - The heading must appear at the START of the line (after optional whitespace)
    - Must be followed by whitespace, end of line, or a colon — prevents
      partial matches like "experienced" matching "experience"

    Returns the character position of the matched line's start in the
    original text, or None if no heading is found.

    Used by all section extractors (education, experience, skills) to
    locate where each section begins.
    """
    lines = text[start_pos:].split('\n')
    cumulative_pos = start_pos

    for line in lines:
        lowered_line = line.lower().strip()
        for heading in headings:
            if re.match(r'^\s*' + re.escape(heading) + r'(?:\s|$|:)', lowered_line):
                # Return the position of the first non-whitespace char on this line
                return cumulative_pos + len(line) - len(line.lstrip())
        cumulative_pos += len(line) + 1  # +1 accounts for the '\n' character

    return None


# ════════════════════════════════════════════════
#  EDUCATION EXTRACTION
# ════════════════════════════════════════════════

def extract_education_section(text):
    """
    Isolates the raw education section text from the full resume.

    Strategy:
    1. Find the start using common education heading variants
    2. Find the next major section heading after it (stop_headings)
    3. Return only the text between those two positions

    Returns the section as a string, or None if no education heading found.
    """
    headings = [
        "education",
        "educational qualifications",
        "education & training",
        "academic qualifications",
        "academic",
        "education:"
    ]

    # These headings signal the end of the education section
    stop_headings = [
        "experience",
        "work experience",
        "professional experience",
        "skills",
        "projects",
        "certifications",
        "achievements",
    ]

    start_idx = _find_section_header(text, headings)
    if start_idx is None:
        return None

    # Default end is the full document; narrow it down to the next section
    end_idx = len(text)
    for s in stop_headings:
        idx = _find_section_header(text, [s], start_idx)
        if idx is not None and idx < end_idx:
            end_idx = idx

    return text[start_idx:end_idx].strip()


def extract_education(text):
    """
    Parses the education section into a structured list of entries.

    Each entry is a dict with keys:
        institution, degree, start_year, end_year

    Parsing strategy (in order of priority):
    1. Look for a year range on the CURRENT line (e.g. "2016 – 2020")
    2. If not found, look for a year range on the NEXT line
       (some resume formats put dates below the institution name)
    3. Fallback: if a line contains a degree keyword (bachelor, master, etc.)
       treat it as a degree-only entry with no dates

    Year ranges can use: –, —, -, or "to" as separators.
    End year can also be: "present", "current", "now" (case-insensitive).
    """
    education_section = extract_education_section(text)
    if not education_section:
        return []

    lines = [l.strip() for l in education_section.split("\n") if l.strip()]
    education_list = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Pattern: "2016 – 2020" or "2018 to Present"
        range_match = re.search(
            r"(\d{4})\s*(?:[–—-]|to)\s*(\d{4}|present|current|now)",
            line, re.IGNORECASE
        )

        # ── Case: year range is on the NEXT line ──
        if not range_match and i + 1 < len(lines):
            range_match = re.search(
                r"(\d{4})\s*(?:[–—-]|to)\s*(\d{4}|present|current|now)",
                lines[i + 1], re.IGNORECASE
            )
            if range_match:
                # Current line is the institution name, next line has dates,
                # line after that (if it exists) is likely the degree
                institution = line
                start_year = range_match.group(1)
                end_year = range_match.group(2).capitalize()
                degree = lines[i + 2] if i + 2 < len(lines) else None
                education_list.append({
                    "institution": institution,
                    "degree": degree,
                    "start_year": start_year,
                    "end_year": end_year
                })
                i += 3
                continue

        # ── Case: year range is on the CURRENT line ──
        if range_match:
            start_year = range_match.group(1)
            end_year = range_match.group(2).capitalize()

            # Everything before the date range on the same line = institution name
            institution = re.sub(range_match.re.pattern, "", line, flags=re.IGNORECASE).strip()
            if not institution:
                # If nothing before the date, fall back to the previous line
                institution = lines[i - 1] if i - 1 >= 0 else None

            # The next line is the degree IF it doesn't also contain a year
            # (avoids accidentally treating another date line as a degree)
            degree = (
                lines[i + 1]
                if i + 1 < len(lines) and not re.search(r"\d{4}", lines[i + 1])
                else None
            )

            education_list.append({
                "institution": institution,
                "degree": degree,
                "start_year": start_year,
                "end_year": end_year
            })
            i += 2

        else:
            # ── Fallback: no year range found ──
            # Heuristic: if the line contains a known degree keyword,
            # treat it as a degree-only entry (institution and dates unknown)
            degree_keywords = [
                "bachelor", "master", "b.sc", "m.sc", "phd",
                "degree", "ba", "bs", "mba", "b.e", "m.e"
            ]
            if any(k in line.lower() for k in degree_keywords):
                education_list.append({
                    "institution": None,
                    "degree": line,
                    "start_year": None,
                    "end_year": None
                })
            i += 1

    return education_list


# ════════════════════════════════════════════════
#  EXPERIENCE EXTRACTION
# ════════════════════════════════════════════════

def extract_experience_section(text):
    """
    Isolates the raw work experience section from the full resume text.
    Same start/stop strategy as extract_education_section().
    Returns the section as a string, or None if not found.
    """
    headings = [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "experience:"
    ]

    stop_headings = [
        "education",
        "skills",
        "projects",
        "certifications",
        "achievements",
    ]

    start_idx = _find_section_header(text, headings)
    if start_idx is None:
        return None

    end_idx = len(text)
    for s in stop_headings:
        idx = _find_section_header(text, [s], start_idx)
        if idx is not None and idx < end_idx:
            end_idx = idx

    return text[start_idx:end_idx].strip()


def _parse_year(token):
    """
    Converts a date token into an integer year.

    Handles:
    - "present" / "current" / "now"  → current calendar year
    - Any string containing a 4-digit year → extracts and returns it as int
    - Anything unrecognisable → returns None
    """
    token = token.strip().lower()
    if token in ("present", "current", "now"):
        return datetime.datetime.now().year
    try:
        return int(re.search(r"(\d{4})", token).group(1))
    except Exception:
        return None


def extract_experience(text):
    """
    Parses the experience section into a structured list of job entries.

    Each entry is a dict with keys:
        company, position, start_date, end_date, start_year, end_year

    Date detection uses 4 regex patterns (tried in priority order):
    1. "Jan 2020 – Mar 2022"   (month-name + year range)
    2. "2020 – 2022"           (year-only range)
    3. "Jan 2020 to Mar 2022"  (month-name + year with 'to')
    4. "2020 to 2022"          (year-only with 'to')

    Role/company parsing tries to handle these common resume formats:
    - "Senior Engineer at Google – Mumbai"   → split on 'at'
    - "Senior Engineer, Google – Mumbai"     → split on ','
    - "Senior Engineer – Google"             → split on '–'
    - Dates on the next line, role/company on the line above

    For each date-matched line, role info is extracted from the text
    BEFORE the date. If nothing is found there, the previous line is used.
    """
    section = extract_experience_section(text)
    if not section:
        return []

    lines = [line.strip() for line in section.split("\n") if line.strip()]
    exp_list = []

    # All supported date range formats, tried in order
    date_patterns = [
        # "Jan 2020 – Mar 2022" or "January 2020 - Present"
        re.compile(
            r'([A-Za-z]{3,9}\s+\d{4})\s*[–—-]\s*([A-Za-z]{3,9}\s+\d{4}|present|current|now)',
            re.IGNORECASE
        ),
        # "2020 – 2022" or "2020 - Present"
        re.compile(
            r'(\d{4})\s*[–—-]\s*(\d{4}|present|current|now)',
            re.IGNORECASE
        ),
        # "Jan 2020 to Mar 2022"
        re.compile(
            r'([A-Za-z]{3,9}\s+\d{4})\s*to\s*([A-Za-z]{3,9}\s+\d{4}|present|current|now)',
            re.IGNORECASE
        ),
        # "2020 to 2022"
        re.compile(
            r'(\d{4})\s*to\s*(\d{4}|present|current|now)',
            re.IGNORECASE
        ),
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        match = None
        matched_line_index = i

        # Try to find a date range on the current line
        for pat in date_patterns:
            m = pat.search(line)
            if m:
                match = m
                break

        # If not found, try the next line (some formats put dates below the role)
        if not match and i + 1 < len(lines):
            for pat in date_patterns:
                m = pat.search(lines[i + 1])
                if m:
                    match = m
                    matched_line_index = i + 1
                    break

        if match:
            matched_line = lines[matched_line_index]
            start_token = match.group(1)
            end_token = match.group(2)

            # Text before the date on the same line likely contains role/company
            role_info = matched_line[:match.start()].strip()

            position = None
            company = None

            if not role_info:
                # ── Dates on their own line: look at the line above for role/company ──
                if matched_line_index - 1 >= 0:
                    prev = lines[matched_line_index - 1]

                    if re.search(r'\bat\b', prev, re.IGNORECASE):
                        # Format: "Senior Engineer at Google"
                        parts = re.split(r'\s+at\s+', prev, flags=re.IGNORECASE)
                        position = parts[0].strip()
                        company = parts[1].strip() if len(parts) > 1 else None

                    elif re.search(r'[–—-]', prev):
                        # Format: "Senior Engineer – Google"
                        parts = re.split(r'\s+[–—-]\s+', prev)
                        position = parts[0].strip()
                        company = parts[1].strip() if len(parts) > 1 else None

                    elif ',' in prev:
                        # Format: "Senior Engineer, Google"
                        parts = [p.strip() for p in prev.split(',')]
                        position = parts[0]
                        company = parts[1] if len(parts) > 1 else None

                    else:
                        # Whole previous line is assumed to be the company name
                        company = prev
            else:
                # ── Role/company info is on the same line as the date ──
                if re.search(r'\bat\b', role_info, re.IGNORECASE):
                    # Format: "Senior Engineer at Google – Mumbai  Jan 2020 – Present"
                    parts = re.split(r'\s+at\s+', role_info, flags=re.IGNORECASE)
                    position = parts[0].strip()
                    company_location = parts[1].strip() if len(parts) > 1 else None

                    # Strip location from company if separated by a dash
                    if company_location and re.search(r'[–—-]', company_location):
                        company_parts = re.split(r'\s+[–—-]\s+', company_location)
                        company = company_parts[0].strip() if company_parts else None
                    else:
                        company = company_location

                elif ',' in role_info:
                    # Format: "Senior Engineer, Google – Mumbai  Jan 2020 – Present"
                    parts = [p.strip() for p in role_info.split(',')]
                    position = parts[0].strip() if parts else None
                    if len(parts) > 1:
                        company_part = parts[1]
                        # Strip location from company if separated by a dash
                        if re.search(r'[–—-]', company_part):
                            company_parts = re.split(r'\s+[–—-]\s+', company_part)
                            company = company_parts[0].strip() if company_parts else None
                        else:
                            company = company_part

                elif re.search(r'[–—-]', role_info):
                    # Format: "Senior Engineer – Google"
                    parts = re.split(r'\s+[–—-]\s+', role_info)
                    position = parts[0].strip()
                    company = parts[1].strip() if len(parts) > 1 else None

                else:
                    # Only a position title found, no company separator
                    position = role_info

            start_year = _parse_year(start_token)
            end_year = _parse_year(end_token)

            exp_list.append({
                "company": company,
                "position": position,
                "start_date": start_token,
                "end_date": end_token,
                "start_year": start_year,
                "end_year": end_year
            })

            # Jump past the matched line to avoid re-processing it
            i = matched_line_index + 1
        else:
            i += 1

    return exp_list


def compute_total_experience_years(exp_list):
    """
    Calculates total years of experience by summing the duration of each
    job entry in the parsed experience list.

    If end_year is None (i.e. the role is current/present), the current
    calendar year is used as the end date.

    Skips any entry where start_year is missing.
    Uses max(0, ...) to guard against malformed data where end < start.
    Returns an integer total (can be 0 if no valid entries).
    """
    total = 0
    for e in exp_list:
        s = e.get("start_year")
        en = e.get("end_year")
        if s is None:
            continue
        if en is None:
            en = datetime.datetime.now().year
        try:
            total += max(0, int(en) - int(s))
        except Exception:
            continue
    return total


# ════════════════════════════════════════════════
#  SKILLS EXTRACTION
# ════════════════════════════════════════════════

def extract_skills_section(text):
    """
    Isolates the raw skills section from the full resume text.
    Uses the same start/stop strategy as the other section extractors.

    Note: stop_headings deliberately excludes "experience" and "education"
    because those sections appear BEFORE skills in most resumes — including
    them as stop words would cause false early termination.
    """
    headings = [
        "skills",
        "technical skills",
        "competencies",
        "core competencies"
    ]

    # These come AFTER skills in a typical resume — used to detect section end
    stop_headings = [
        "projects",
        "certifications",
        "achievements",
        "awards",
        "publications",
        "volunteer",
        "additional information"
    ]

    start_idx = _find_section_header(text, headings)
    if start_idx is None:
        return None

    end_idx = len(text)
    for s in stop_headings:
        idx = _find_section_header(text, [s], start_idx)
        if idx is not None and idx < end_idx:
            end_idx = idx

    return text[start_idx:end_idx].strip()


def extract_skills(text):
    """
    Parses the skills section into a categorised dictionary.

    Expected format (handles multi-line values):
        Languages: Python, SQL, Bash
        Frameworks: Django, FastAPI,
                    Flask, Celery

    Returns a dict like:
        {
            "Languages": ["Python", "SQL", "Bash"],
            "Frameworks": ["Django", "FastAPI", "Flask", "Celery"]
        }

    Detection logic:
    - A line is treated as a NEW CATEGORY if it contains ':' AND
      starts with an uppercase letter (or leading space)
    - Lines without ':' that follow a category are treated as
      continuation lines (comma-separated skills that wrapped)

    Limitation: flat skill lists (no "Category:" prefix) return {}
    This is a known gap — a fallback for flat lists should be added.
    """
    section = extract_skills_section(text)
    if not section:
        return {}

    lines = [line.strip() for line in section.split("\n") if line.strip()]

    skills_dict = {}
    current_category = None
    current_skills_text = ""

    for line in lines:
        # Detect a category header line: starts uppercase and contains ':'
        if ":" in line and (line[0].isupper() or line.startswith(" ")):

            # Flush the previous category before starting a new one
            if current_category and current_skills_text:
                skill_list = [s.strip() for s in current_skills_text.split(",") if s.strip()]
                skills_dict[current_category] = skill_list

            # Split on first ':' only — skill values may also contain colons
            parts = line.split(":", 1)
            current_category = parts[0].strip()
            current_skills_text = parts[1].strip() if len(parts) > 1 else ""

        elif current_category:
            # Continuation line: append to the current category's skill text
            if current_skills_text:
                current_skills_text += " " + line
            else:
                current_skills_text = line

    # Flush the last category after the loop ends
    if current_category and current_skills_text:
        skill_list = [s.strip() for s in current_skills_text.split(",") if s.strip()]
        skills_dict[current_category] = skill_list

    return skills_dict


# ════════════════════════════════════════════════
#  MASTER FUNCTION
# ════════════════════════════════════════════════

def parse_basic_details(text):
    """
    Orchestrates all individual extractors and returns a single unified
    candidate profile dictionary.

    Called by the FastAPI /analyze endpoint after text extraction.

    Note: total_experience here uses the regex-based extractor.
    The API layer replaces it with compute_total_experience_years()
    (calculated from date ranges) if this returns None or 0.

    Returns:
        {
            name, email, phone, location,
            total_experience,
            education: [...],
            experience: [...],
            skills: {...}
        }
    """
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(text),
        "total_experience": extract_total_experience(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "skills": extract_skills(text)
    }