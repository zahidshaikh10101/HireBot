import re
import datetime

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


# ---------- EMAIL ----------
def extract_email(text):
    pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    
    match = re.search(pattern, text)
    
    return match.group() if match else None


# ---------- PHONE ----------
def extract_phone(text):
    pattern = r'(\+91[-\s]?)?[6-9]\d{9}'
    
    match = re.search(pattern, text)
    
    return match.group() if match else None


# ---------- NAME ----------
def extract_name(text):
    doc = nlp(text[:1000])   # first section usually contains name
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
            
    return None


# ---------- LOCATION ----------
def extract_location(text):
    doc = nlp(text[:1000])
    
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text
            
    return None

# ---------- TOTAL EXPERIENCE ----------
def extract_total_experience(text):

    pattern = r'(\d+)\+?\s+years?\s+of\s+experience'

    match = re.search(pattern, text.lower())

    if match:
        return int(match.group(1))

    return None

# ---------- EDUCATION ----------
def _find_section_header(text, headings, start_pos=0):
    """Find section header at start of a line (whole word only)."""
    lines = text[start_pos:].split('\n')
    cumulative_pos = start_pos
    
    for line in lines:
        lowered_line = line.lower().strip()
        for heading in headings:
            # Match heading as a complete word at start of line
            if re.match(r'^\s*' + re.escape(heading) + r'(?:\s|$|:)', lowered_line):
                # Return position after the line prefix (whitespace)
                return cumulative_pos + len(line) - len(line.lstrip())
        cumulative_pos += len(line) + 1  # +1 for newline
    
    return None

def extract_education_section(text):
    headings = [
        "education",
        "educational qualifications",
        "education & training",
        "academic qualifications",
        "academic",
        "education:" 
    ]

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

    end_idx = len(text)
    for s in stop_headings:
        idx = _find_section_header(text, [s], start_idx)
        if idx is not None and idx < end_idx:
            end_idx = idx

    return text[start_idx:end_idx].strip()


def extract_education(text):
    education_section = extract_education_section(text)
    if not education_section:
        return []

    lines = [l.strip() for l in education_section.split("\n") if l.strip()]

    education_list = []
    i = 0

    while i < len(lines):
        line = lines[i]

        range_match = re.search(r"(\d{4})\s*(?:[–—-]|to)\s*(\d{4}|present|current|now)", line, re.IGNORECASE)

        # If current line doesn't contain years, maybe next line does
        if not range_match and i + 1 < len(lines):
            range_match = re.search(r"(\d{4})\s*(?:[–—-]|to)\s*(\d{4}|present|current|now)", lines[i + 1], re.IGNORECASE)
            if range_match:
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

        if range_match:
            start_year = range_match.group(1)
            end_year = range_match.group(2).capitalize()

            # institution may be the part of the line before the year match
            institution = re.sub(range_match.re.pattern, "", line, flags=re.IGNORECASE).strip()
            if not institution:
                # maybe institution is previous line
                institution = lines[i - 1] if i - 1 >= 0 else None

            # degree often follows on the next line
            degree = lines[i + 1] if i + 1 < len(lines) and not re.search(r"\d{4}", lines[i + 1]) else None

            education_list.append({
                "institution": institution,
                "degree": degree,
                "start_year": start_year,
                "end_year": end_year
            })

            i += 2
        else:
            # fallback: lines without explicit years might still contain degree+institution
            # heuristically treat lines that contain common degree words as an education entry
            degree_keywords = ["bachelor", "master", "b.sc", "m.sc", "phd", "degree", "ba", "bs", "mba", "b.e", "m.e"]
            if any(k in line.lower() for k in degree_keywords):
                education_list.append({
                    "institution": None,
                    "degree": line,
                    "start_year": None,
                    "end_year": None
                })
            i += 1

    return education_list


# ---------- EXPERIENCE ----------
def extract_experience_section(text):
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
    token = token.strip().lower()
    if token in ("present", "current", "now"):
        return datetime.datetime.now().year
    try:
        return int(re.search(r"(\d{4})", token).group(1))
    except Exception:
        return None


def extract_experience(text):

    section = extract_experience_section(text)

    if not section:
        return []

    lines = [
        line.strip()
        for line in section.split("\n")
        if line.strip()
    ]

    exp_list = []

    date_patterns = [
        re.compile(r'([A-Za-z]{3,9}\s+\d{4})\s*[–—-]\s*([A-Za-z]{3,9}\s+\d{4}|present|current|now)', re.IGNORECASE),
        re.compile(r'(\d{4})\s*[–—-]\s*(\d{4}|present|current|now)', re.IGNORECASE),
        re.compile(r'([A-Za-z]{3,9}\s+\d{4})\s*to\s*([A-Za-z]{3,9}\s+\d{4}|present|current|now)', re.IGNORECASE),
        re.compile(r'(\d{4})\s*to\s*(\d{4}|present|current|now)', re.IGNORECASE),
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        match = None
        matched_line_index = i
        for pat in date_patterns:
            m = pat.search(line)
            if m:
                match = m
                break
        if not match and i + 1 < len(lines):
            # sometimes dates are on the next line
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

            # role/company info likely is the part before the date on the matched line
            role_info = matched_line[:match.start()].strip()

            position = None
            company = None

            if not role_info:
                # try previous lines for role/company
                if matched_line_index - 1 >= 0:
                    prev = lines[matched_line_index - 1]
                    if re.search(r'\bat\b', prev, re.IGNORECASE):
                        parts = re.split(r'\s+at\s+', prev, flags=re.IGNORECASE)
                        position = parts[0].strip()
                        company = parts[1].strip() if len(parts) > 1 else None
                    elif re.search(r'[–—-]', prev):
                        parts = re.split(r'\s+[–—-]\s+', prev)
                        position = parts[0].strip()
                        company = parts[1].strip() if len(parts) > 1 else None
                    elif ',' in prev:
                        parts = [p.strip() for p in prev.split(',')]
                        position = parts[0]
                        # company is typically before the location comma
                        company = parts[1] if len(parts) > 1 else None
                    else:
                        company = prev
            else:
                # parse role_info on the same line
                # Format: "Position, Company – Location" or "Position at Company – Location"
                if re.search(r'\bat\b', role_info, re.IGNORECASE):
                    parts = re.split(r'\s+at\s+', role_info, flags=re.IGNORECASE)
                    position = parts[0].strip()
                    company_location = parts[1].strip() if len(parts) > 1 else None
                    # Extract company before the dash
                    if company_location and re.search(r'[–—-]', company_location):
                        company_parts = re.split(r'\s+[–—-]\s+', company_location)
                        company = company_parts[0].strip() if company_parts else None
                    else:
                        company = company_location
                elif ',' in role_info:
                    # Split by comma: "Position, Company – Location" or "Position, Company" 
                    parts = [p.strip() for p in role_info.split(',')]
                    position = parts[0].strip() if parts else None
                    if len(parts) > 1:
                        # Company might still have location with dash
                        company_part = parts[1]
                        if re.search(r'[–—-]', company_part):
                            company_parts = re.split(r'\s+[–—-]\s+', company_part)
                            company = company_parts[0].strip() if company_parts else None
                        else:
                            company = company_part
                elif re.search(r'[–—-]', role_info):
                    parts = re.split(r'\s+[–—-]\s+', role_info)
                    position = parts[0].strip()
                    company = parts[1].strip() if len(parts) > 1 else None
                else:
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

            # advance past the matched line
            i = matched_line_index + 1
        else:
            i += 1

    return exp_list


def compute_total_experience_years(exp_list):
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


# ---------- SKILLS ----------
def extract_skills_section(text):
    headings = [
        "skills",
        "technical skills",
        "competencies",
        "core competencies"
    ]

    # Only use major section headers that won't conflict with skill categories
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
    section = extract_skills_section(text)
    if not section:
        return {}

    lines = [line.strip() for line in section.split("\n") if line.strip()]

    skills_dict = {}
    current_category = None
    current_skills_text = ""

    for line in lines:
        # Check if line contains a category (e.g., "Languages:", "Data & Analytics:")
        if ":" in line and (line[0].isupper() or line.startswith(" ")):
            # Save previous category if exists
            if current_category and current_skills_text:
                skill_list = [s.strip() for s in current_skills_text.split(",") if s.strip()]
                skills_dict[current_category] = skill_list

            # Parse new category
            parts = line.split(":", 1)
            current_category = parts[0].strip()
            current_skills_text = parts[1].strip() if len(parts) > 1 else ""
        elif current_category:
            # Continuation of previous category skills (multi-line)
            if current_skills_text:
                current_skills_text += " " + line
            else:
                current_skills_text = line

    # Don't forget the last category
    if current_category and current_skills_text:
        skill_list = [s.strip() for s in current_skills_text.split(",") if s.strip()]
        skills_dict[current_category] = skill_list

    return skills_dict

# ---------- MASTER FUNCTION ----------
def parse_basic_details(text):
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