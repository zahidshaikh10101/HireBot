from parser.extract_text import (
    extract_resume_text,
    extract_pdf_links,
    extract_professional_links
)
from parser.parse_resume import parse_basic_details
import json
import datetime


file_path = r"E:\Users\Zahid.Shaikh\Downloads\Docs\Zahid_Salim_Shaikh.pdf"


# STEP 1 → Resume Text
resume_text = extract_resume_text(file_path)


# STEP 2 → Basic Parsing
basic_details = parse_basic_details(resume_text)


# STEP 3 → Hyperlinks
links = extract_pdf_links(file_path)

professional_links = extract_professional_links(links)


# STEP 4 → Merge Everything
final_data = {
    **basic_details,
    **professional_links
}

print(
    json.dumps(
        final_data,
        indent=4
    )
)