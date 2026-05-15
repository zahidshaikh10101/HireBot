import pdfplumber
import docx

def extract_pdf_text(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)

            if page_text:
                text += page_text + "\n"

    return text


def extract_docx_text(file_path):
    text = ""

    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
        
    return text

def extract_resume_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_pdf_text(file_path)

    elif file_path.endswith(".docx"):
        return extract_docx_text(file_path)

    else:
        raise ValueError("Unsupported file format")
    
def extract_pdf_links(file_path):
    links = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:

            if page.annots:
                for annot in page.annots:

                    if annot.get("uri"):
                        links.append(annot["uri"])

    return list(set(links))

def extract_professional_links(links):

    data = {
        "github": None,
        "linkedin": None,
        "portfolio": None
    }

    for link in links:

        link = link.strip().lower()

        # Ignore email / phone links
        if link.startswith("mailto:") or link.startswith("tel:"):
            continue


        # LINKEDIN
        if "linkedin.com/in/" in link:
            data["linkedin"] = link


        # GITHUB PROFILE ONLY
        elif "github.com/" in link:

            # Remove github domain part
            path = link.replace("https://github.com/", "").split("/")

            # Profile URLs have only username
            if len(path) == 1:
                data["github"] = link


        # PORTFOLIO
        elif "portfolio" in link:
            data["portfolio"] = link


    return data
