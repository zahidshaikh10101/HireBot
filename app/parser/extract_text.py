import pdfplumber
import docx


def extract_pdf_text(file_path):
    """
    Extracts all text from a PDF file page by page.
    Uses tight x/y tolerances to preserve spacing between characters
    and words that are close together in the PDF layout.
    """
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # x_tolerance=2, y_tolerance=2 keeps characters that are
            # slightly misaligned (common in scanned or exported PDFs)
            # grouped into the same word/line
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)

            # Only append if the page actually contains text
            # (some pages may be images or blank)
            if page_text:
                text += page_text + "\n"

    return text


def extract_docx_text(file_path):
    """
    Extracts all text from a DOCX file by iterating over paragraphs.
    Note: this captures body text only — text inside tables, headers,
    footers, or text boxes is NOT extracted here.
    """
    text = ""

    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


def extract_resume_text(file_path):
    """
    Router function — detects the file format from its extension
    and delegates to the appropriate extractor.

    Supported formats: .pdf, .docx
    Raises ValueError for anything else.
    """
    if file_path.endswith(".pdf"):
        return extract_pdf_text(file_path)

    elif file_path.endswith(".docx"):
        return extract_docx_text(file_path)

    else:
        raise ValueError("Unsupported file format")


def extract_pdf_links(file_path):
    """
    Extracts all hyperlink URIs embedded as annotations in a PDF.
    Iterates over every page's annotation list and pulls out
    any entry that contains a 'uri' key.

    Returns a deduplicated list — a link appearing on multiple
    pages is only included once.
    """
    links = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:

            # page.annots is None if the page has no annotations at all
            if page.annots:
                for annot in page.annots:

                    # Only collect annotations that are actual hyperlinks
                    # (PDFs can have other annotation types: highlights, comments, etc.)
                    if annot.get("uri"):
                        links.append(annot["uri"])

    # Deduplicate — same URL can appear on multiple pages
    return list(set(links))


def extract_professional_links(links):
    """
    Classifies a raw list of URLs into professional profile categories:
    github, linkedin, and portfolio.

    Rules:
    - mailto: and tel: links are skipped (not web URLs)
    - LinkedIn: must contain 'linkedin.com/in/' (profile URLs only)
    - GitHub: must be a profile URL (single path segment after domain)
              repo URLs like github.com/user/repo are ignored
    - Portfolio: any URL that contains the word 'portfolio'

    Returns a dict with keys github, linkedin, portfolio.
    Values are None if no matching link was found.
    """
    data = {
        "github": None,
        "linkedin": None,
        "portfolio": None
    }

    for link in links:

        # Normalize: strip whitespace and lowercase for consistent matching
        link = link.strip().lower()

        # Skip non-web links — mailto: and tel: are not profile URLs
        if link.startswith("mailto:") or link.startswith("tel:"):
            continue

        # ── LinkedIn ──
        # Match profile URLs specifically (linkedin.com/in/username)
        # This excludes company pages (linkedin.com/company/...) etc.
        if "linkedin.com/in/" in link:
            data["linkedin"] = link

        # ── GitHub ──
        elif "github.com/" in link:
            # Strip the domain and split the remaining path into segments
            # e.g. "https://github.com/zahid" → ["zahid"]
            # e.g. "https://github.com/zahid/repo" → ["zahid", "repo"]
            path = link.replace("https://github.com/", "").split("/")

            # A profile URL has exactly one segment (just the username)
            # Repo URLs have two or more segments — skip those
            if len(path) == 1:
                data["github"] = link

        # ── Portfolio ──
        # Simple keyword match — any URL containing "portfolio" is treated
        # as a personal portfolio site
        elif "portfolio" in link:
            data["portfolio"] = link

    return data