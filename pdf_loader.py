import pdfplumber
import os
import re


def extract_pdf_metadata(pdf):
    """
    Extract metadata from PDF automatically.
    No hardcoding - just what's in the PDF.
    """
    metadata = {}
    
    # 1. Get PDF properties (title, author, subject, keywords)
    if pdf.metadata:
        metadata['title'] = pdf.metadata.get('Title', '').strip()
        metadata['author'] = pdf.metadata.get('Author', '').strip()
        metadata['subject'] = pdf.metadata.get('Subject', '').strip()
        metadata['keywords'] = pdf.metadata.get('Keywords', '').strip()
    
    # 2. Extract document structure from first page
    if pdf.pages:
        first_page_text = pdf.pages[0].extract_text()
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        metadata['first_line'] = lines[0] if lines else ''
        metadata['first_para'] = ' '.join(lines[:3]) if lines else ''
        
        # Extract header (top 10% of page)
        first_page = pdf.pages[0]
        if hasattr(first_page, 'height'):
            header_box = (0, 0, first_page.width, first_page.height * 0.15)
            header_text = first_page.crop(header_box).extract_text()
            metadata['page_header'] = header_text.strip() if header_text else ''
    
    # 3. Extract footer from last page (usually has version/issue info)
    if pdf.pages and len(pdf.pages) > 0:
        last_page = pdf.pages[-1]
        if hasattr(last_page, 'height'):
            footer_box = (0, last_page.height * 0.85, last_page.width, last_page.height)
            footer_text = last_page.crop(footer_box).extract_text()
            metadata['page_footer'] = footer_text.strip() if footer_text else ''
    
    # 4. Extract document classification info (usually in first few lines)
    if pdf.pages:
        all_first_page_lines = [l.strip() for l in first_page_text.split('\n')]
        # Look for document identifiers (e.g., CSS, RRES, RRP, etc.)
        metadata['document_id'] = ' '.join(all_first_page_lines[:5])
    
    # 5. Count pages for context
    metadata['total_pages'] = len(pdf.pages)
    
    return metadata


def extract_document_identifier(metadata: dict) -> str:
    """
    Extract a unique document identifier from metadata.
    No hardcoding - automatically finds patterns like "RRP 51011", "CSS 12", "RRES 90002"
    """
    # Try to get from first_line (usually the identifier)
    first_line = metadata.get('first_line', '')
    if first_line:
        # Match patterns like "CSS 12", "RRES 90002", "RRP 51011"
        match = re.match(r'([A-Z]{2,5}\s*\d+[\w\-]*)', first_line)
        if match:
            return match.group(1).strip()
    
    # Fallback: extract from page_header
    page_header = metadata.get('page_header', '')
    if page_header:
        match = re.match(r'([A-Z]{2,5}\s*\d+[\w\-]*)', page_header)
        if match:
            return match.group(1).strip()
    
    # Last resort: use first part of title
    title = metadata.get('title', 'UNKNOWN')
    if title and title != 'UNKNOWN':
        match = re.match(r'([A-Z]{2,5}\s*\d+[\w\-]*)', title)
        if match:
            return match.group(1).strip()
    
    return 'UNKNOWN'
    documents = []

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, file)
        with pdfplumber.open(path) as pdf:
            # Extract metadata once per PDF
            pdf_metadata = extract_pdf_metadata(pdf)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append({
                        "pdf_name": file,
                        "page_number": page_num,
                        "text": text,
                        "metadata": pdf_metadata  # Store metadata with each page
                    })

    return documents
