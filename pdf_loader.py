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


def parse_filename_for_document_number(filename: str) -> dict:
    """
    Extract document number, revision, and title from filename.
    Gracefully handles all PDF types - returns 'Unknown' for unmatchable patterns.
    Handles filenames with spaces or plus signs (e.g., "RRES 90008" or "RRES+90008").
    All document numbers are normalized to UPPERCASE for consistent filtering.
    PRIORITIZES: CSS, RRES, RRP patterns (aerospace documents)
    """
    name_without_ext = filename.rsplit('.', 1)[0] if filename.endswith('.pdf') else filename
    
    # Replace plus signs with spaces for consistent parsing
    normalized_name = name_without_ext.replace('+', ' ')
    
    # Look specifically for CSS, RRES, RRP patterns first (most reliable)
    # Avoids matching random text like month names or other dates
    aerospace_pattern = r'\b((?:CSS|RRES|RRP)\s+\d+[\w\-]*)\b'
    doc_match = re.search(aerospace_pattern, normalized_name, re.IGNORECASE)
    
    if doc_match:
        document_number = doc_match.group(1).strip().upper()
        
        # Try to find revision (Issue, Version, Rev, etc.)
        rev_pattern = r'(Issue|Version|Rev|v)\s+([A-Za-z0-9\-]+)'
        rev_match = re.search(rev_pattern, normalized_name, re.IGNORECASE)
        revision = rev_match.group(0).strip() if rev_match else 'Unknown'
        
        # Extract title (text after document number and revision)
        title_pattern = f'{re.escape(document_number)}\s+(?:{re.escape(revision)})?\s*-?\s*(.*)'
        title_match = re.search(title_pattern, normalized_name, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match and title_match.group(1) else ''
        
        return {
            'document_number': document_number,
            'revision': revision,
            'document_title': title,
            'source': 'filename'
        }
    
    # No document number found - return defaults
    return {
        'document_number': 'Unknown',
        'revision': 'Unknown',
        'document_title': name_without_ext,
        'source': 'none'
    }


def extract_document_number_from_pdf_header(pdf_metadata: dict) -> dict:
    """
    Extract document number, revision, and title from PDF header/metadata.
    This is more reliable than filename-based extraction.
    
    Returns dict with keys: document_number, revision, document_title
    All document numbers are normalized to UPPERCASE for consistent filtering.
    """
    # This is the preferred extraction method - from actual PDF content
    
    # Try first line of PDF (usually contains document number)
    first_line = pdf_metadata.get('first_line', '').strip()
    
    # Try page header (top 15% of first page)
    page_header = pdf_metadata.get('page_header', '').strip()
    
    # Try document_id field (first 5 lines)
    document_id = pdf_metadata.get('document_id', '').strip()
    
    # Combine sources, prioritize first_line
    sources_to_check = [first_line, page_header, document_id]
    
    for source_text in sources_to_check:
        if not source_text:
            continue
        
        # Look specifically for CSS, RRES, RRP patterns first (most important for aerospace docs)
        # These are more explicit and avoid matching dates like "AUGUST 2023"
        aerospace_pattern = r'\b((?:CSS|RRES|RRP)\s+\d+[\w\-]*)\b'
        aerospace_match = re.search(aerospace_pattern, source_text, re.IGNORECASE)
        
        if aerospace_match:
            document_number = aerospace_match.group(1).strip().upper()
            
            # Try to find revision (Issue, Version, Rev, etc.)
            # Look in the same source text
            rev_pattern = r'(Issue|Version|Rev|v)\s+([A-Za-z0-9\-]+)'
            rev_match = re.search(rev_pattern, source_text, re.IGNORECASE)
            revision = rev_match.group(0).strip() if rev_match else 'Unknown'
            
            # Extract title (text after document number and revision)
            # Look for title after the document number in source text
            title_pattern = f'{re.escape(document_number)}[^-]*(?:{re.escape(revision)})?[^-]*-?\s*(.*)'
            title_match = re.search(title_pattern, source_text, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match and title_match.group(1) else ''
            
            return {
                'document_number': document_number,
                'revision': revision,
                'document_title': title,
                'source': 'pdf_header'
            }
    
    # No document number found in PDF header - return defaults
    return {
        'document_number': 'Unknown',
        'revision': 'Unknown',
        'document_title': '',
        'source': 'none'
    }


def load_pdfs(pdf_dir):
    """
    Load all PDFs from a directory and extract text with metadata.
    
    Args:
        pdf_dir: Directory path containing PDF files
        
    Returns:
        List of documents with keys: pdf_name, page_number, text, metadata
    """
    documents = []

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, file)
        try:
            with pdfplumber.open(path) as pdf:
                # Extract metadata once per PDF
                pdf_metadata = extract_pdf_metadata(pdf)
                
                # 🔑 TWO-STEP APPROACH: Filename first (more structured), then PDF header
                filename_info = parse_filename_for_document_number(file)
                
                # If filename has a clear CSS/RRES/RRP pattern, use it (it's more reliable)
                if filename_info['document_number'] != 'Unknown':
                    doc_info = filename_info
                    doc_info['source'] = 'filename'
                else:
                    # Try PDF header extraction as fallback
                    doc_info = extract_document_number_from_pdf_header(pdf_metadata)
                    
                    # If PDF header found something, enhance it with filename title if needed
                    if doc_info['source'] == 'pdf_header' and not doc_info['document_title']:
                        doc_info['document_title'] = filename_info['document_title']
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        documents.append({
                            "pdf_name": file,
                            "page_number": page_num,
                            "text": text,
                            "metadata": pdf_metadata,
                            "document_number": doc_info['document_number'],
                            "revision": doc_info['revision'],
                            "document_title": doc_info['document_title']
                        })
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
            continue

    return documents
