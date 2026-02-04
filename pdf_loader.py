import pdfplumber
import os

def load_pdfs(pdf_dir):
    documents = []

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, file)
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append({
                        "pdf_name": file,
                        "page_number": page_num,
                        "text": text
                    })

    return documents
