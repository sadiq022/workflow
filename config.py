import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "pdfs")
PDF_CACHE_DIR = os.path.join(BASE_DIR, "pdf_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MILVUS_DIR = os.path.join(BASE_DIR, "milvus_data")

COLLECTION_NAME = "pdf_chunks"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
