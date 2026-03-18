import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "pdfs")
PDF_CACHE_DIR = os.path.join(BASE_DIR, "pdf_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MILVUS_DIR = os.path.join(BASE_DIR, "milvus_data")

COLLECTION_NAME = "pdf_chunks"

# Milvus configuration
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Semantic chunking parameters
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # Threshold for keeping sentences together

# Sections to skip during chunking
SKIP_SECTIONS = ["references", "bibliography", "appendix", "table of contents", "index"]

