import os
import shutil

# Set gRPC keepalive settings to prevent "too_many_pings" errors
os.environ['GRPC_KEEPALIVE_TIME_MS'] = '30000'        # Send pings every 30 seconds
os.environ['GRPC_KEEPALIVE_TIMEOUT_MS'] = '10000'     # Wait 10 seconds for response
os.environ['GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS'] = 'true'  # Allow pings without active calls
os.environ['GRPC_HTTP2_MIN_TIME_BETWEEN_PINGS_MS'] = '30000'  # Minimum 30s between pings

from config import *
from pdf_loader import load_pdfs
from chunker import chunk_documents
from embedder import Embedder
from milvus_store import MilvusStore
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def prepare_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MILVUS_DIR, exist_ok=True)


def run(input_pdf_dir):
    prepare_dirs()

    os.makedirs(PDF_CACHE_DIR, exist_ok=True)
    # Copy PDFs
    for f in os.listdir(PDF_CACHE_DIR):
        os.remove(os.path.join(PDF_CACHE_DIR, f))

    for f in os.listdir(input_pdf_dir):
        if f.lower().endswith(".pdf"):
            shutil.copy(
                os.path.join(input_pdf_dir, f),
                os.path.join(PDF_CACHE_DIR, f)
            )

    print("Loading PDFs")
    docs = load_pdfs(PDF_CACHE_DIR)

    print("Chunking")
    chunks = chunk_documents(docs)

    print("Embedding")
    embedder = Embedder(EMBEDDING_MODEL)
    embeddings = embedder.embed([c["text"] for c in chunks])

    records = []
    for chunk, vector in zip(chunks, embeddings):
        records.append({
            **chunk,
            "vector": vector
        })

    print("Storing in Milvus")
    from config import MILVUS_URI
    store = MilvusStore(
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        dim=EMBEDDING_DIM
    )

    store.reset_collection()
    store.insert(records)
    store.load_collection()  # Load collection for faster queries

    print("Pipeline completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()

    run(args.input_dir)
