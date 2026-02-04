import os
import shutil

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
    store = MilvusStore(
        uri=os.path.join(MILVUS_DIR, "milvus.db"),
        collection_name=COLLECTION_NAME,
        dim=EMBEDDING_DIM
    )

    store.reset_collection()
    store.insert(records)
    # store.create_index()

    print("Pipeline completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()

    run(args.input_dir)
