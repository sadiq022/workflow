"""
Handles incremental PDF uploads and merges them into the existing Milvus database.
Unlike run_pipeline.py which replaces all data, this appends new PDFs to existing ones.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    COLLECTION_NAME,
    MILVUS_DIR,
)
from pdf_loader import load_pdfs
from chunker import chunk_documents
from embedder import Embedder
from milvus_store import MilvusStore


def process_uploaded_pdfs(uploaded_files: List[str]) -> Tuple[bool, str]:
    """
    Process uploaded PDF files and merge them into the existing Milvus database.
    
    Args:
        uploaded_files: List of file paths to uploaded PDFs
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if not uploaded_files:
            return False, "No files selected for upload."
        
        # Create temporary directory for processing
        temp_upload_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
        os.makedirs(temp_upload_dir, exist_ok=True)
        
        # Copy uploaded files to temp directory
        valid_pdfs = []
        for file_path in uploaded_files:
            if isinstance(file_path, str) and file_path.lower().endswith(".pdf"):
                try:
                    # Handle both direct file paths and Gradio file objects
                    if os.path.isfile(file_path):
                        file_name = os.path.basename(file_path)
                        dest_path = os.path.join(temp_upload_dir, file_name)
                        shutil.copy(file_path, dest_path)
                        valid_pdfs.append(dest_path)
                except Exception as e:
                    print(f"Error copying file {file_path}: {e}")
                    continue
        
        if not valid_pdfs:
            return False, "No valid PDF files found in the upload."
        
        print(f"Processing {len(valid_pdfs)} PDFs...")
        
        # Load PDFs
        docs = load_pdfs(temp_upload_dir)
        if not docs:
            return False, "No text content extracted from PDFs."
        
        print(f"Loaded {len(docs)} pages from {len(valid_pdfs)} PDFs")
        
        # Chunk documents
        chunks = chunk_documents(docs)
        if not chunks:
            return False, "No chunks created from documents."
        
        print(f"Created {len(chunks)} chunks")
        
        # Embed chunks
        print("Embedding chunks...")
        embedder = Embedder(EMBEDDING_MODEL)
        embeddings = embedder.embed([c["text"] for c in chunks])
        
        # Prepare records
        records = []
        for chunk, vector in zip(chunks, embeddings):
            records.append({
                **chunk,
                "vector": vector
            })
        
        # Insert into Milvus WITHOUT resetting the database
        # This appends to existing data instead of replacing it
        print("Merging into database...")
        store = MilvusStore(
            uri=os.path.join(MILVUS_DIR, "milvus.db"),
            collection_name=COLLECTION_NAME,
            dim=EMBEDDING_DIM
        )
        
        store.insert(records)
        
        # Cleanup
        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        
        success_msg = (
            f"✅ Successfully added {len(valid_pdfs)} PDF(s) and "
            f"{len(chunks)} chunks to the database!"
        )
        print(success_msg)
        return True, success_msg
        
    except Exception as e:
        print(f"Error during PDF upload processing: {e}")
        return False, f"Error processing PDFs: {str(e)}"


def get_upload_status() -> str:
    """Get information about the current database status."""
    try:
        store = MilvusStore(
            uri=os.path.join(MILVUS_DIR, "milvus.db"),
            collection_name=COLLECTION_NAME,
            dim=EMBEDDING_DIM
        )
        
        all_chunks = store.fetch_all_chunks()
        if all_chunks:
            # Get unique PDFs
            pdf_names = set(chunk["pdf_name"] for chunk in all_chunks)
            return (
                f"📊 Database Status: {len(all_chunks)} chunks from {len(pdf_names)} PDF(s)"
            )
        else:
            return "📊 Database is empty. Please upload some PDFs."
    except Exception as e:
        return f"⚠️ Unable to retrieve database status: {str(e)}"
