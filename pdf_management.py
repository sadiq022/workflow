"""
Advanced PDF management utilities for detecting duplicates and managing the database.
Optional module for enhanced control over PDF uploads.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Set

from config import COLLECTION_NAME, MILVUS_DIR, EMBEDDING_DIM
from milvus_store import MilvusStore


def get_pdf_hash(file_path: str) -> str:
    """Generate a hash of the PDF file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_existing_pdfs() -> Set[str]:
    """
    Get all PDF names currently in the database.
    
    Returns:
        Set of PDF file names in the database
    """
    try:
        store = MilvusStore(
            uri=os.path.join(MILVUS_DIR, "milvus.db"),
            collection_name=COLLECTION_NAME,
            dim=EMBEDDING_DIM
        )
        
        all_chunks = store.fetch_all_chunks()
        pdf_names = set()
        for chunk in all_chunks:
            if "pdf_name" in chunk:
                pdf_names.add(chunk["pdf_name"])
        
        return pdf_names
    except Exception as e:
        print(f"Error getting existing PDFs: {e}")
        return set()


def find_duplicate_pdfs(file_paths: List[str]) -> Dict[str, str]:
    """
    Check if uploaded PDFs already exist in the database.
    
    Args:
        file_paths: List of file paths to check
        
    Returns:
        Dictionary mapping file names to "exists" or "new"
    """
    existing_pdfs = get_existing_pdfs()
    result = {}
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name in existing_pdfs:
            result[file_name] = "exists"
        else:
            result[file_name] = "new"
    
    return result


def get_database_statistics() -> Dict:
    """
    Get detailed statistics about the database.
    
    Returns:
        Dictionary with database statistics
    """
    try:
        store = MilvusStore(
            uri=os.path.join(MILVUS_DIR, "milvus.db"),
            collection_name=COLLECTION_NAME,
            dim=EMBEDDING_DIM
        )
        
        all_chunks = store.fetch_all_chunks()
        
        # Aggregate statistics
        pdf_stats = {}
        for chunk in all_chunks:
            pdf_name = chunk.get("pdf_name", "unknown")
            page_number = chunk.get("page_number", 0)
            
            if pdf_name not in pdf_stats:
                pdf_stats[pdf_name] = {
                    "chunks": 0,
                    "pages": set()
                }
            
            pdf_stats[pdf_name]["chunks"] += 1
            pdf_stats[pdf_name]["pages"].add(page_number)
        
        # Convert sets to counts
        for pdf_name in pdf_stats:
            pdf_stats[pdf_name]["pages"] = len(pdf_stats[pdf_name]["pages"])
        
        return {
            "total_chunks": len(all_chunks),
            "total_pdfs": len(pdf_stats),
            "pdf_details": pdf_stats
        }
    except Exception as e:
        print(f"Error getting database statistics: {e}")
        return {
            "total_chunks": 0,
            "total_pdfs": 0,
            "pdf_details": {}
        }


def remove_pdf_from_database(pdf_name: str) -> bool:
    """
    Remove all chunks associated with a specific PDF from the database.
    
    Args:
        pdf_name: Name of the PDF to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        store = MilvusStore(
            uri=os.path.join(MILVUS_DIR, "milvus.db"),
            collection_name=COLLECTION_NAME,
            dim=EMBEDDING_DIM
        )
        
        # Query all chunks for the PDF
        expr = f'pdf_name == "{pdf_name}"'
        chunks = store.collection.query(
            expr=expr,
            output_fields=["pdf_name"],
            limit=10000
        )
        
        if chunks:
            # Delete by expression
            store.collection.delete(expr)
            print(f"Deleted {len(chunks)} chunks from {pdf_name}")
            return True
        else:
            print(f"No chunks found for {pdf_name}")
            return False
            
    except Exception as e:
        print(f"Error removing PDF from database: {e}")
        return False


def clear_database() -> bool:
    """
    Clear all data from the database.
    WARNING: This will delete all PDFs and chunks.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        store = MilvusStore(
            uri=os.path.join(MILVUS_DIR, "milvus.db"),
            collection_name=COLLECTION_NAME,
            dim=EMBEDDING_DIM
        )
        
        store.reset_collection()
        print("Database cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False
