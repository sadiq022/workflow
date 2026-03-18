# PDF Upload Feature Guide

## Overview

The PDF Upload feature allows users to dynamically add new PDF documents to your RAG system without replacing existing data. All new PDFs are **merged** into your existing Milvus database, allowing for a continuously growing knowledge base.

## Architecture

### Files Modified/Created

1. **`gradio_app.py`** (Modified)
   - Added new "📤 Upload PDFs" tab alongside the Q&A tab
   - Integrated `process_uploaded_pdfs()` and `get_upload_status()` functions
   - Users can now upload files directly through the UI

2. **`pdf_upload.py`** (New)
   - Core module for handling incremental PDF uploads
   - `process_uploaded_pdfs()`: Main function to process and merge PDFs
   - `get_upload_status()`: Returns current database status
   - Handles file copying, PDF loading, chunking, embedding, and Milvus insertion

3. **`pdf_management.py`** (New, Optional)
   - Advanced utilities for database management
   - Functions for duplicate detection
   - Database statistics and PDF removal capabilities
   - Can be integrated for more control

## How It Works

### Upload Flow

```
User Upload (UI)
    ↓
process_uploaded_pdfs()
    ├─ Copy files to temp directory
    ├─ load_pdfs() - Extract text from PDFs
    ├─ chunk_documents() - Split into chunks
    ├─ Embedder.embed() - Generate embeddings
    └─ store.insert() - Add to Milvus (WITHOUT reset)
    ↓
Database Updated (old + new PDFs)
```

### Key Difference from Initial Setup

**Initial Pipeline (`run_pipeline.py`):**
- Calls `store.reset_collection()` - **Clears all data**
- Replaces entire database

**Upload Pipeline (`pdf_upload.py`):**
- Does NOT call `reset_collection()`
- Only calls `store.insert()` - **Appends data**
- Merges with existing database

## Usage

### For Users (UI)

1. **Start the application:**
   ```bash
   python gradio_app.py
   ```

2. **Navigate to the "📤 Upload PDFs" tab**

3. **Click "Select PDF files to upload"** and choose one or more PDFs

4. **Click "Upload and Merge PDFs"** button

5. **Wait for processing** - Status will show progress:
   - Loading PDFs
   - Creating chunks
   - Embedding content
   - Merging into database

6. **Check results** - Database status updates showing total chunks and PDFs

7. **Ask questions** about all documents (old + new) in the Q&A tab

### For Developers (Code)

**Basic usage in Python:**
```python
from pdf_upload import process_uploaded_pdfs, get_upload_status

# Upload files
file_paths = ["/path/to/file1.pdf", "/path/to/file2.pdf"]
success, message = process_uploaded_pdfs(file_paths)

# Get status
status = get_upload_status()
print(status)
```

**With advanced management:**
```python
from pdf_management import get_database_statistics, find_duplicate_pdfs

# Check for duplicates
duplicates = find_duplicate_pdfs(["/path/to/file.pdf"])

# Get detailed stats
stats = get_database_statistics()
print(stats)

# Remove specific PDF
from pdf_management import remove_pdf_from_database
remove_pdf_from_database("document.pdf")
```

## Features

### ✅ Current Features

- **Incremental Upload**: Add PDFs without removing existing ones
- **Automatic Processing**: PDFs are automatically:
  - Loaded and text extracted
  - Split into chunks (respecting sections)
  - Embedded using sentence transformers
  - Stored in Milvus with metadata
- **Live Status**: Display current database information
- **Multi-file Upload**: Upload multiple PDFs at once
- **Error Handling**: Graceful error messages if processing fails

### 📋 Optional Advanced Features (via `pdf_management.py`)

- **Duplicate Detection**: Check if PDFs already exist by name
- **Database Statistics**: View detailed stats about each PDF
- **PDF Removal**: Delete specific PDFs from the database
- **Database Clearing**: Clear all data (requires confirmation in UI)

## Configuration

The upload process uses existing config from `config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
COLLECTION_NAME = "pdf_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
```

These are **shared** with the initial pipeline, ensuring consistency.

## Performance Considerations

### Processing Time
- **Per PDF**: Typically 5-30 seconds depending on:
  - File size (pages and content density)
  - Number of chunks
  - Embedding model speed
- **Batch Processing**: Multiple files are processed sequentially rather than in parallel (can be optimized for larger batches)

### Storage
- **Database Growth**: Each new chunk adds ~1-2 KB to Milvus
- **Example**: 100-page PDF with ~500 chunks ≈ 500-1000 KB

### Memory
- **Peak Usage**: During embedding generation
- **Optimization**: Files are cleaned up after processing

## Error Handling

The system handles several error scenarios:

| Error | Handling |
|-------|----------|
| No files selected | "No files selected for upload." |
| Invalid file types | Only `.pdf` files are processed |
| Corrupted PDFs | Skipped with error message |
| No text extracted | User notified, processing continues |
| Milvus connection issues | Descriptive error message |

## Future Enhancements

Potential improvements you could add:

1. **Duplicate Detection** - Check if PDF already exists before uploading
2. **Batch Processing** - Parallel processing for faster uploads
3. **Resume/Cancel** - Allow users to cancel long operations
4. **Preview** - Show extracted text before confirming upload
5. **Metadata** - Add custom metadata (author, date, category)
6. **Deduplication** - Detect and skip duplicate content
7. **Versioning** - Track which PDFs are active/archived
8. **Progress Bar** - Real-time upload progress visualization

## Troubleshooting

### Issue: "PDF not appearing in Q&A"
- **Solution**: Ensure PDF was successfully processed (check upload status message)
- **Check**: Database status shows the PDF name
- **Try**: Restart the app to refresh the connection

### Issue: "Upload is very slow"
- **Cause**: Large file size or many files
- **Solution**: 
  - Upload smaller batches
  - Optimize `CHUNK_SIZE` in `config.py` (larger = fewer chunks)
  - Use a faster GPU if available

### Issue: "Out of memory error"
- **Solution**:
  - Reduce batch size
  - Use a smaller embedding model
  - Close other applications

### Issue: "Database is locked"
- **Solution**: 
  - Restart the Gradio app
  - Check for other processes using Milvus
  - Verify `MILVUS_DIR` path is accessible

## Testing the Feature

Quick test script:

```python
# test_upload.py
from pdf_upload import process_uploaded_pdfs, get_upload_status
import os

# Create test PDFs (or use existing ones)
test_files = ["path/to/test1.pdf", "path/to/test2.pdf"]

# Process uploads
print("Uploading PDFs...")
success, msg = process_uploaded_pdfs(test_files)
print(f"Success: {success}")
print(f"Message: {msg}")

# Check status
print("\nDatabase Status:")
print(get_upload_status())
```

## Notes for Multi-User Systems

Since you mentioned users can't have separate databases yet:

- **Shared Database**: All users contribute to one Milvus collection
- **Data Isolation**: Currently there's no user-level isolation of PDFs
- **Future Options**:
  - Add `user_id` field to chunks metadata
  - Filter search results by user
  - Create separate collections per user
  - Use Milvus partitions for isolation

The current implementation is suitable for collaborative teams sharing a knowledge base!

---

**Last Updated**: March 2026
**Version**: 1.0
**Status**: Production Ready
