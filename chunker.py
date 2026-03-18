# Section aware chunking with hierarchy, semantic boundaries, and labeling

import re
import tiktoken
import numpy as np
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, SKIP_SECTIONS, SEMANTIC_SIMILARITY_THRESHOLD
import logging

logger = logging.getLogger(__name__)

tokenizer = tiktoken.get_encoding("cl100k_base")

# Lazy load NLTK and embedder to avoid blocking imports
_nltk_ready = False
_embedder = None


def _init_nltk():
    """Initialize NLTK punkt tokenizer (lazy load)."""
    global _nltk_ready
    if not _nltk_ready:
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            _nltk_ready = True
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}. Falling back to basic tokenization.")


def _get_embedder():
    """Get or initialize the embedder (lazy load)."""
    global _embedder
    if _embedder is None:
        try:
            from embedder import Embedder
            _embedder = Embedder(EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            raise RuntimeError("Embedder initialization failed") from e
    return _embedder


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


SECTION_HEADER_RE = re.compile(
    r"^\s*((\d+(\.\d+)*))?\s*[A-Z][A-Za-z\s\-]{3,}$"
)


def should_skip_section(header: str) -> bool:
    """Check if section should be skipped (boilerplate, references, etc)."""
    return any(k in header.lower() for k in SKIP_SECTIONS)


def extract_section_info(line: str):
    """Extract section number and hierarchy level from header."""
    match = SECTION_HEADER_RE.match(line.strip())
    if not match:
        return None, None

    section_number = match.group(1)
    return section_number, line.strip()


def classify_chunk(text: str) -> str:
    """Classify chunk by content type."""
    text_lower = text.lower()

    if any(k in text_lower for k in ["requirement", "shall", "must"]):
        return "requirement"
    elif "table" in text_lower:
        return "table"
    elif "definition" in text_lower or "defined" in text_lower:
        return "definition"
    elif "procedure" in text_lower or "process" in text_lower:
        return "procedure"
    return "general"


def _tokenize_sentences(text: str) -> list:
    """Tokenize text into sentences with fallback."""
    try:
        _init_nltk()
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK tokenization failed: {e}. Using basic splitting.")
        # Fallback: split by common punctuation
        return re.split(r'[.!?]+', text)


def semantic_chunk(text: str, threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> list:
    """
    Chunk text by semantic similarity between sentences.
    Uses embeddings to find natural boundaries where meaning shifts.
    """
    sentences = _tokenize_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text]

    try:
        embedder = _get_embedder()
        chunks = []
        current_chunk = [sentences[0]]

        # Embed sentences in batch for efficiency
        embedded_sents = embedder.embed(sentences)

        for i in range(1, len(sentences)):
            emb1 = np.array(embedded_sents[i - 1])
            emb2 = np.array(embedded_sents[i])

            # Cosine similarity
            sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10
            )

            if sim > threshold:
                current_chunk.append(sentences[i])
            else:
                # Semantic boundary detected
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    except Exception as e:
        logger.error(f"Semantic chunking failed: {e}. Returning sentences as-is.")
        return sentences


def section_aware_chunk_text(
    text: str,
    target_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list:
    """
    Chunk text while preserving section hierarchies and semantic boundaries.
    
    Features:
    - Respects document structure (headers with levels)
    - Skips boilerplate sections
    - Uses semantic chunking to find natural boundaries
    - Maintains section context for each chunk
    
    Args:
        text: Document text to chunk
        target_tokens: Target size for each chunk
        overlap_tokens: Token overlap between chunks
        
    Returns:
        List of chunks with metadata (header, section_path, text)
    """
    try:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to parse text: {e}")
        return [{"text": text, "section_path": [], "header": None}]

    sections = []
    current_section = {"header": None, "content": [], "section_path": []}
    section_stack = []

    for line in lines:
        section_number, _ = extract_section_info(line)

        if section_number:
            if should_skip_section(line):
                logger.debug(f"Skipping section: {line}")
                current_section = {"header": None, "content": [], "section_path": []}
                continue

            if current_section["content"]:
                sections.append(current_section)

            # Track hierarchy level
            level = section_number.count(".")
            section_stack = section_stack[:level]
            section_stack.append(section_number)

            current_section = {
                "header": line,
                "section_path": section_stack.copy(),
                "content": []
            }
        else:
            current_section["content"].append(line)

    if current_section["content"]:
        sections.append(current_section)

    chunks = []

    for section in sections:
        try:
            section_text = "\n".join(section["content"])
            paras = semantic_chunk(section_text)
        except Exception as e:
            logger.warning(f"Semantic chunking failed for section {section['header']}: {e}")
            paras = section["content"]

        current_chunk = []
        current_tokens = 0

        for para in paras:
            para = para.strip()
            if not para:
                continue

            try:
                para_tokens = count_tokens(para)
            except Exception as e:
                logger.warning(f"Token counting failed: {e}. Using length estimate.")
                para_tokens = len(para) // 4  # Rough estimate

            if current_tokens + para_tokens > target_tokens and current_chunk:
                chunk_text = (
                    (section["header"] + "\n\n" if section["header"] else "")
                    + "\n\n".join(current_chunk)
                )

                chunks.append({
                    "text": chunk_text,
                    "section_path": section["section_path"],
                    "header": section["header"]
                })

                # Compute overlap
                overlap_chunk = []
                overlap_count = 0

                for p in reversed(current_chunk):
                    try:
                        p_tokens = count_tokens(p)
                    except:
                        p_tokens = len(p) // 4
                    
                    if overlap_count + p_tokens > overlap_tokens:
                        break
                    overlap_chunk.insert(0, p)
                    overlap_count += p_tokens

                current_chunk = overlap_chunk
                current_tokens = overlap_count

            current_chunk.append(para)
            current_tokens += para_tokens

        if current_chunk:
            chunk_text = (
                (section["header"] + "\n\n" if section["header"] else "")
                + "\n\n".join(current_chunk)
            )

            chunks.append({
                "text": chunk_text,
                "section_path": section["section_path"],
                "header": section["header"]
            })

    return chunks if chunks else [{"text": text, "section_path": [], "header": None}]


def build_sliding_window_chunks(chunks, window_size=2) -> list:
    """
    Create sliding-window chunks by merging adjacent atomic chunks.
    
    Example: window_size=2 merges chunk[0]+chunk[1], chunk[1]+chunk[2], etc.
    Provides context overlap for retrieval.
    
    Args:
        chunks: List of atomic chunks
        window_size: Number of adjacent chunks to merge
        
    Returns:
        List of sliding window chunks with window metadata
    """
    window_chunks = []

    if window_size < 2 or len(chunks) < window_size:
        return window_chunks

    for i in range(len(chunks) - window_size + 1):
        base = chunks[i]

        try:
            merged_text = "\n\n".join(
                chunks[j]["text"] for j in range(i, i + window_size)
            )

            window_chunks.append({
                "pdf_name": base.get("pdf_name", "unknown"),
                "document_number": base.get("document_number", "Unknown"),
                "revision": base.get("revision", "Unknown"),
                "document_title": base.get("document_title", ""),
                "page_number": base.get("page_number", 0),
                "chunk_index": f"{base.get('chunk_index', 0)}-w{window_size}",
                "text": merged_text,
                "section_path": base.get("section_path", []),
                "header": base.get("header"),
                "label": base.get("label", "general"),
                "chunk_type": "sliding_window",
                "source_chunks": list(range(i, i + window_size)),
            })
        except Exception as e:
            logger.warning(f"Failed to create window chunk at index {i}: {e}")
            continue

    return window_chunks


def chunk_documents(docs) -> list:
    """
    Chunk a list of documents while preserving metadata.
    
    Process:
    1. Extract text from each document
    2. Apply section-aware chunking with semantic boundaries
    3. Fall back to recursive splitting if main method fails
    4. Create atomic and sliding-window chunks
    5. Classify and label each chunk
    
    Args:
        docs: List of documents with keys: pdf_name, page_number, text
        
    Returns:
        List of all chunks (atomic + sliding window) with metadata
    """
    all_chunks = []

    for doc in docs:
        try:
            chunks = section_aware_chunk_text(
                text=doc["text"],
                target_tokens=CHUNK_SIZE,
                overlap_tokens=CHUNK_OVERLAP,
            )
        except Exception as e:
            logger.error(f"Section-aware chunking failed for {doc.get('pdf_name')}: {e}")
            chunks = []

        # Fallback: use recursive splitting
        if not chunks:
            try:
                logger.info(f"Using fallback recursiv splitting for {doc.get('pdf_name')}")
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", " ", ""],
                )

                raw_chunks = splitter.split_text(doc["text"])

                chunks = [
                    {"text": c, "section_path": [], "header": None}
                    for c in raw_chunks
                ]
            except Exception as e:
                logger.error(f"Fallback chunking also failed: {e}. Skipping document.")
                continue

        atomic_chunks = []

        for idx, chunk in enumerate(chunks):
            try:
                atomic_chunks.append(
                    {
                        "pdf_name": doc.get("pdf_name", "unknown"),
                        "document_number": doc.get("document_number", "Unknown"),
                        "revision": doc.get("revision", "Unknown"),
                        "document_title": doc.get("document_title", ""),
                        "page_number": doc.get("page_number", 0),
                        "chunk_index": idx,
                        "text": chunk["text"],
                        "section_path": chunk.get("section_path", []),
                        "header": chunk.get("header"),
                        "label": classify_chunk(chunk["text"]),
                        "chunk_type": "atomic",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create atomic chunk {idx}: {e}")
                continue

        # Add atomic chunks
        if atomic_chunks:
            all_chunks.extend(atomic_chunks)

            # Add sliding window chunks
            try:
                window_chunks = build_sliding_window_chunks(
                    atomic_chunks,
                    window_size=2,
                )
                all_chunks.extend(window_chunks)
            except Exception as e:
                logger.warning(f"Failed to create sliding window chunks: {e}")

    logger.info(f"Created {len(all_chunks)} total chunks from {len(docs)} documents")
    return all_chunks
