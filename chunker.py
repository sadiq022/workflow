#Section aware chunking:

import re
import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


SECTION_HEADER_RE = re.compile(
    r"^\s*(\d+(\.\d+)*\s+)?[A-Z][A-Za-z\s\-]{3,}$"
)


def is_section_header(text: str) -> bool:
    return bool(SECTION_HEADER_RE.match(text.strip()))


def section_aware_chunk_text(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 100,
):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    sections = []
    current_section = {"header": None, "content": []}

    for line in lines:
        if is_section_header(line):
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"header": line, "content": []}
        else:
            current_section["content"].append(line)

    if current_section["content"]:
        sections.append(current_section)

    chunks = []

    for section in sections:
        paras = "\n".join(section["content"]).split("\n\n")
        current_chunk = []
        current_tokens = 0

        for para in paras:
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > target_tokens and current_chunk:
                chunk_text = (
                    (section["header"] + "\n\n" if section["header"] else "")
                    + "\n\n".join(current_chunk)
                )
                chunks.append(chunk_text)

                # overlap
                overlap_chunk = []
                overlap_count = 0
                for p in reversed(current_chunk):
                    p_tokens = count_tokens(p)
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
            chunks.append(chunk_text)

    return chunks


def build_sliding_window_chunks(chunks, window_size=2):
    """
    Create sliding-window chunks by merging adjacent chunks.
    window_size=2 means: chunk[i] + chunk[i+1]
    """

    window_chunks = []

    for i in range(len(chunks) - window_size + 1):
        base = chunks[i]

        merged_text = "\n\n".join(
            chunks[j]["text"] for j in range(i, i + window_size)
        )

        window_chunks.append({
            "pdf_name": base["pdf_name"],
            "page_number": base["page_number"],
            "chunk_index": f"{base['chunk_index']}-w{window_size}",
            "text": merged_text,
        })

    return window_chunks


def chunk_documents(docs):
    all_chunks = []

    for doc in docs:
        chunks = section_aware_chunk_text(
            text=doc["text"],
            target_tokens=CHUNK_SIZE,
            overlap_tokens=CHUNK_OVERLAP,
        )

        # 🔑 FALLBACK if section chunking fails
        if not chunks:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            )

            chunks = splitter.split_text(doc["text"])

        atomic_chunks = []
        for idx, chunk in enumerate(chunks):
            atomic_chunks.append(
                {
                    "pdf_name": doc["pdf_name"],
                    "page_number": doc["page_number"],
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

        all_chunks.extend(atomic_chunks)
        window_chunks = build_sliding_window_chunks(
            atomic_chunks,
            window_size=2,
        )

    return all_chunks
