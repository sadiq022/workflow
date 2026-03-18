import os
import re
from sentence_transformers import SentenceTransformer
from config import *
from milvus_store import MilvusStore
from llm import call_llm


model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

def generate_search_queries(query: str):
    """
    Generate multiple retrieval-focused queries based on question type.
    No hallucination — just reformulations.
    """
    q = query.lower().strip()
    queries = [query]  # always include original

    if q.startswith("how"):
        queries.extend([
            f"{query} formulation",
            f"{query} derivation",
            f"{query} method",
        ])

    elif q.startswith("what"):
        queries.extend([
            f"definition of {query}",
            f"{query} explanation",
        ])

    elif q.startswith("why"):
        queries.extend([
            f"motivation for {query}",
            f"reason for {query}",
        ])

    else:
        queries.extend([
            f"{query} theory",
            f"{query} model",
        ])

    # remove duplicates
    return list(dict.fromkeys(queries))


# def build_context(hits, min_score=0.10, max_chunks=5):
#     context_chunks = []
#     references = []
#     scores = []

#     for hit in hits:
#         if hit.score < min_score:
#             continue

#         entity = hit.entity
#         text = hit.entity.get("text")
#         pdf = hit.entity.get("pdf_name")
#         page = hit.entity.get("page_number")

#         block = f"[Source: {pdf}, page {page}]\n{text}"
#         context_chunks.append(block)

#         references.append((pdf, page))
#         scores.append(hit.score)

#         if len(context_chunks) >= max_chunks:
#             break

#     context = "\n\n---\n\n".join(context_chunks)

#     return context, references, scores

def classify_section(text: str) -> str:
    t = text.lower()

    if "abstract" in t:
        return "abstract"
    if "introduction" in t:
        return "introduction"
    if any(k in t for k in ["method", "theory", "formulation", "derivation", "model development"]):
        return "method"
    if any(k in t for k in ["matrix", "stress", "strain", "constitutive", "="]):
        return "theory"
    if any(k in t for k in ["results", "validation", "comparison"]):
        return "results"
    if any(k in t for k in ["references", "bibliography"]):
        return "references"

    return "unknown"


def score_hit(hit, query_type: str):
    text = hit.entity["text"]
    section = classify_section(text)

    score = hit.score

    # Penalize weak sections
    if section == "abstract":
        score -= 0.25
    if section == "references":
        score -= 0.5

    # Boost explanatory sections
    if query_type == "how":
        if section in ("method", "theory"):
            score += 0.25
    elif query_type == "what":
        if section in ("introduction", "method"):
            score += 0.15
    elif query_type == "why":
        if section in ("theory", "results"):
            score += 0.2

    return score, section


def detect_query_type(query: str) -> str:
    q = query.lower()
    if q.startswith("how"):
        return "how"
    if q.startswith("why"):
        return "why"
    if q.startswith("what"):
        return "what"
    return "general"


def detect_document_number_query(query: str) -> str:
    """
    Detect if query mentions a specific document number (RRES, CSS, RRP, etc).
    Case-insensitive matching. Returns document number normalized to UPPERCASE.
    
    Examples:
        "RRES 90027 Issue D" -> "RRES 90027"
        "css 12" -> "CSS 12"
        "What is torque tightening" -> None
    """
    # Pattern: 2-5 letters + digits, optionally followed by Issue/Version/Rev
    # Case-insensitive match, normalize result to uppercase
    pattern = r'([A-Za-z]{2,5}\s+\d+[\w\-]*)'
    match = re.search(pattern, query, re.IGNORECASE)
    
    if match:
        return match.group(1).strip().upper()
    return None


def fuzzy_document_number_query(query: str):
    """
    Fuzzy match document numbers in query.
    Returns a list of possible document numbers (CSS/RRES/RRP + number) found in the query.
    Matches both 'CSS 200' and 'CSS200' patterns.
    """
    # Find all patterns like CSS 177, CSS177, RRES 90008, RRES90008, etc.
    pattern = r'((?:CSS|RRES|RRP)[\s-]?\d{2,6})'
    matches = re.findall(pattern, query, re.IGNORECASE)
    # Normalize: remove spaces and dashes, uppercase
    return [re.sub(r'[^A-Za-z0-9]', '', m).upper() for m in matches]


def build_context(
    hits,
    store,
    query,
    max_chunks=6,
    page_window=1,
    min_score=0.15,
):
    query_type = detect_query_type(query)

    scored_hits = []
    for hit in hits:
        adjusted_score, section = score_hit(hit, query_type)
        if adjusted_score >= min_score:
            scored_hits.append((hit, adjusted_score, section))

    scored_hits.sort(key=lambda x: x[1], reverse=True)

    context_blocks = []
    references = []
    used_keys = set()

    # 1️⃣ Anchor chunks
    anchors = scored_hits[:2]

    for hit, score, section in anchors:
        pdf = hit.entity["pdf_name"]
        page = hit.entity["page_number"]
        text = hit.entity["text"]

        key = (pdf, page, hit.entity["chunk_index"])
        if key in used_keys:
            continue

        context_blocks.append(
            f"[Source: {pdf}, page {page}, section={section}]\n{text}"
        )
        references.append((pdf, page))
        used_keys.add(key)

        # 2️⃣ Expand nearby pages
        neighbors = store.fetch_by_pdf_and_page_range(
            pdf_name=pdf,
            page=page,
            window=page_window,
        )

        for n in neighbors:
            nk = (pdf, n["page_number"], n["chunk_index"])
            if nk in used_keys:
                continue

            context_blocks.append(
                f"[Source: {pdf}, page {n['page_number']}]\n{n['text']}"
            )
            references.append((pdf, n["page_number"]))
            used_keys.add(nk)

            if len(context_blocks) >= max_chunks:
                break

    context = "\n\n---\n\n".join(context_blocks)

    return context, sorted(set(references)), [h[1] for h in scored_hits[:max_chunks]]


def rerank_hits_for_how_question(hits):
    scored = []

    for hit in hits:
        text = hit.entity["text"].lower()
        boost = 0.0

        if "derive" in text or "derivation" in text or "formulation" in text:
            boost += 0.15
        if "constitutive" in text:
            boost += 0.15
        if "mapping" in text or "stress" in text or "strain" in text:
            boost += 0.10
        if "abstract" in text:
            boost -= 0.20

        scored.append((hit, hit.score + boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [h for h, _ in scored]


def rag_search(query, mode, top_k=5):
    from config import MILVUS_URI
    store = MilvusStore(
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        dim=EMBEDDING_DIM,
    )

    # 🔑 Fuzzy document number matching
    fuzzy_doc_numbers = fuzzy_document_number_query(query)

    # Generate multiple retrieval queries
    search_queries = generate_search_queries(query)

    all_hits = []

    for q in search_queries:
        q_vector = model.encode(q, normalize_embeddings=True).tolist()
        if fuzzy_doc_numbers:
            # Fetch all chunks and filter in Python for robust fuzzy matching
            all_chunks = store.fetch_all_chunks()
            filtered_chunks = []
            for chunk in all_chunks:
                doc_num = chunk.get('document_number', '')
                doc_num_normalized = re.sub(r'[^A-Za-z0-9]', '', doc_num).upper()
                if any(fuzzy in doc_num_normalized for fuzzy in fuzzy_doc_numbers):
                    filtered_chunks.append(chunk)
            # If no matches, fallback to normal search
            if not filtered_chunks:
                results = store.search(q_vector, top_k)
                all_hits.extend(results[0] if results else [])
            else:
                # Build fake hit objects for reranking/context
                class FakeHit:
                    def __init__(self, entity):
                        self.entity = entity
                        self.score = 1.0  # max score for filtered
                all_hits.extend([FakeHit(chunk) for chunk in filtered_chunks])
        else:
            results = store.search(q_vector, top_k)
            all_hits.extend(results[0] if results else [])

    unique_hits = {}
    for h in all_hits:
        key = (
            h.entity["pdf_name"],
            h.entity["page_number"],
            h.entity["chunk_index"],
        )
        if key not in unique_hits:
            unique_hits[key] = h

    hits = list(unique_hits.values())

    hits = rerank_hits_for_how_question(hits)

    context, references, anchor_scores = build_context(
        hits=hits,
        store=store,
        query=query,
    )

    print("Context tokens:", len(context.split()))

    if not context.strip():
        return {
            "answer": "I cannot find the answer in the provided documents.",
            "confidence": 0.0,
            "references": [],
        }

    answer = call_llm(query, context, mode)

    print("Anchor scores:", anchor_scores)
    confidence = (
        sum(anchor_scores) / len(anchor_scores)
        if anchor_scores else 0.0
    )

    unique_refs = sorted(set(references))

    return {
        "answer": answer,
        "confidence": round(confidence, 3),
        "references": unique_refs,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    # result = rag_search(args.query)
    result = rag_search(query=args.query, mode=args.mode)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nConfidence:")
    print(result["confidence"])

    print("\nReferences:")
    for pdf, page in result["references"]:
        print(f"- {pdf}, page {page}")
