import os
from sentence_transformers import SentenceTransformer
from config import *
from milvus_store import MilvusStore
from llm import generate_search_queries
import torch

from rag_search import (
    rerank_hits_for_how_question,
    score_hit,
    detect_query_type,
)

model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def search(query, top_k=10):
    """
    Vector search + reranking (same logic as rag_search).
    Returns reranked hits with adjusted scores.
    """

    from config import MILVUS_URI
    store = MilvusStore(
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        dim=EMBEDDING_DIM,
    )

    # Encode query
    vector = model.encode(query, normalize_embeddings=True).tolist()

    # Initial vector search
    results = store.search(vector, top_k)
    hits = results[0]

    # Rerank (same as RAG)
    hits = rerank_hits_for_how_question(hits)

    # Attach adjusted scores for debugging
    query_type = detect_query_type(query)

    enriched_hits = []
    for h in hits:
        adj_score, section = score_hit(h, query_type)
        enriched_hits.append((h, adj_score, section))

    # Sort by adjusted score
    enriched_hits.sort(key=lambda x: x[1], reverse=True)

    return enriched_hits


# -------------------------
# CLI debug mode
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    for i, (hit, adj_score, section) in enumerate(results, 1):
        entity = hit.entity

        print(f"\nResult {i}")
        print("Raw score:", round(hit.score, 4))
        print("Adjusted score:", round(adj_score, 4))
        print("Section:", section)
        print("PDF:", entity.get("pdf_name"))
        print("Page:", entity.get("page_number"))
        print("Chunk index:", entity.get("chunk_index"))
        print("Text:")
        print(entity.get("text")[:800])
        print("-" * 80)
