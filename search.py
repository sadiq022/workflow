# import os
# from sentence_transformers import SentenceTransformer
# from config import *
# from milvus_store import MilvusStore

# # Load embedding model once
# model = SentenceTransformer(EMBEDDING_MODEL)


# def search(query, top_k=10):
#     """
#     Performs vector search and RETURNS hits
#     (does NOT do reranking or context building).
#     """

#     store = MilvusStore(
#         uri=os.path.join(MILVUS_DIR, "milvus.db"),
#         collection_name=COLLECTION_NAME,
#         dim=EMBEDDING_DIM,
#     )

#     # Encode query
#     vector = model.encode(query, normalize_embeddings=True).tolist()

#     # Milvus vector search
#     results = store.search(vector, top_k)

#     # Return raw hits for RAG pipeline
#     return results[0]


# # -------------------------
# # CLI debug mode
# # -------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", required=True)
#     parser.add_argument("--top_k", type=int, default=5)
#     args = parser.parse_args()

#     hits = search(args.query, args.top_k)

#     for i, hit in enumerate(hits, 1):
#         entity = hit.entity

#         print(f"\nResult {i}")
#         print("Score:", round(hit.score, 4))
#         print("PDF:", entity.get("pdf_name"))
#         print("Page:", entity.get("page_number"))
#         print("Chunk index:", entity.get("chunk_index"))
#         print("Text:")
#         print(entity.get("text")[:1000])  # safety truncate
#         print("-" * 80)

# exit()

#Working one!
import os
from sentence_transformers import SentenceTransformer
from config import *
from milvus_store import MilvusStore
from llm import generate_search_queries

from rag_search import (
    rerank_hits_for_how_question,
    score_hit,
    detect_query_type,
)

model = SentenceTransformer(EMBEDDING_MODEL)


def search(query, top_k=10):
    """
    Vector search + reranking (same logic as rag_search).
    Returns reranked hits with adjusted scores.
    """

    store = MilvusStore(
        uri=os.path.join(MILVUS_DIR, "milvus.db"),
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

# def search(query, top_k=5):
#     """
#     Multi-query vector retrieval.
#     Returns merged Milvus hits (no score cheating).
#     """

#     store = MilvusStore(
#         uri=os.path.join(MILVUS_DIR, "milvus.db"),
#         collection_name=COLLECTION_NAME,
#         dim=EMBEDDING_DIM,
#     )

#     # 1️⃣ Generate expanded queries
#     expanded_queries = [query] + generate_search_queries(query)

#     all_hits = {}

#     # 2️⃣ Vector search for each query
#     for q in expanded_queries:
#         vector = model.encode(q, normalize_embeddings=True).tolist()
#         results = store.search(vector, top_k)[0]

#         for hit in results:
#             key = (
#                 hit.entity["pdf_name"],
#                 hit.entity["page_number"],
#                 hit.entity["chunk_index"],
#             )
#             if key not in all_hits:
#                 all_hits[key] = hit

#     return list(all_hits.values())


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

##Working one ends here!
# import os
# import re
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# import json

# from config import *
# from milvus_store import MilvusStore

# # Load embedding model once
# model = SentenceTransformer(EMBEDDING_MODEL)


# # -------------------------
# # Helpers
# # -------------------------

# def tokenize(text: str):
#     return re.findall(r"\w+", text.lower())


# def load_bm25_corpus_from_milvus(store):
#     """
#     Load all texts directly from Milvus for BM25.
#     """
#     results = store.fetch_all_chunks()

#     corpus = []
#     meta = []

#     for r in results:
#         corpus.append(tokenize(r["text"]))
#         meta.append(r)

#     return BM25Okapi(corpus), meta


# # -------------------------
# # Hybrid search
# # -------------------------

# def search(query, top_k=10):
#     """
#     Hybrid retrieval:
#     - Vector search (Milvus)
#     - BM25 keyword search
#     - Union merge (no score cheating)
#     """

#     store = MilvusStore(
#         uri=os.path.join(MILVUS_DIR, "milvus.db"),
#         collection_name=COLLECTION_NAME,
#         dim=EMBEDDING_DIM,
#     )

#     # --------------------
#     # 1️⃣ Vector search (Milvus)
#     # --------------------
#     query_vector = model.encode(query, normalize_embeddings=True).tolist()
#     vector_results = store.search(query_vector, top_k)
#     vector_hits = vector_results[0]

#     # --------------------
#     # 2️⃣ BM25 search (from Milvus directly)
#     # --------------------
#     bm25, meta = load_bm25_corpus_from_milvus(store)
#     scores = bm25.get_scores(tokenize(query))

#     top_idx = sorted(
#         range(len(scores)),
#         key=lambda i: scores[i],
#         reverse=True
#     )[:top_k]

#     bm25_hits = [
#         {"score": scores[i], "entity": meta[i]}
#         for i in top_idx if scores[i] > 0
#     ]

#     # --------------------
#     # 3️⃣ Merge results (union of BM25 and vector search)
#     # --------------------
#     merged = {}

#     # Merge vector hits
#     for h in vector_hits:
#         key = (
#             h.entity["pdf_name"],
#             h.entity["page_number"],
#             h.entity["chunk_index"],
#         )
#         merged[key] = h

#     # Merge BM25 hits
#     for h in bm25_hits:
#         key = (
#             h["entity"]["pdf_name"],
#             h["entity"]["page_number"],
#             h["entity"]["chunk_index"],
#         )
#         if key not in merged:
#             merged[key] = h

#     return list(merged.values())


# # -------------------------
# # CLI debug mode
# # -------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", required=True)
#     parser.add_argument("--top_k", type=int, default=5)
#     args = parser.parse_args()

#     hits = search(args.query, args.top_k)

#     if not hits:
#         print("No results found.")
#         exit()

#     for i, hit in enumerate(hits, 1):
#         entity = hit.entity

#         print(f"\nResult {i}")
#         print("Score:", round(hit.score, 4))
#         print("PDF:", entity.get("pdf_name"))
#         print("Page:", entity.get("page_number"))
#         print("Chunk index:", entity.get("chunk_index"))
#         print("Text:")
#         print(entity.get("text")[:800])  # Truncated to the first 800 characters
#         print("-" * 80)
