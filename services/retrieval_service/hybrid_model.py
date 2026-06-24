import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from services.retrieval_service.tfidf_model import get_tfidf_model
from services.retrieval_service.bm25_model import get_bm25_model
from services.retrieval_service.embedding_model import get_embedding_model
import config


def linear_combination_fusion(results_list: list, weights: list) -> list:
    
    score_map = {}
    for results, weight in zip(results_list, weights):
        if not results:
            continue
        max_score = max(s for _, s in results) if results else 1
        max_score = max_score if max_score > 0 else 1
        for doc_id, score in results:
            normalized = score / max_score
            score_map[doc_id] = score_map.get(doc_id, 0) + weight * normalized

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion (RRF)
    score(d) = Σ 1 / (k + rank(d))
    """
    score_map = {}
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, start=1):
            score_map[doc_id] = score_map.get(doc_id, 0) + 1 / (k + rank)

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


class HybridModel:
   

    def __init__(self, dataset_key: str):
        print(f"[Hybrid] Loading models for {dataset_key}...")
        self.tfidf = get_tfidf_model(dataset_key)
        self.bm25 = get_bm25_model(dataset_key)
        self.emb = get_embedding_model(dataset_key)
        self.dataset_key = dataset_key
        print("[Hybrid] ✅ All sub-models ready (TF-IDF + BM25 + Embedding)")

    def search_parallel(
        self,
        query: str,
        top_k: int = None,
        fusion_method: str = None,
        weights: list = None,
    ) -> list:
        top_k = top_k or config.RETRIEVAL["top_k"]
        fusion_method = fusion_method or config.HYBRID["fusion_method"]

        if weights is None:
            tw = config.HYBRID["tfidf_weight"]
            bw = config.HYBRID["bm25_weight"]
            ew = config.HYBRID["embedding_weight"]
            weights = [tw, bw, ew]

        r_tfidf = self.tfidf.search(query, top_k=top_k)
        r_bm25 = self.bm25.search(query, top_k=top_k)
        r_emb = self.emb.search(query, top_k=top_k)

        results_list = [r_tfidf, r_bm25, r_emb]

        if fusion_method == "rrf":
            fused = reciprocal_rank_fusion(results_list, k=config.HYBRID["rrf_k"])
        else:
            fused = linear_combination_fusion(results_list, weights)

        return fused[:top_k]

    def search_serial(self, query: str, top_k: int = None, stage1_k: int = 500) -> list:
       
        top_k = top_k or config.RETRIEVAL["top_k"]

        candidates = self.bm25.search(query, top_k=stage1_k)
        candidate_ids = set(doc_id for doc_id, _ in candidates)

        if not candidates:
            return []

        query_emb = self.emb.get_query_embedding(query)

        from sklearn.metrics.pairwise import cosine_similarity

        candidate_indices = [
            i for i, doc_id in enumerate(self.emb.doc_ids) if doc_id in candidate_ids
        ]

        if not candidate_indices:
            return candidates[:top_k]

        candidate_embs = self.emb.doc_embeddings[candidate_indices]
        scores = cosine_similarity([query_emb], candidate_embs).flatten()

        results = [
            (self.emb.doc_ids[candidate_indices[i]], float(scores[i]))
            for i in range(len(candidate_indices))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def get_hybrid_model(dataset_key: str) -> HybridModel:
    return HybridModel(dataset_key)


if __name__ == "__main__":
    model = get_hybrid_model("dataset2")
    query = "natural language processing text classification"

    print("\n=== Parallel (RRF) ===")
    r1 = model.search_parallel(query, top_k=5, fusion_method="rrf")
    print(r1)

    print("\n=== Parallel (Linear) ===")
    r2 = model.search_parallel(query, top_k=5, fusion_method="linear")
    print(r2)

    print("\n=== Serial ===")
    r3 = model.search_serial(query, top_k=5)
    print(r3)