"""
Hybrid Retrieval Model
يدمج TF-IDF + BM25 + Embedding بطريقتين:
  - Serial   (تسلسلي): النتائج تمرّ من نموذج للثاني
  - Parallel (متوازي): كل نموذج يشتغل بشكل مستقل ثم ندمج النتائج بـ Fusion
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from services.retrieval_service.tfidf_model import TFIDFModel, get_tfidf_model
from services.retrieval_service.bm25_model import BM25Model, get_bm25_model
from services.retrieval_service.embedding_model import (
    EmbeddingModel,
    get_embedding_model,
)
import config


# =============================================
# دوال الـ Fusion (دمج النتائج)
# =============================================


def linear_combination_fusion(results_list: list, weights: list) -> list:
    """
    Weighted Linear Combination
    score_final = w1*score1 + w2*score2 + ...

    results_list: قائمة من قوائم (doc_id, score)
    weights: قائمة أوزان بنفس طول results_list
    """
    score_map = {}
    for results, weight in zip(results_list, weights):
        if not results:
            continue
        # تطبيع الدرجات بين 0 و 1
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

    طريقة ممتازة لأنها ما تحتاج تطبيع وبتشتغل مع أي عدد نماذج
    k=60 هو الافتراضي الأكثر استخداماً في الأبحاث
    """
    score_map = {}
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, start=1):
            score_map[doc_id] = score_map.get(doc_id, 0) + 1 / (k + rank)

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


# =============================================
# النموذج الهجين
# =============================================


class HybridModel:
    def __init__(self, dataset_key: str):
        print(f"[Hybrid] Loading models for {dataset_key}...")
        self.tfidf = get_tfidf_model(dataset_key)
        self.bm25 = get_bm25_model(dataset_key)
        self.emb = get_embedding_model(dataset_key)
        self.dataset_key = dataset_key
        print("[Hybrid] ✅ All sub-models are ready")

    # --------------------------------------------------
    # Parallel Hybrid (متوازي)
    # --------------------------------------------------
    def search_parallel(
        self,
        query: str,
        top_k: int = None,
        fusion_method: str = None,
        weights: list = None,
    ) -> list:
        """
        كل نموذج يشتغل باستقلالية ثم ندمج النتائج

        fusion_method: 'rrf' (Reciprocal Rank Fusion) أو 'linear'
        weights: [tfidf_w, bm25_w, emb_w] — تُستخدم فقط مع 'linear'
        """
        top_k = top_k or config.RETRIEVAL["top_k"]
        fusion_method = fusion_method or config.HYBRID["fusion_method"]
        weights = weights or [
            config.HYBRID["tfidf_weight"],
            config.HYBRID["bm25_weight"],
            config.HYBRID["embedding_weight"],
        ]

        # تشغيل النماذج الثلاثة
        r_tfidf = self.tfidf.search(query, top_k=top_k)
        r_bm25 = self.bm25.search(query, top_k=top_k)
        r_emb = self.emb.search(query, top_k=top_k)

        results_list = [r_tfidf, r_bm25, r_emb]

        if fusion_method == "rrf":
            fused = reciprocal_rank_fusion(results_list, k=config.HYBRID["rrf_k"])
        else:
            fused = linear_combination_fusion(results_list, weights)

        return fused[:top_k]

    # --------------------------------------------------
    # Serial Hybrid (تسلسلي)
    # --------------------------------------------------
    def search_serial(self, query: str, top_k: int = None, stage1_k: int = 500) -> list:
        """
        المرحلة 1: BM25 يسترجع stage1_k وثيقة مرشحة (سريع)
        المرحلة 2: Embedding يعيد ترتيب المرشحين (دقيق)

        هاد النهج بيوفر وقت كبير لأن Embedding بشتغل على عدد محدود من الوثائق
        """
        top_k = top_k or config.RETRIEVAL["top_k"]

        # المرحلة 1: استرجاع أولي بـ BM25
        candidates = self.bm25.search(query, top_k=stage1_k)
        candidate_ids = set(doc_id for doc_id, _ in candidates)

        if not candidates:
            return []

        # المرحلة 2: إعادة الترتيب بـ Embedding
        # نحسب embedding للاستعلام
        query_emb = self.emb.get_query_embedding(query)

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # نجيب embeddings الوثائق المرشحة فقط
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


# ===== للتجربة =====
if __name__ == "__main__":
    model = get_hybrid_model("dataset1")
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
