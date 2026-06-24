import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def _fetch_full_texts(doc_ids: list, dataset_key: str = "dataset2") -> dict:
    result = {doc_id: "" for doc_id in doc_ids}
    try:
        from pymongo import MongoClient
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client[config.MONGO_DB_NAME]
        collection = db[config.MONGO_COLLECTIONS[dataset_key]]
        cursor = collection.find(
            {"doc_id": {"$in": doc_ids}},
            {"doc_id": 1, "text": 1, "_id": 0}
        )
        for doc in cursor:
            result[doc["doc_id"]] = doc.get("text", "")
        client.close()
    except Exception as e:
        print(f"[SearchService] ⚠️ MongoDB fetch failed, falling back to JSON...")
        try:
            import json
            with open(config.DATASET2_DOCS, "r", encoding="utf-8") as f:
                data = json.load(f)
            for doc_id in doc_ids:
                if doc_id in data:
                    v = data[doc_id]
                    result[doc_id] = str(v.get("text", v) if isinstance(v, dict) else v)
        except Exception as e2:
            print(f"[SearchService] ⚠️ JSON fallback also failed: {e2}")
    return result

class SearchService:
    _models_cache = {}

    SUPPORTED_MODELS = ["tfidf", "bm25", "embedding", "hybrid_parallel", "hybrid_serial"]

    @classmethod
    def get_supported_models(cls) -> list:
        return cls.SUPPORTED_MODELS

    @classmethod
    def _get_model(cls, dataset_key: str, model_type: str):
        cache_key = f"{dataset_key}_{model_type}"

        if cache_key in cls._models_cache:
            return cls._models_cache[cache_key]

        print(f"[SearchService] Loading {model_type} for {dataset_key}...")

        if model_type == "tfidf":
            from services.retrieval_service.tfidf_model import get_tfidf_model
            model = get_tfidf_model(dataset_key)
        elif model_type == "bm25":
            from services.retrieval_service.bm25_model import get_bm25_model
            model = get_bm25_model(dataset_key)
        elif model_type == "embedding":
            from services.retrieval_service.embedding_model import get_embedding_model
            model = get_embedding_model(dataset_key)
        elif model_type in ("hybrid_parallel", "hybrid_serial"):
            from services.retrieval_service.hybrid_model import get_hybrid_model
            model = get_hybrid_model(dataset_key)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        cls._models_cache[cache_key] = model
        return model

    @classmethod
    def search(
        cls,
        query: str,
        dataset_key: str = "dataset2",
        model_type: str = "bm25",
        top_k: int = None,
        **kwargs,
    ) -> dict:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_type}. Supported: {cls.SUPPORTED_MODELS}"
            )

        top_k = top_k or config.RETRIEVAL["top_k"]
        model = cls._get_model(dataset_key, model_type)

        if model_type == "tfidf":
            raw_results = model.search(query, top_k=top_k)

        elif model_type == "bm25":
            k1 = kwargs.get("bm25_k1")
            b = kwargs.get("bm25_b")
            raw_results = model.search(query, top_k=top_k, k1=k1, b=b)

        elif model_type == "embedding":
            raw_results = model.search(query, top_k=top_k)

        elif model_type == "hybrid_parallel":
            fusion = kwargs.get("fusion_method", config.HYBRID["fusion_method"])
            weights = None
            if "tfidf_weight" in kwargs:
                weights = [
                    float(kwargs["tfidf_weight"]),
                    float(kwargs["bm25_weight"]),
                    float(kwargs["embedding_weight"]),
                ]
            raw_results = model.search_parallel(
                query, top_k=top_k, fusion_method=fusion, weights=weights
            )

        elif model_type == "hybrid_serial":
            raw_results = model.search_serial(query, top_k=top_k)

        return {
            "query": query,
            "model": model_type,
            "dataset": dataset_key,
            "raw_results": raw_results,
            "total_found": len(raw_results),
        }

    @classmethod
    def search_and_rank(
        cls,
        query: str,
        dataset_key: str = "dataset2",
        model_type: str = "bm25",
        top_k_display: int = None,
        **kwargs,
    ) -> dict:
        top_k_display = top_k_display or config.RETRIEVAL["top_k_display"]

        search_result = cls.search(
            query=query,
            dataset_key=dataset_key,
            model_type=model_type,
            top_k=top_k_display * 5,
            **kwargs,
        )

        from services.ranking_service.ranker import rank_and_enrich

        ranked = rank_and_enrich(
            results=search_result["raw_results"],
            dataset_key=dataset_key,
            top_k=top_k_display,
            query=query,
        )

        doc_ids = [str(r["doc_id"]) for r in ranked]
        full_texts = _fetch_full_texts(doc_ids, dataset_key)
        for r in ranked:
            full_text = full_texts.get(str(r["doc_id"]), "")
            r["full_text"] = full_text
            if not r.get("snippet") and full_text:
                r["snippet"] = (full_text[:300] + "...") if len(full_text) > 300 else full_text

        return {
            "query": query,
            "model": model_type,
            "dataset": dataset_key,
            "total_results": len(ranked),
            "results": ranked,
        }

    @classmethod
    def clear_cache(cls):
        cls._models_cache.clear()
        print("[SearchService] 🗑️ Models cache cleared")


def search(query: str, model_type: str = "bm25", **kwargs) -> dict:
    return SearchService.search_and_rank(query=query, model_type=model_type, **kwargs)


if __name__ == "__main__":
    result = SearchService.search_and_rank(
        query="machine learning",
        model_type="bm25",
        top_k_display=5,
    )
    print(f"Found {result['total_results']} results")
    for r in result["results"][:3]:
        print(f"  #{r['rank']} - {r['doc_id']} (score: {r['score']})")