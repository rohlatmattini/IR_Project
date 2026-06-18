
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from pymongo import MongoClient

_mongo_client = None


def _get_collection(dataset_key: str):
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(config.MONGO_URI)
    db = _mongo_client[config.MONGO_DB_NAME]
    collection_name = config.MONGO_COLLECTIONS.get(
        dataset_key, f"documents_{dataset_key}"
    )
    return db[collection_name]


_docs_cache = {}


def _load_docs_from_json(dataset_key: str) -> dict:
    if dataset_key not in _docs_cache:
        docs_path = config.DATASET2_DOCS

        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        normalized_docs = {}

        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    doc_id = str(v.get("doc_id", k))
                    normalized_docs[doc_id] = v
                else:
                    normalized_docs[str(k)] = {"doc_id": str(k), "text": str(v)}

        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    doc_id = str(item.get("doc_id") or item.get("id") or idx)
                    item["doc_id"] = doc_id
                    normalized_docs[doc_id] = item
                elif isinstance(item, str):
                    normalized_docs[str(idx)] = {"doc_id": str(idx), "text": item}

        _docs_cache[dataset_key] = normalized_docs

    return _docs_cache[dataset_key]


def _make_snippet(text: str, length: int = 300) -> str:
    snippet = text[:length]
    last_period = snippet.rfind(".")
    if last_period > 100:
        snippet = snippet[: last_period + 1]
    else:
        snippet = snippet + "..."
    return snippet


def rank_and_enrich(
    results: list, dataset_key: str, top_k: int = None, query: str = ""
) -> list:
   
    top_k = top_k or config.RETRIEVAL["top_k_display"]
    top_results = results[:top_k]
    doc_ids = [str(doc_id) for doc_id, _ in top_results]

    docs_map = {}
    use_mongo = True

    try:
        collection = _get_collection(dataset_key)
        cursor = collection.find({"doc_id": {"$in": doc_ids}}, {"_id": 0})
        docs_map = {d["doc_id"]: d for d in cursor}
    except Exception as e:
        print(f"[Ranker] ⚠️ MongoDB not available: {e}")
        print(f"[Ranker] 📂 Falling back to local JSON file...")
        use_mongo = False

    if not use_mongo or not docs_map:
        try:
            local_docs = _load_docs_from_json(dataset_key)
            docs_map = {d_id: local_docs.get(d_id, {}) for d_id in doc_ids}
        except Exception as e:
            print(f"[Ranker] ❌ Failed to load from JSON too: {e}")

    ranked = []
    for rank, (doc_id, score) in enumerate(top_results, start=1):
        doc_id_str = str(doc_id)
        doc = docs_map.get(doc_id_str, {})
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)

        snippet = _make_snippet(text) if text else ""

        ranked.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": round(score, 4),
                "title": doc.get("title", f"Document {doc_id}")
                if isinstance(doc, dict)
                else f"Document {doc_id}",
                "snippet": snippet,
                "full_text": text,
            }
        )

    return ranked


def get_document_by_id(doc_id: str, dataset_key: str) -> dict:
    """يجيب الوثيقة كاملة من MongoDB أو JSON"""
    try:
        collection = _get_collection(dataset_key)
        doc = collection.find_one({"doc_id": str(doc_id)}, {"_id": 0})
        if doc:
            return doc
    except Exception:
        pass

    # Fallback
    try:
        local_docs = _load_docs_from_json(dataset_key)
        return local_docs.get(str(doc_id), None)
    except Exception:
        return None