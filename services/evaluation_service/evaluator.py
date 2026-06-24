"""
Evaluation Service - يستخدم كل الـ queries الموجودة في qrels
"""

import os
import json
import math
import sys
import hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

CACHE_DIR = os.path.join(os.path.dirname(__file__), "eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_key(dataset_key, model_name, use_refinement):
    key_str = f"{dataset_key}_{model_name}_{use_refinement}_ALL"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _load_from_cache(cache_key):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_to_cache(cache_key, data):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Cache] ⚠️ Failed to save: {e}")


def clear_cache():
    count = 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(CACHE_DIR, f))
            count += 1
    print(f"[Cache] 🗑️ Cleared {count} cached results")
    return count


def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / k if k > 0 else 0.0


def recall(retrieved, relevant):
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved if d in relevant) / len(relevant)


def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            hits += 1
            sum_precision += hits / i
    return sum_precision / len(relevant)


def dcg(retrieved, relevance_scores, k):
    score = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        rel = relevance_scores.get(d, 0)
        rel = max(rel, 0)  
        score += rel / math.log2(i + 1)
    return score


def ndcg(retrieved, relevance_scores, k=10):
    actual = dcg(retrieved, relevance_scores, k)
    positive_rels = {d: r for d, r in relevance_scores.items() if r > 0}
    ideal_ranking = sorted(positive_rels.keys(), key=lambda d: positive_rels[d], reverse=True)
    ideal = dcg(ideal_ranking, positive_rels, k)
    return actual / ideal if ideal > 0 else 0.0


def load_queries_safely(queries_path):
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries_map = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                q_id = str(v.get("query_id", k))
                text = str(v.get("text", v.get("query", "")))
            else:
                q_id = str(k)
                text = str(v)
            queries_map[q_id] = text
    elif isinstance(data, list):
        for i, q in enumerate(data):
            if isinstance(q, dict):
                q_id = str(q.get("query_id") or q.get("id") or f"q_{i}")
                text = str(q.get("text", q.get("query", "")))
                queries_map[q_id] = text
            elif isinstance(q, str):
                queries_map[f"q_{i}"] = q
    return queries_map


def _normalize_qrels(qrels_data):
    """يرجع: {query_id: {doc_id: relevance}} — فقط الـ relevance > 0"""
    normalized = {}
    if isinstance(qrels_data, list):
        for item in qrels_data:
            if isinstance(item, dict):
                q_id = str(item.get("query_id") or item.get("id") or item.get("q_id", ""))
                d_id = str(item.get("doc_id") or item.get("document_id") or item.get("d_id", ""))
                score = int(item.get("relevance") or item.get("score", 1))
                if q_id and d_id and score > 0:  
                    if q_id not in normalized:
                        normalized[q_id] = {}
                    normalized[q_id][d_id] = score
    elif isinstance(qrels_data, dict):
        for q_id, docs in qrels_data.items():
            q_id_str = str(q_id)
            if isinstance(docs, dict):
                positive = {str(k): int(v) for k, v in docs.items() if int(v) > 0}
                if positive:
                    normalized[q_id_str] = positive
            elif isinstance(docs, list):
                normalized[q_id_str] = {str(d_id): 1 for d_id in docs}
    return normalized


def evaluate_model(
    search_fn,
    dataset_key="dataset2",
    use_refinement=False,
    max_queries=None,  
    model_name=None,
    use_cache=True,
):
    if model_name is None:
        model_name = getattr(search_fn, "__qualname__", "unknown")

    cache_key = _get_cache_key(dataset_key, model_name, use_refinement)

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            print(f"[Cache] ⚡ HIT: {model_name} ({'with' if use_refinement else 'without'} refinement)")
            print(f"   ➜ Cached eval used {cached.get('num_queries_evaluated', '?')} queries")
            return cached

    print(f"[Cache] 💭 MISS: computing {model_name}...")

    queries_path = config.DATASET2_QUERIES
    qrels_path = config.DATASET2_QRELS

    queries_map = load_queries_safely(queries_path)
    with open(qrels_path, "r", encoding="utf-8") as f:
        raw_qrels = json.load(f)
    qrels = _normalize_qrels(raw_qrels)

    print(f"\n{'='*60}")
    print(f"📊 EVALUATION DETAILS")
    print(f"{'='*60}")
    print(f"   Dataset:              {dataset_key}")
    print(f"   Model:                {model_name}")
    print(f"   Refinement:           {use_refinement}")
    print(f"   Queries file:         {len(queries_map)} queries")
    print(f"   Qrels file:           {len(raw_qrels) if isinstance(raw_qrels, list) else 'dict'} entries")
    print(f"   Unique q_ids in qrels:{len(qrels)}")

    query_ids_to_eval = list(qrels.keys())
    if max_queries is not None and max_queries > 0:
        query_ids_to_eval = query_ids_to_eval[:max_queries]

    print(f"   Will evaluate:        {len(query_ids_to_eval)} queries")
    print(f"{'='*60}\n")

    if use_refinement:
        from services.query_refinement_service.refiner import refine_query

    ap_scores, recall_scores, p10_scores, ndcg_scores = [], [], [], []
    skipped_no_text = 0
    skipped_errors = 0

    for idx, query_id in enumerate(query_ids_to_eval, 1):
        query_text = queries_map.get(query_id)
        if not query_text:
            skipped_no_text += 1
            continue

        relevant_docs = qrels[query_id]
        relevant_set = set(relevant_docs.keys())

        if use_refinement:
            try:
                refined = refine_query(query_text)
                query_text = refined["refined_query"]
            except Exception:
                pass

        try:
            results = search_fn(query_text)
            retrieved_ids = [str(doc_id) for doc_id, _ in results]
        except Exception as e:
            print(f"⚠️ Error on query {query_id}: {e}")
            skipped_errors += 1
            continue

        ap_scores.append(average_precision(retrieved_ids, relevant_set))
        recall_scores.append(recall(retrieved_ids, relevant_set))
        p10_scores.append(precision_at_k(retrieved_ids, relevant_set, k=10))
        ndcg_scores.append(ndcg(retrieved_ids, relevant_docs, k=10))

        if idx % 10 == 0:
            print(f"   Progress: {idx}/{len(query_ids_to_eval)}")

    n = len(ap_scores)
    print(f"\n✅ Evaluated: {n}/{len(query_ids_to_eval)} queries")
    print(f"   Skipped (no text):   {skipped_no_text}")
    print(f"   Skipped (errors):    {skipped_errors}\n")

    if n == 0:
        result = {
            "dataset": dataset_key,
            "model": model_name,
            "use_refinement": use_refinement,
            "MAP": 0.0, "Recall": 0.0, "P@10": 0.0, "nDCG@10": 0.0,
            "map": 0.0, "recall": 0.0, "p10": 0.0, "ndcg": 0.0,
            "num_queries_evaluated": 0,
            "num_queries_in_qrels": len(qrels),
            "num_queries_in_file": len(queries_map),
            "note": "No queries evaluated",
        }
    else:
        map_val = round(sum(ap_scores) / n, 4)
        rec_val = round(sum(recall_scores) / n, 4)
        p10_val = round(sum(p10_scores) / n, 4)
        ndcg_val = round(sum(ndcg_scores) / n, 4)

        result = {
            "dataset": dataset_key,
            "model": model_name,
            "use_refinement": use_refinement,
            "num_queries_evaluated": n,
            "num_queries_in_qrels": len(qrels),
            "num_queries_in_file": len(queries_map),
            "num_skipped_no_text": skipped_no_text,
            "num_skipped_errors": skipped_errors,
            "MAP": map_val, "Recall": rec_val, "P@10": p10_val, "nDCG@10": ndcg_val,
            "map": map_val, "recall": rec_val, "p10": p10_val, "precision": p10_val,
            "ndcg": ndcg_val, "ndcg10": ndcg_val,
        }

    if use_cache:
        _save_to_cache(cache_key, result)
        print(f"[Cache] 💾 Saved: {model_name}")

    return result


def evaluate_all_models(dataset_key="dataset2", max_queries=None, use_cache=True):
    import time
    from services.retrieval_service.tfidf_model import get_tfidf_model
    from services.retrieval_service.bm25_model import get_bm25_model
    from services.retrieval_service.embedding_model import get_embedding_model
    from services.retrieval_service.hybrid_model import get_hybrid_model

    print(f"\n{'='*60}")
    print(f" Starting Full Evaluation on ALL qrels queries")
    print(f"   Cache: {'ON' if use_cache else 'OFF'}")
    print(f"{'='*60}\n")

    print("[Evaluator] 📦 Loading models...")
    t0 = time.time()
    tfidf = get_tfidf_model(dataset_key)
    bm25 = get_bm25_model(dataset_key)
    emb = get_embedding_model(dataset_key)
    hybrid = get_hybrid_model(dataset_key)
    print(f"[Evaluator] ✅ Models loaded in {time.time()-t0:.1f}s\n")

    models = {
        "TF-IDF": tfidf.search,
        "BM25": bm25.search,
        "Embedding": emb.search,
        "Hybrid_Parallel": hybrid.search_parallel,
        "Hybrid_Serial": hybrid.search_serial,
    }

    all_results = {}
    total_start = time.time()

    for i, (model_name, search_fn) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] 🔍 {model_name}")
        t1 = time.time()
        r_before = evaluate_model(
            search_fn, dataset_key, use_refinement=False,
            max_queries=max_queries, model_name=model_name, use_cache=use_cache
        )
        r_after = evaluate_model(
            search_fn, dataset_key, use_refinement=True,
            max_queries=max_queries, model_name=model_name, use_cache=use_cache
        )
        print(f"   ⏱️ {time.time()-t1:.1f}s")

        all_results[model_name] = {
            "before_refinement": r_before,
            "after_refinement": r_after,
        }

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ Done in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}\n")
    return all_results