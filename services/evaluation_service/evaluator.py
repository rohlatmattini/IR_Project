"""
Evaluation Service
Calculates standard evaluation metrics in IR:
- MAP  (Mean Average Precision)
- Recall
- Precision@10
- nDCG (Normalized Discounted Cumulative Gain)
"""

import os
import json
import math
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    P@k = Number of relevant documents in top k results / k
    """
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_retrieved / k if k > 0 else 0.0


def recall(retrieved: list, relevant: set) -> float:
    """
    Recall = Number of retrieved relevant documents / Total actual relevant documents
    """
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for doc_id in retrieved if doc_id in relevant)
    return retrieved_relevant / len(relevant)


def average_precision(retrieved: list, relevant: set) -> float:
    """
    AP = Average Precision at each relevant document in the results
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant)


def dcg(retrieved: list, relevance_scores: dict, k: int) -> float:
    """
    DCG@k = Σ rel_i / log2(i+1)
    """
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevance_scores.get(doc_id, 0)
        score += rel / math.log2(i + 1)
    return score


def ndcg(retrieved: list, relevance_scores: dict, k: int = 10) -> float:
    """
    nDCG@k = DCG@k / IDCG@k
    """
    actual_dcg = dcg(retrieved, relevance_scores, k)

    ideal_ranking = sorted(
        relevance_scores.keys(), key=lambda d: relevance_scores[d], reverse=True
    )
    ideal_dcg = dcg(ideal_ranking, relevance_scores, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def load_queries_safely(queries_path: str):
    """Safely loads queries from a JSON file handling multiple structures"""
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        queries = data.get("queries", data.get("data", []))
        if not queries:
            queries = [{"query_id": str(k), "text": str(v)} for k, v in data.items()]
            return queries
    elif isinstance(data, list):
        queries = data
    else:
        raise ValueError(f"Unexpected queries format: {type(data)}")

    normalized_queries = []
    for i, q in enumerate(queries):
        if isinstance(q, str):
            normalized_queries.append({"query_id": f"q_{i}", "text": q})
        elif isinstance(q, dict):
            q_id = str(q.get("query_id") or q.get("id") or f"q_{i}")
            text = str(q.get("text", q.get("query", "")))
            normalized_queries.append({"query_id": q_id, "text": text})

    return normalized_queries


def _normalize_qrels(qrels_data) -> dict:
    """
    Normalizes qrels reading and converts it into a Dictionary: { query_id: { doc_id: score } }
    """
    normalized = {}

    # 1. If qrels is a list of objects
    if isinstance(qrels_data, list):
        for item in qrels_data:
            if isinstance(item, dict):
                q_id = str(
                    item.get("query_id") or item.get("id") or item.get("q_id", "")
                )
                d_id = str(
                    item.get("doc_id")
                    or item.get("document_id")
                    or item.get("d_id", "")
                )
                score = int(item.get("relevance") or item.get("score", 1))

                if q_id and d_id:
                    if q_id not in normalized:
                        normalized[q_id] = {}
                    normalized[q_id][d_id] = score

    # 2. If qrels is a nested dictionary { query_id: [doc_ids] } or { query_id: { doc_id: score } }
    elif isinstance(qrels_data, dict):
        for q_id, docs in qrels_data.items():
            q_id_str = str(q_id)
            if isinstance(docs, dict):
                normalized[q_id_str] = {str(k): int(v) for k, v in docs.items()}
            elif isinstance(docs, list):
                normalized[q_id_str] = {str(d_id): 1 for d_id in docs}
            elif isinstance(docs, (str, int)):
                normalized[q_id_str] = {str(docs): 1}

    return normalized


def evaluate_model(
    search_fn, dataset_key: str, use_refinement: bool = False, max_queries: int = 20
) -> dict:
    """
    Evaluates a search model on an entire dataset
    """
    queries_path = (
        config.DATASET1_QUERIES
        if dataset_key == "dataset1"
        else config.DATASET2_QUERIES
    )
    qrels_path = (
        config.DATASET1_QRELS if dataset_key == "dataset1" else config.DATASET2_QRELS
    )

    queries = load_queries_safely(queries_path)

    with open(qrels_path, "r", encoding="utf-8") as f:
        raw_qrels = json.load(f)

    # Normalize qrels to a dict to handle potential list format issues
    qrels = _normalize_qrels(raw_qrels)

    if use_refinement:
        from services.query_refinement_service.refiner import refine_query

    ap_scores = []
    recall_scores = []
    p10_scores = []
    ndcg_scores = []

    queries_to_eval = queries[:max_queries]
    print(f"[Evaluator] Evaluating {len(queries_to_eval)} queries on {dataset_key}...")

    for q in queries_to_eval:
        query_id = q.get("query_id")
        query_text = q.get("text")

        if not query_text:
            continue

        relevant_docs = qrels.get(query_id, {})
        if not relevant_docs:
            continue

        relevant_set = set(relevant_docs.keys())

        if use_refinement:
            refined = refine_query(query_text)
            query_text = refined["refined_query"]

        try:
            results = search_fn(query_text)
            retrieved_ids = [str(doc_id) for doc_id, _ in results]
        except Exception as e:
            print(f"⚠️ Error while searching for query {query_id}: {e}")
            continue

        ap_scores.append(average_precision(retrieved_ids, relevant_set))
        recall_scores.append(recall(retrieved_ids, relevant_set))
        p10_scores.append(precision_at_k(retrieved_ids, relevant_set, k=10))
        ndcg_scores.append(ndcg(retrieved_ids, relevant_docs, k=10))

    n = len(ap_scores)
    if n == 0:
        return {
            "dataset": dataset_key,
            "MAP": 0.0,
            "Recall": 0.0,
            "P@10": 0.0,
            "nDCG@10": 0.0,
            "map": 0.0,
            "recall": 0.0,
            "p10": 0.0,
            "ndcg": 0.0,
            "num_queries_evaluated": 0,
            "note": "No intersection found between IDs",
        }

    map_val = round(sum(ap_scores) / n, 4)
    rec_val = round(sum(recall_scores) / n, 4)
    p10_val = round(sum(p10_scores) / n, 4)
    ndcg_val = round(sum(ndcg_scores) / n, 4)

    results_dict = {
        "dataset": dataset_key,
        "use_refinement": use_refinement,
        "num_queries_evaluated": n,
        "MAP": map_val,
        "Recall": rec_val,
        "P@10": p10_val,
        "nDCG@10": ndcg_val,
        "map": map_val,
        "recall": rec_val,
        "p10": p10_val,
        "precision": p10_val,
        "ndcg": ndcg_val,
        "ndcg10": ndcg_val,
    }

    return results_dict


def evaluate_all_models(dataset_key: str) -> dict:
    """
    Evaluates all models and returns a structure fully compatible with the table directly
    """
    from services.retrieval_service.tfidf_model import get_tfidf_model
    from services.retrieval_service.bm25_model import get_bm25_model
    from services.retrieval_service.embedding_model import get_embedding_model
    from services.retrieval_service.hybrid_model import get_hybrid_model

    models = {
        "TF-IDF": get_tfidf_model(dataset_key).search,
        "BM25": get_bm25_model(dataset_key).search,
        "Embedding": get_embedding_model(dataset_key).search,
        "Hybrid_Parallel": get_hybrid_model(dataset_key).search_parallel,
        "Hybrid_Serial": get_hybrid_model(dataset_key).search_serial,
    }

    all_results = {}
    for model_name, search_fn in models.items():
        print(f"[Evaluator] Running evaluation for model {model_name}...")
        r_before = evaluate_model(search_fn, dataset_key, use_refinement=False)
        r_after = evaluate_model(search_fn, dataset_key, use_refinement=True)

        all_results[model_name] = {
            "before_refinement": r_before,
            "after_refinement": r_after,
        }

    return all_results


if __name__ == "__main__":
    from services.retrieval_service.tfidf_model import get_tfidf_model

    model = get_tfidf_model("dataset1")
    results = evaluate_model(model.search, "dataset1", max_queries=10)
    print(results)
