"""
API Gateway - Flask
نقطة الدخول الوحيدة لكل الخدمات
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

app = Flask(__name__)
CORS(app)

# Cache للنماذج بدون ما نعيد التحميل كل مرة
_models_cache = {}


def get_model(dataset_key: str, model_type: str):
    cache_key = f"{dataset_key}_{model_type}"
    if cache_key not in _models_cache:
        print(f"[API] Loading {model_type} for {dataset_key}...")
        if model_type == "tfidf":
            from services.retrieval_service.tfidf_model import get_tfidf_model
            _models_cache[cache_key] = get_tfidf_model(dataset_key)
        elif model_type == "bm25":
            from services.retrieval_service.bm25_model import get_bm25_model
            _models_cache[cache_key] = get_bm25_model(dataset_key)
        elif model_type == "embedding":
            from services.retrieval_service.embedding_model import get_embedding_model
            _models_cache[cache_key] = get_embedding_model(dataset_key)
        elif model_type in ("hybrid_parallel", "hybrid_serial"):
            from services.retrieval_service.hybrid_model import get_hybrid_model
            _models_cache[cache_key] = get_hybrid_model(dataset_key)
    return _models_cache[cache_key]


import json

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "search_history.json")


def _load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            return []
    return []


def _save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

def _normalize_for_history(query: str) -> str:
    """توحيد الاستعلام للمقارنة فقط (بدون التأثير على الشكل المحفوظ)"""
    return query.strip().lower()


def _append_history(query: str):
    """يحفظ الاستعلام بدون أي تكرار، حتى لو غير متتالي،
    وينقل الاستعلام المكرر ليصبح آخر عنصر (الأحدث)"""
    query = query.strip()
    if not query:
        return

    norm = _normalize_for_history(query)
    _search_history[:] = [
        q for q in _search_history if _normalize_for_history(q) != norm
    ]
    _search_history.append(query)
    _save_history(_search_history)


_search_history = _load_history()# ============================================
# Endpoints
# ============================================


@app.route("/")
def index():
    ui_folder = os.path.join(os.path.dirname(__file__), "..", "ui")
    return send_from_directory(ui_folder, "index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "").strip()
    dataset = data.get("dataset", "dataset1")
    model_type = data.get("model", "bm25")
    top_k = int(data.get("top_k", config.RETRIEVAL["top_k_display"]))
    use_refine = data.get("use_refinement", False)

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    refinement_info = None
    search_query = query
    if use_refine:
        from services.query_refinement_service.refiner import refine_query
        refined = refine_query(query, history=_search_history)
        search_query = refined["refined_query"]
        refinement_info = refined

    # حفظ الاستعلام بدون تكرار
    _append_history(query)

    model = get_model(dataset, model_type)

    if model_type == "tfidf":
        raw_results = model.search(search_query, top_k=top_k * 5)
    elif model_type == "bm25":
        k1 = float(data.get("bm25_k1", config.BM25_PARAMS["k1"]))
        b = float(data.get("bm25_b", config.BM25_PARAMS["b"]))
        raw_results = model.search(search_query, top_k=top_k * 5, k1=k1, b=b)
    elif model_type == "embedding":
        raw_results = model.search(search_query, top_k=top_k * 5)
    elif model_type == "hybrid_parallel":
        fusion = data.get("fusion_method", config.HYBRID["fusion_method"])
        raw_results = model.search_parallel(
            search_query, top_k=top_k * 5, fusion_method=fusion
        )
    elif model_type == "hybrid_serial":
        raw_results = model.search_serial(search_query, top_k=top_k * 5)
    else:
        return jsonify({"error": f"Unknown model type: {model_type}"}), 400

    from services.ranking_service.ranker import rank_and_enrich
    ranked = rank_and_enrich(raw_results, dataset, top_k=top_k, query=search_query)

    return jsonify(
        {
            "query": query,
            "refined_query": search_query if use_refine else None,
            "refinement_info": refinement_info,
            "dataset": dataset,
            "model": model_type,
            "total_results": len(ranked),
            "results": ranked,
        }
    )


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    dataset = data.get("dataset", "dataset1")
    model_type = data.get("model", "tfidf")
    max_q = int(data.get("max_queries", 50))
    use_refine = data.get("use_refinement", False)

    model = get_model(dataset, model_type)

    if model_type == "hybrid_parallel":
        search_fn = model.search_parallel
    elif model_type == "hybrid_serial":
        search_fn = model.search_serial
    else:
        search_fn = model.search

    from services.evaluation_service.evaluator import evaluate_model
    results = evaluate_model(
        search_fn, dataset, use_refinement=use_refine, max_queries=max_q
    )
    return jsonify(results)


@app.route("/api/evaluate/all", methods=["POST"])
def evaluate_all():
    data = request.json
    dataset = data.get("dataset", "dataset1")
    from services.evaluation_service.evaluator import evaluate_all_models
    results = evaluate_all_models(dataset)
    return jsonify(results)


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    return jsonify(
        {
            "datasets": [
                {"key": "dataset2", "name": config.DATASET_NAMES["dataset2"]},
            ],
            "models": config.MODELS_LIST,
        }
    )

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "IR System is running"})


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"history": _search_history[-20:]})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    _search_history.clear()
    _save_history(_search_history)
    return jsonify({"message": "History cleared successfully"})


@app.route("/api/rag", methods=["POST"])
def rag_search():
    """
    RAG: بيعمل بحث عادي + يولّد إجابة بالـ LLM
    """
    data = request.json
    query = data.get("query", "").strip()
    dataset = data.get("dataset", "dataset1")
    model_type = data.get("model", "bm25")

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    # خطوة 1: استرجاع الوثائق
    model = get_model(dataset, model_type)
    if model_type == "hybrid_parallel":
        raw_results = model.search_parallel(query, top_k=10)
    elif model_type == "hybrid_serial":
        raw_results = model.search_serial(query, top_k=10)
    else:
        raw_results = model.search(query, top_k=10)

    from services.ranking_service.ranker import rank_and_enrich
    ranked = rank_and_enrich(raw_results, dataset, top_k=5, query=query)

    # حفظ الاستعلام بدون تكرار (RAG لا يحفظ إذا search حفظ نفس الاستعلام)
    _append_history(query)

    # خطوة 2: توليد الإجابة
    from services.rag_service.rag import generate_answer
    rag_result = generate_answer(query, ranked)

    return jsonify(
        {
            "query": query,
            "dataset": dataset,
            "model": model_type,
            "retrieved_docs": ranked,
            "rag_answer": rag_result.get("answer"),
            "rag_success": rag_result.get("success"),
        }
    )


# ============================================
if __name__ == "__main__":
    print("🚀 Running IR System API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
