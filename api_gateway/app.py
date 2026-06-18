from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys, os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

app = Flask(__name__)
CORS(app)


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
    return query.strip().lower()


def _append_history(query: str):
    """يحفظ الاستعلام بدون أي تكرار"""
    query = query.strip()
    if not query:
        return
    norm = _normalize_for_history(query)
    _search_history[:] = [
        q for q in _search_history if _normalize_for_history(q) != norm
    ]
    _search_history.append(query)
    _save_history(_search_history)


_search_history = _load_history()


@app.route("/")
def index():
    ui_folder = os.path.join(os.path.dirname(__file__), "..", "ui")
    return send_from_directory(ui_folder, "index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "").strip()
    dataset = data.get("dataset", "dataset2")
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

    _append_history(query)

    try:
        from services.search_service.searcher import SearchService

        extra_params = {}
        if model_type == "bm25":
            extra_params["bm25_k1"] = float(data.get("bm25_k1", config.BM25_PARAMS["k1"]))
            extra_params["bm25_b"] = float(data.get("bm25_b", config.BM25_PARAMS["b"]))
        elif model_type == "hybrid_parallel":
            extra_params["fusion_method"] = data.get(
                "fusion_method", config.HYBRID["fusion_method"]
            )

        result = SearchService.search_and_rank(
            query=search_query,
            dataset_key=dataset,
            model_type=model_type,
            top_k_display=top_k,
            **extra_params,
        )

        result["refined_query"] = search_query if use_refine else None
        result["refinement_info"] = refinement_info
        result["query"] = query

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """تقييم نموذج واحد على كل qrels queries"""
    data = request.json
    dataset = data.get("dataset", "dataset2")
    model_type = data.get("model", "bm25")
    max_q = data.get("max_queries", None)
    if max_q is not None:
        try:
            max_q = int(max_q)
            if max_q <= 0:
                max_q = None
        except (ValueError, TypeError):
            max_q = None
    use_refine = data.get("use_refinement", False)
    force_recompute = data.get("force_recompute", False)

    from services.search_service.searcher import SearchService
    model = SearchService._get_model(dataset, model_type)

    if model_type == "hybrid_parallel":
        search_fn = model.search_parallel
    elif model_type == "hybrid_serial":
        search_fn = model.search_serial
    else:
        search_fn = model.search

    from services.evaluation_service.evaluator import evaluate_model
    results = evaluate_model(
        search_fn,
        dataset,
        use_refinement=use_refine,
        max_queries=max_q,
        model_name=model_type,
        use_cache=not force_recompute,
    )
    return jsonify(results)


@app.route("/api/evaluate/all", methods=["POST"])
def evaluate_all():
    """تقييم كل النماذج على كل qrels queries"""
    data = request.json
    dataset = data.get("dataset", "dataset2")
    max_queries = data.get("max_queries", None)
    if max_queries is not None:
        try:
            max_queries = int(max_queries)
            if max_queries <= 0:
                max_queries = None
        except (ValueError, TypeError):
            max_queries = None
    force_recompute = data.get("force_recompute", False)
    from services.evaluation_service.evaluator import evaluate_all_models
    results = evaluate_all_models(
        dataset, max_queries=max_queries, use_cache=not force_recompute
    )
    return jsonify(results)


@app.route("/api/evaluate/cache", methods=["DELETE"])
def clear_eval_cache():
    """مسح cache التقييم"""
    from services.evaluation_service.evaluator import clear_cache
    count = clear_cache()
    return jsonify({"message": f"Cleared {count} cached results"})


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    from services.search_service.searcher import SearchService
    return jsonify(
        {
            "datasets": [
                {"key": "dataset2", "name": config.DATASET_NAMES["dataset2"]},
            ],
            "models": SearchService.get_supported_models(),
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "IR System is running"})


# ضيف هاد الكود بعد route /api/health مباشرة

@app.route("/api/multilingual/translate", methods=["POST"])
def multilingual_translate():
    """endpoint مستقل لتجربة Multilingual لحالها فقط"""
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is empty"}), 400

    try:
        from services.query_refinement_service.refiner import detect_and_translate
        result = detect_and_translate(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


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
    data = request.json
    query = data.get("query", "").strip()
    dataset = data.get("dataset", "dataset2")
    model_type = data.get("model", "bm25")

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    try:
        from services.search_service.searcher import SearchService
        search_result = SearchService.search_and_rank(
            query=query,
            dataset_key=dataset,
            model_type=model_type,
            top_k_display=5,
        )
        ranked = search_result["results"]
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    _append_history(query)

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


if __name__ == "__main__":
    print("🚀 Running IR System API on http://localhost:5000")
    print("📊 Architecture: SOA with Search Service as Facade")
    app.run(debug=False, host="0.0.0.0", port=5000)