import os
import sys
import json
import pickle
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from services.preprocessing_service.preprocessor import preprocess_text

CACHE_PATH = os.path.join(config.INDEX2_DIR, "lda_cache.pkl")


def load_documents() -> list:
    """نفس دالة تحميل الوثائق"""
    with open(config.DATASET2_DOCS, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    if isinstance(data, dict):
        for k, v in data.items():
            text = v.get("text", "") if isinstance(v, dict) else str(v)
            docs.append({"doc_id": str(k), "text": str(text)})
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                docs.append({
                    "doc_id": str(item.get("doc_id", i)),
                    "text": str(item.get("text", ""))
                })
    return docs


def build_count_matrix(docs: list, max_features: int = 5000):
    """CountVectorizer للـ LDA"""
    print(f"[TopicModeling] Building Count matrix for {len(docs)} docs...")
    texts = [preprocess_text(d["text"]) for d in docs]
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.90,
        stop_words='english'
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"[TopicModeling] Matrix shape: {matrix.shape}")
    return matrix, vectorizer


def compute_coherence(
    lda_model,
    feature_names: list,
    matrix,
    n_top_words: int = 10
) -> float:
    """
    Coherence Score مبسّط (PMI-based على الـ co-occurrence)
    """
    # حساب co-occurrence matrix
    doc_term = (matrix > 0).astype(int)
    n_docs = matrix.shape[0]
    
    total_coherence = 0.0
    n_topics = lda_model.n_components
    
    for topic_idx in range(n_topics):
        topic = lda_model.components_[topic_idx]
        top_word_indices = topic.argsort()[-n_top_words:][::-1]
        
        topic_coherence = 0.0
        count = 0
        
        for i in range(len(top_word_indices)):
            for j in range(i + 1, len(top_word_indices)):
                wi = top_word_indices[i]
                wj = top_word_indices[j]
                
                # عدد الوثائق التي تحتوي wi و wj معاً
                co_occur = float(
                    doc_term[:, wi].multiply(doc_term[:, wj]).sum()
                )
                # عدد الوثائق التي تحتوي wj
                wj_count = float(doc_term[:, wj].sum())
                
                if co_occur > 0 and wj_count > 0:
                    # UMass Coherence
                    topic_coherence += np.log(
                        (co_occur + 1) / wj_count
                    )
                    count += 1
        
        if count > 0:
            total_coherence += topic_coherence / count
    
    return round(total_coherence / n_topics, 4)


def get_topic_distribution(lda_model, matrix, doc_ids: list) -> list:
    """
    توزيع الـ topics على كل وثيقة
    """
    doc_topic_matrix = lda_model.transform(matrix)
    
    distributions = []
    for i, doc_id in enumerate(doc_ids):
        dominant_topic = int(np.argmax(doc_topic_matrix[i]))
        distributions.append({
            "doc_id": doc_id,
            "dominant_topic": dominant_topic,
            "topic_weights": [
                round(float(w), 4)
                for w in doc_topic_matrix[i]
            ]
        })
    
    return distributions, doc_topic_matrix


def run_lda(
    n_topics: int = 5,
    use_cache: bool = True,
    force_recompute: bool = False
) -> dict:
    """
    الدالة الرئيسية: LDA + تقييم شامل
    """
    cache_key = f"lda_{n_topics}"
    
    # تحقق من الـ cache
    if use_cache and not force_recompute and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                print(f"[TopicModeling] ⚡ Cache HIT (n_topics={n_topics})")
                return cached["data"]
        except Exception:
            pass

    docs = load_documents()
    doc_ids = [d["doc_id"] for d in docs]
    
    # بناء المصفوفة
    matrix, vectorizer = build_count_matrix(docs)
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    # تشغيل LDA
    print(f"[TopicModeling] Training LDA with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online',
        batch_size=256,
        evaluate_every=5,
        verbose=1
    )
    lda.fit(matrix)
    print(f"[TopicModeling] ✅ LDA trained. Perplexity={lda.perplexity(matrix):.2f}")
    
    # استخراج الـ topics
    n_top_words = 10
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [round(float(topic[i]), 4) for i in top_indices]
        
        topics.append({
            "topic_id": topic_idx,
            "label": f"Topic {topic_idx + 1}",
            "words": top_words,
            "weights": top_weights
        })
    
    # توزيع الـ topics
    distributions, doc_topic_matrix = get_topic_distribution(lda, matrix, doc_ids)
    
    # Coherence Score
    print("[TopicModeling] Computing Coherence Score...")
    coherence = compute_coherence(lda, feature_names, matrix)
    print(f"[TopicModeling] Coherence={coherence:.4f}")
    
    # Perplexity
    perplexity = round(float(lda.perplexity(matrix)), 2)
    
    # حساب حجم كل topic (عدد الوثائق الغالبة فيه)
    dominant_topics = [d["dominant_topic"] for d in distributions]
    topic_sizes = {}
    for t in dominant_topics:
        topic_sizes[t] = topic_sizes.get(t, 0) + 1
    
    for topic in topics:
        topic["doc_count"] = topic_sizes.get(topic["topic_id"], 0)
    
    # Topic-Word heatmap data (top 8 topics × top 10 words)
    heatmap_data = []
    for topic in topics:
        heatmap_data.append(topic["weights"])
    
    result = {
        "n_topics": n_topics,
        "n_documents": len(docs),
        "perplexity": perplexity,
        "coherence_score": coherence,
        "topics": topics,
        "topic_sizes": [
            topic_sizes.get(i, 0) for i in range(n_topics)
        ],
        "heatmap": {
            "topic_labels": [f"Topic {i+1}" for i in range(n_topics)],
            "word_labels": topics[0]["words"] if topics else [],
            "data": heatmap_data
        },
        # sample توزيع (أول 100 وثيقة)
        "sample_distributions": [
            {
                "doc_id": d["doc_id"],
                "dominant_topic": d["dominant_topic"],
                "weights": d["topic_weights"]
            }
            for d in distributions[:100]
        ]
    }
    
    # حفظ في cache
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": result}, f)
        print(f"[TopicModeling] 💾 Cached (n_topics={n_topics})")
    except Exception as e:
        print(f"[TopicModeling] ⚠️ Cache save failed: {e}")
    
    return result


def compare_topic_counts(n_range: list = None) -> dict:
    """
    مقارنة Perplexity و Coherence لعدة قيم n_topics
    """
    if n_range is None:
        n_range = [3, 5, 7, 10]
    
    docs = load_documents()
    matrix, vectorizer = build_count_matrix(docs)
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    results = []
    for n in n_range:
        print(f"\n[TopicModeling] Testing n_topics={n}...")
        lda = LatentDirichletAllocation(
            n_components=n,
            random_state=42,
            max_iter=15,
            learning_method='online'
        )
        lda.fit(matrix)
        perplexity = round(float(lda.perplexity(matrix)), 2)
        coherence = compute_coherence(lda, feature_names, matrix)
        
        results.append({
            "n_topics": n,
            "perplexity": perplexity,
            "coherence": coherence
        })
        print(f"   n={n}: perplexity={perplexity}, coherence={coherence}")
    
    return {
        "n_values": n_range,
        "perplexity_values": [r["perplexity"] for r in results],
        "coherence_values": [r["coherence"] for r in results],
        "details": results
    }