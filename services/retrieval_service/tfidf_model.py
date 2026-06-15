"""
TF-IDF / VSM Retrieval Model
يمثل الوثائق والاستعلامات كمتجهات TF-IDF ويحسب التشابه بـ Cosine Similarity
"""

import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from services.preprocessing_service.preprocessor import preprocess_text
import config


class TFIDFModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=50000,  # أكبر 50000 مصطلح
            sublinear_tf=True,  # log(TF) بدل TF العادي — بيحسّن النتائج
            min_df=2,  # تجاهل المصطلحات اللي بتظهر بوثيقة وحدة بس
        )
        self.doc_matrix = None  # مصفوفة TF-IDF للوثائق (sparse)
        self.doc_ids = []  # قائمة IDs بنفس ترتيب الصفوف
        self.is_fitted = False

    def fit(self, documents: list):
        """
        documents: قائمة تحتوي على الوثائق الموحدة
        يعالج النصوص ويبني مصفوفة TF-IDF
        """
        if not documents:
            raise ValueError(
                "[TF-IDF] ❌ Error: Document list is empty! Cannot fit the model."
            )

        print(f"[TF-IDF] Processing {len(documents)} documents...")

        self.doc_ids = [str(d.get("doc_id")) for d in documents]
        processed_texts = [preprocess_text(str(d.get("text", ""))) for d in documents]

        print("[TF-IDF] Building TF-IDF matrix...")
        self.doc_matrix = self.vectorizer.fit_transform(processed_texts)
        self.is_fitted = True
        print(
            f"[TF-IDF] ✅ Model ready — {self.doc_matrix.shape[0]} documents, {self.doc_matrix.shape[1]} terms"
        )

    def search(self, query: str, top_k: int = None) -> list:
        """
        يبحث عن الوثائق الأقرب للاستعلام
        يرجع: قائمة من (doc_id, score) مرتبة تنازلياً
        """
        if not self.is_fitted or self.doc_matrix is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        top_k = top_k or config.RETRIEVAL["top_k"]

        # معالجة الاستعلام بنفس طريقة الوثائق
        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])

        # حساب Cosine Similarity بين الاستعلام وكل الوثائق
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # ترتيب تنازلي وأخذ أعلى top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0
        ]
        return results

    def save(self, path: str):
        if not self.is_fitted or self.doc_matrix is None:
            print("[TF-IDF] ⚠️ Warning: Save skipped because model is empty.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "doc_matrix": self.doc_matrix,
                    "doc_ids": self.doc_ids,
                },
                f,
            )
        print(f"[TF-IDF] 💾 Saved successfully to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.doc_matrix = data["doc_matrix"]
        self.doc_ids = data["doc_ids"]
        self.is_fitted = True
        print(f"[TF-IDF] 📂 Loaded from {path}")


def load_documents_safely(docs_path: str) -> list:
    """
    تحميل الوثائق من ملف JSON بشكل مرن وآمن مهما كانت بنيته.
    توحد المخرجات لتكون دائماً قائمة قواميس تحتوي على 'doc_id' و 'text'.
    """
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_list = []

    # 1. إذا كانت البيانات قاموساً رئيساً (Dict)
    if isinstance(data, dict):
        sub_list = data.get("documents", data.get("data", data.get("docs")))
        if isinstance(sub_list, list):
            raw_docs = sub_list
        else:
            # إذا كان الـ Dict يمثل المعرف كمفتاح والنص كقيمة
            for k, v in data.items():
                if isinstance(v, dict):
                    doc_id = str(v.get("doc_id", k))
                    text = str(v.get("text", v.get("body", "")))
                    normalized_list.append({"doc_id": doc_id, "text": text})
                else:
                    normalized_list.append({"doc_id": str(k), "text": str(v)})
            return normalized_list

    # 2. إذا كانت البيانات قائمة مباشرة (List)
    elif isinstance(data, list):
        raw_docs = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    # توحيد القائمة المستخرجة
    for i, doc in enumerate(raw_docs):
        if isinstance(doc, str):
            normalized_list.append({"doc_id": f"doc_{i}", "text": doc})
        elif isinstance(doc, dict):
            doc_id = str(doc.get("doc_id") or doc.get("id") or f"doc_{i}")
            text = str(doc.get("text", doc.get("body", "")))
            normalized_list.append({"doc_id": doc_id, "text": text})

    return normalized_list


def get_tfidf_model(dataset_key: str) -> TFIDFModel:
    """تحميل أو بناء نموذج TF-IDF لمجموعة بيانات معينة"""
    index_dir = config.INDEX1_DIR if dataset_key == "dataset1" else config.INDEX2_DIR
    model_path = os.path.join(index_dir, "tfidf_model.pkl")

    model = TFIDFModel()

    # محاولة التحميل فقط إن كان الملف موجوداً وغير فارغ
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model.load(model_path)
            if model.doc_matrix is not None and model.doc_matrix.shape[0] > 0:
                return model
            print("[TF-IDF] ⚠️ Stored file is empty, rebuilding model...")
        except Exception as e:
            print(f"[TF-IDF] Load error: {e}, model will be rebuilt")

    # بناء النموذج من الصفر
    docs_path = (
        config.DATASET1_DOCS if dataset_key == "dataset1" else config.DATASET2_DOCS
    )
    documents = load_documents_safely(docs_path)

    model.fit(documents)
    model.save(model_path)
    return model


# ===== للتجربة =====
if __name__ == "__main__":
    model = get_tfidf_model("dataset1")
    results = model.search("information retrieval systems", top_k=5)
    print("Results:", results)