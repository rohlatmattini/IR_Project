
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
            max_features=50000, 
            sublinear_tf=True, 
            min_df=2, 
        )
        self.doc_matrix = None  
        self.doc_ids = []  
        self.is_fitted = False

    def fit(self, documents: list):
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
        
        if not self.is_fitted or self.doc_matrix is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        top_k = top_k or config.RETRIEVAL["top_k"]

        # تأكد من أن هذه الأسطر تبدأ بـ 8 مسافات (مستوى واحد تحت تعريف الدالة)
        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])

        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

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
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_list = []

    if isinstance(data, dict):
        sub_list = data.get("documents", data.get("data", data.get("docs")))
        if isinstance(sub_list, list):
            raw_docs = sub_list
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    doc_id = str(v.get("doc_id", k))
                    text = str(v.get("text", v.get("body", "")))
                    normalized_list.append({"doc_id": doc_id, "text": text})
                else:
                    normalized_list.append({"doc_id": str(k), "text": str(v)})
            return normalized_list

    elif isinstance(data, list):
        raw_docs = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    for i, doc in enumerate(raw_docs):
        if isinstance(doc, str):
            normalized_list.append({"doc_id": f"doc_{i}", "text": doc})
        elif isinstance(doc, dict):
            doc_id = str(doc.get("doc_id") or doc.get("id") or f"doc_{i}")
            text = str(doc.get("text", doc.get("body", "")))
            normalized_list.append({"doc_id": doc_id, "text": text})

    return normalized_list

def get_tfidf_model(dataset_key: str) -> TFIDFModel:
    # استخدم INDEX2_DIR مباشرة
    index_dir = config.INDEX2_DIR
    model_path = os.path.join(index_dir, "tfidf_model.pkl")

    model = TFIDFModel()

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model.load(model_path)
            if model.doc_matrix is not None and model.doc_matrix.shape[0] > 0:
                return model
            print("[TF-IDF] ⚠️ Stored file is empty, rebuilding model...")
        except Exception as e:
            print(f"[TF-IDF] Load error: {e}, model will be rebuilt")

    docs_path = config.DATASET2_DOCS
    documents = load_documents_safely(docs_path)

    model.fit(documents)
    model.save(model_path)
    return model

if __name__ == "__main__":
    model = get_tfidf_model("dataset2")
    results = model.search("information retrieval systems", top_k=5)
    print("Results:", results)