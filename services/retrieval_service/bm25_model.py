import os
import sys
import pickle

from rank_bm25 import BM25Okapi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from services.preprocessing_service.preprocessor import preprocess_text
import config


class BM25Model:
  

    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 or config.BM25_PARAMS["k1"]
        self.b = b or config.BM25_PARAMS["b"]
        self.bm25 = None
        self.doc_ids = []
        self.is_fitted = False

    def fit(self, documents: list):
        print(f"[BM25] Building BM25Okapi for {len(documents)} documents...")
        tokenized_corpus = []
        self.doc_ids = []

        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                doc_id = f"doc_{i}"
                text = doc
            elif isinstance(doc, dict):
                doc_id = str(doc.get("doc_id") or doc.get("id") or f"doc_{i}")
                text = str(doc.get("text", doc.get("body", "")))
            else:
                continue

            tokens = preprocess_text(text, return_tokens=True)
            self.doc_ids.append(doc_id)
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.is_fitted = True
        print(f"[BM25] ✅ BM25Okapi ready — {len(self.doc_ids)} documents (k1={self.k1}, b={self.b})")

    def search(self, query: str, top_k: int = None, k1: float = None, b: float = None) -> list:
     
        if not self.is_fitted or self.bm25 is None:
            raise RuntimeError("[BM25] Model not fitted. Call fit() first.")

        if k1 is not None and k1 != self.k1:
            self.k1 = k1
            self.bm25.k1 = k1
        if b is not None and b != self.b:
            self.b = b
            self.bm25.b = b

        top_k = top_k or config.RETRIEVAL["top_k"]
        query_tokens = preprocess_text(query, return_tokens=True)

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            ((self.doc_ids[i], float(scores[i])) for i in range(len(self.doc_ids)) if scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def save(self, path: str):
        if not self.is_fitted:
            print("[BM25] ⚠️ Save skipped — model not fitted.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "doc_ids": self.doc_ids,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )
        print(f"[BM25] 💾 Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.k1 = data.get("k1", self.k1)
        self.b = data.get("b", self.b)
        self.is_fitted = True
        print(f"[BM25] 📂 Loaded from {path} — {len(self.doc_ids)} documents")


def _load_documents(docs_path: str) -> list:
    import json
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                normalized.append({"doc_id": str(v.get("doc_id", k)), "text": str(v.get("text", v.get("body", "")))})
            else:
                normalized.append({"doc_id": str(k), "text": str(v)})
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                doc_id = str(item.get("doc_id") or item.get("id") or f"doc_{i}")
                text = str(item.get("text", item.get("body", "")))
                normalized.append({"doc_id": doc_id, "text": text})
            else:
                normalized.append({"doc_id": str(i), "text": str(item)})
    return normalized


def get_bm25_model(dataset_key: str = "dataset2", k1: float = None, b: float = None) -> BM25Model:
    index_dir = config.INDEX2_DIR
    model_path = os.path.join(index_dir, "bm25_model.pkl")

    model = BM25Model(k1=k1, b=b)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model.load(model_path)
            if model.bm25 is not None and len(model.doc_ids) > 0:
                return model
            print("[BM25] ⚠️ Cached file invalid, rebuilding...")
        except Exception as e:
            print(f"[BM25] Load error: {e}, rebuilding...")

    documents = _load_documents(config.DATASET2_DOCS)
    model.fit(documents)
    model.save(model_path)
    return model


if __name__ == "__main__":
    model = get_bm25_model("dataset2")
    results = model.search("machine learning algorithms", top_k=5)
    print("Results:", results)
