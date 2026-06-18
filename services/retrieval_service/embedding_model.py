
import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class EmbeddingModel:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING["model_name"]
        self.model = None
        self.doc_embeddings = None  # مصفوفة (num_docs × embedding_dim)
        self.doc_ids = []
        self.is_fitted = False
        self.faiss_index = None  # ⭐ FAISS index للبحث السريع

    def _load_model(self):
        if self.model is None:
            print(f"[Embedding] Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("[Embedding] ✅ Model is ready")

    def fit(self, documents: list):
        if not documents:
            raise ValueError(
                "[Embedding] ❌ Error: Document list is empty! Cannot fit the model."
            )

        self._load_model()
        print(f"[Embedding] Generating embeddings for {len(documents)} documents...")

        texts = []
        self.doc_ids = []

        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                self.doc_ids.append(f"doc_{i}")
                texts.append(doc[:512])
            elif isinstance(doc, dict):
                self.doc_ids.append(
                    str(doc.get("doc_id") or doc.get("id") or f"doc_{i}")
                )
                texts.append(str(doc.get("text", doc.get("body", "")))[:512])

        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=config.EMBEDDING["batch_size"],
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.is_fitted = True
        self._build_faiss_index()
        print(f"[Embedding] ✅ Model ready — matrix shape: {self.doc_embeddings.shape}")

    def _build_faiss_index(self):
        """⭐ يبني FAISS index من doc_embeddings الحالية (IndexFlatIP = cosine بعد normalize)"""
        embeddings = np.array(self.doc_embeddings, dtype="float32")
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        print(f"[Embedding] 🔎 FAISS index built — {self.faiss_index.ntotal} vectors")

    def search(self, query: str, top_k: int = None) -> list:
        """
        يرجع: قائمة (doc_id, score) مرتبة تنازلياً — باستخدام FAISS بدل cosine اليدوي
        """
        if (
            not self.is_fitted
            or self.doc_embeddings is None
            or self.doc_embeddings.shape[0] == 0
        ):
            raise RuntimeError("Model has not been fitted successfully or contains no documents.")

        # ⭐ fallback: لو الموديل محمّل من ملف قديم بدون faiss_index، نبنيه الآن
        if self.faiss_index is None:
            self._build_faiss_index()

        self._load_model()
        top_k = top_k or config.RETRIEVAL["top_k"]

        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")

        scores, indices = self.faiss_index.search(query_emb, top_k)

        results = [
            (self.doc_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1 and score > 0
        ]
        return results
    def get_query_embedding(self, query: str) -> np.ndarray:
        self._load_model()
        return self.model.encode([query], normalize_embeddings=True)[0]

    def save(self, path: str):
        if not self.is_fitted or self.doc_embeddings is None:
            print("[Embedding] ⚠️ Warning: Save skipped because model is not fitted.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "doc_embeddings": self.doc_embeddings,
                    "doc_ids": self.doc_ids,
                    "model_name": self.model_name,
                },
                f,
            )
        print(f"[Embedding] 💾 Saved successfully to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.doc_embeddings = data["doc_embeddings"]
        self.doc_ids = data["doc_ids"]
        self.model_name = data["model_name"]
        self.is_fitted = True
        self._build_faiss_index()  # ⭐ نبني الـ FAISS index من جديد بعد التحميل
        print(f"[Embedding] 📂 Loaded from {path}")

def load_documents_safely(docs_path: str) -> list:
    """
    تحميل الوثائق من ملف JSON بشكل مرن وآمن مهما كانت بنيته الأساسية.
    """
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_list = []

    if isinstance(data, dict):
        sub_list = data.get("documents", data.get("data", data.get("docs")))
        if isinstance(sub_list, list):
            return sub_list

        for k, v in data.items():
            if isinstance(v, dict):
                v["doc_id"] = v.get("doc_id", str(k))
                normalized_list.append(v)
            else:
                normalized_list.append({"doc_id": str(k), "text": str(v)})

    elif isinstance(data, list):
        normalized_list = data

    else:
        raise ValueError(f"Unexpected document file format: {type(data)}")

    return normalized_list


def get_embedding_model(dataset_key: str) -> EmbeddingModel:
    index_dir = config.INDEX2_DIR
    model_path = os.path.join(index_dir, "embedding_model.pkl")

    model = EmbeddingModel()

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model.load(model_path)
            if model.doc_embeddings is not None and model.doc_embeddings.shape[0] > 0:
                return model
            print("[Embedding] ⚠️ Stored file is empty, rebuilding model...")
        except Exception as e:
            print(f"[Embedding] Load error: {e}, model will be rebuilt")

    docs_path = config.DATASET2_DOCS
    documents = load_documents_safely(docs_path)

    model.fit(documents)
    model.save(model_path)
    return model

if __name__ == "__main__":
    model = get_embedding_model("dataset2")
    results = model.search("deep learning neural networks", top_k=5)
    print("Results:", results)