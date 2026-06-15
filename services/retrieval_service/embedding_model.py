"""
Embedding Retrieval Model (Sentence-BERT)
يحوّل الوثائق والاستعلامات لمتجهات كثيفة ويحسب التشابه بـ Cosine Similarity
"""

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class EmbeddingModel:
    """
    يستخدم all-MiniLM-L6-v2 — نموذج خفيف وسريع ودقيق
    مناسب للمشاريع التي لا تمتلك GPU قوي
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING["model_name"]
        self.model = None
        self.doc_embeddings = None  # مصفوفة (num_docs × embedding_dim)
        self.doc_ids = []
        self.is_fitted = False

    def _load_model(self):
        if self.model is None:
            print(f"[Embedding] Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("[Embedding] ✅ Model is ready")

    def fit(self, documents: list):
        """
        documents: قائمة من الوثائق الموحدة
        يولّد embedding لكل وثيقة ويحفظها
        """
        if not documents:
            raise ValueError(
                "[Embedding] ❌ Error: Document list is empty! Cannot fit the model."
            )

        self._load_model()
        print(f"[Embedding] Generating embeddings for {len(documents)} documents...")

        # استخراج النصوص والمعرفات بشكل آمن
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
        print(f"[Embedding] ✅ Model ready — matrix shape: {self.doc_embeddings.shape}")

    def search(self, query: str, top_k: int = None) -> list:
        """
        يرجع: قائمة (doc_id, score) مرتبة تنازلياً
        """
        if (
            not self.is_fitted
            or self.doc_embeddings is None
            or self.doc_embeddings.shape[0] == 0
        ):
            raise RuntimeError("Model has not been fitted successfully or contains no documents.")

        self._load_model()
        top_k = top_k or config.RETRIEVAL["top_k"]

        # توليد embedding للاستعلام
        query_emb = self.model.encode([query], normalize_embeddings=True)

        # حساب التشابه مع كل الوثائق
        scores = cosine_similarity(query_emb, self.doc_embeddings).flatten()

        # ترتيب تنازلي
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0
        ]
        return results

    def get_query_embedding(self, query: str) -> np.ndarray:
        """ترجع embedding الاستعلام فقط (للاستخدام في Hybrid)"""
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
        print(f"[Embedding] 📂 Loaded from {path}")


def load_documents_safely(docs_path: str) -> list:
    """
    تحميل الوثائق من ملف JSON بشكل مرن وآمن مهما كانت بنيته الأساسية.
    """
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_list = []

    # 1. إذا كان الملف عبارة عن قاموس (Dict)
    if isinstance(data, dict):
        # التحقق إذا كانت الوثائق مخزنة داخل حقل فرعي
        sub_list = data.get("documents", data.get("data", data.get("docs")))
        if isinstance(sub_list, list):
            return sub_list

        # إذا كان الـ Dict يمثل الـ doc_id كمفتاح والوثيقة كقيمة
        for k, v in data.items():
            if isinstance(v, dict):
                v["doc_id"] = v.get("doc_id", str(k))
                normalized_list.append(v)
            else:
                normalized_list.append({"doc_id": str(k), "text": str(v)})

    # 2. إذا كان الملف عبارة عن قائمة مباشرة من الوثائق (List)
    elif isinstance(data, list):
        normalized_list = data

    else:
        raise ValueError(f"Unexpected document file format: {type(data)}")

    return normalized_list


def get_embedding_model(dataset_key: str) -> EmbeddingModel:
    index_dir = config.INDEX1_DIR if dataset_key == "dataset1" else config.INDEX2_DIR
    model_path = os.path.join(index_dir, "embedding_model.pkl")

    model = EmbeddingModel()

    # محاولة التحميل فقط إذا كان الملف موجوداً وحجمه أكبر من 0
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model.load(model_path)
            # تأكيد إضافي للتأكد من أن الملف القديم لم يحفظ مصفوفة فارغة
            if model.doc_embeddings is not None and model.doc_embeddings.shape[0] > 0:
                return model
            print("[Embedding] ⚠️ Stored file is empty, rebuilding model...")
        except Exception as e:
            print(f"[Embedding] Load error: {e}, model will be rebuilt")

    docs_path = (
        config.DATASET1_DOCS if dataset_key == "dataset1" else config.DATASET2_DOCS
    )
    documents = load_documents_safely(docs_path)

    model.fit(documents)
    model.save(model_path)
    return model


# ===== للتجربة =====
if __name__ == "__main__":
    model = get_embedding_model("dataset1")
    results = model.search("deep learning neural networks", top_k=5)
    print("Results:", results)