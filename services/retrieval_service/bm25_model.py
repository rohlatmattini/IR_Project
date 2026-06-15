"""
BM25 Retrieval Model
نموذج احتمالي يحسب درجة صلة كل وثيقة باستعلام معين
"""

import os
import json
import pickle
import math
from collections import defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from services.preprocessing_service.preprocessor import preprocess_text
import config


class BM25Model:
    """
    صيغة BM25:

    Score(D, Q) = Σ IDF(qi) * [ TF(qi,D) * (k1+1) ] / [ TF(qi,D) + k1*(1-b+b*|D|/avgdl) ]

    k1 : يتحكم بتشبع تكرار المصطلح (القيم الشائعة: 1.2 – 2.0)
    b  : يتحكم بتطبيع الطول (0 = بدون تطبيع، 1 = تطبيع كامل)
    """

    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 or config.BM25_PARAMS["k1"]
        self.b = b or config.BM25_PARAMS["b"]

        self.doc_ids = []
        self.doc_tokens = {}        # doc_id → tokens list
        self.doc_lengths = {}       # doc_id → عدد الكلمات
        self.avg_doc_length = 0
        self.doc_count = 0
        self.df = defaultdict(int)  # term → عدد الوثائق اللي تحتوي عليه
        self.posting_list = defaultdict(set)  # term → set of doc_ids (للبحث السريع)
        self.is_fitted = False

    def fit(self, documents: list):
        """
        documents: قائمة dict فيها 'doc_id' و 'text'
        """
        print(f"[BM25] Training model on {len(documents)} documents (k1={self.k1}, b={self.b})...")

        if not isinstance(documents, list):
            raise ValueError(f"Documents must be a list, got {type(documents)}")

        total_length = 0

        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                doc_id = f"doc_{i}"
                text = doc
            else:
                doc_id = doc.get('doc_id', f"doc_{i}")
                text = doc.get('text', '')

            tokens = preprocess_text(text, return_tokens=True)
            self.doc_ids.append(doc_id)
            self.doc_tokens[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # DF + Posting List: كل مصطلح يُعدّ مرة واحدة لكل وثيقة
            for term in set(tokens):
                self.df[term] += 1
                self.posting_list[term].add(doc_id)  # ← تخزين الوثائق لكل مصطلح

        self.doc_count = len(documents)
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 1
        self.is_fitted = True
        print(f"[BM25] ✅ Model is ready — average doc length: {self.avg_doc_length:.1f} terms")

    def _idf(self, term: str) -> float:
        """حساب IDF لمصطلح معين"""
        df = self.df.get(term, 0)
        # صيغة IDF المحسّنة (Robertson)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_tokens: list, doc_id: str) -> float:
        """حساب درجة BM25 لوثيقة واحدة"""
        tokens = self.doc_tokens[doc_id]
        doc_len = self.doc_lengths[doc_id]

        tf_map = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1

        score = 0.0
        for term in query_tokens:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_length
            )
            score += idf * (numerator / denominator)
        return score

    def search(
        self, query: str, top_k: int = None, k1: float = None, b: float = None
    ) -> list:
        """
        يبحث ويرجع أفضل top_k وثائق
        يمكن تغيير k1 و b عند التنفيذ من الواجهة
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b

        top_k = top_k or config.RETRIEVAL["top_k"]
        query_tokens = preprocess_text(query, return_tokens=True)

        # جلب المرشحين مباشرة من الـ Posting List (سريع)
        candidate_ids = set()
        for term in query_tokens:
            candidate_ids.update(self.posting_list.get(term, set()))

        scores = []
        for doc_id in candidate_ids:
            s = self._score(query_tokens, doc_id)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[BM25] 💾 Saved successfully to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        # توافق مع النماذج القديمة المحفوظة بدون posting_list
        if not hasattr(self, 'posting_list'):
            print("[BM25] ⚠️ Old model detected, rebuilding posting list...")
            self.posting_list = defaultdict(set)
            for doc_id, tokens in self.doc_tokens.items():
                for term in set(tokens):
                    self.posting_list[term].add(doc_id)
        print(f"[BM25] 📂 Loaded from {path}")


def load_documents_safely(docs_path: str):
    with open(docs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        # تحقق من وجود sub-key أولاً
        sub = data.get('documents', data.get('data', None))
        if isinstance(sub, list):
            return sub
        # وإلا اعتبر الـ dict نفسه هو الوثائق (key=id, value=text)
        return [{'doc_id': str(k), 'text': str(v)} for k, v in data.items()]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

def get_bm25_model(dataset_key: str, k1: float = None, b: float = None) -> BM25Model:
    index_dir = config.INDEX1_DIR if dataset_key == "dataset1" else config.INDEX2_DIR
    model_path = os.path.join(index_dir, "bm25_model.pkl")

    model = BM25Model(k1=k1, b=b)

    if os.path.exists(model_path) and k1 is None and b is None:
        try:
            model.load(model_path)
            return model
        except Exception as e:
            print(f"[BM25] Load error: {e}, model will be rebuilt")

    docs_path = config.DATASET1_DOCS if dataset_key == "dataset1" else config.DATASET2_DOCS
    documents = load_documents_safely(docs_path)

    model.fit(documents)
    if k1 is None and b is None:
        model.save(model_path)
    return model


# ===== للتجربة =====
if __name__ == "__main__":
    model = get_bm25_model("dataset1")
    results = model.search("machine learning algorithms", top_k=5)
    print("Results:", results)

    results2 = model.search("machine learning algorithms", top_k=5, k1=1.2, b=0.5)
    print("Results with k1=1.2, b=0.5:", results2)
