
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from services.preprocessing_service.preprocessor import preprocess_text
from services.indexing_service.indexer import get_index, InvertedIndex
import config


class BM25Model:
    """
    BM25 يعتمد كلياً على الـ Inverted Index.
    ما عاد يخزن posting_list خاص فيه.
    """

    def __init__(self, k1: float = None, b: float = None, index: InvertedIndex = None):
        self.k1 = k1 or config.BM25_PARAMS["k1"]
        self.b = b or config.BM25_PARAMS["b"]
        self.index = index
        self.is_fitted = index is not None

    def fit(self, index: InvertedIndex = None):
        """BM25 ما عنده تدريب فعلي. بس بياخد reference للفهرس."""
        if index is not None:
            self.index = index
        if self.index is None:
            raise ValueError("BM25 needs an InvertedIndex instance!")
        self.is_fitted = True
        print(f"[BM25] ✅ Connected to InvertedIndex ({self.index.doc_count} docs, k1={self.k1}, b={self.b})")

    def _score(self, query_tokens: list, doc_id: str) -> float:
        """يحسب BM25 score لوثيقة واحدة"""
        doc_len = self.index.get_doc_length(doc_id)
        avg_len = self.index.avg_doc_length

        score = 0.0
        for term in query_tokens:
            postings = self.index.get_postings(term)
            tf = postings.get(doc_id, 0)
            if tf == 0:
                continue
            idf = self.index.get_idf(term, variant="bm25")
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = None, k1: float = None, b: float = None) -> list:
        """يبحث ويرجع أفضل top_k وثائق باستخدام الفهرس"""
        if not self.is_fitted:
            raise RuntimeError("BM25 not connected to index.")

        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b

        top_k = top_k or config.RETRIEVAL["top_k"]
        query_tokens = preprocess_text(query, return_tokens=True)

        # استرجاع الوثائق المرشحة من الفهرس
        candidate_ids = self.index.get_candidates_for_query(query_tokens)

        scores = []
        for doc_id in candidate_ids:
            s = self._score(query_tokens, doc_id)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def get_bm25_model(dataset_key: str = "dataset2", k1: float = None, b: float = None) -> BM25Model:
    """
    يرجع BM25 جاهز ومتصل بالفهرس.
    """
    index = get_index(dataset_key)
    model = BM25Model(k1=k1, b=b, index=index)
    model.fit()
    return model


if __name__ == "__main__":
    model = get_bm25_model("dataset2")
    results = model.search("machine learning algorithms", top_k=5)
    print("Results:", results)