
import os
import json
import pickle
import math
from collections import defaultdict
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from services.preprocessing_service.preprocessor import preprocess_text
import config


class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(dict)
        self.doc_lengths = {}
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.doc_ids = []
        self.df = {}               
        self.doc_texts = {}        
        self.doc_tokens = {}       
    def build(self, documents: list):
        print(f"[Indexer] Building index for {len(documents)} documents...")
        total_length = 0

        for i, doc in enumerate(tqdm(documents, desc="Indexing")):
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
            self.doc_lengths[doc_id] = len(tokens)
            self.doc_texts[doc_id] = text          
            self.doc_tokens[doc_id] = tokens       
            total_length += len(tokens)

            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq

        self.doc_count = len(self.doc_ids)
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

        self.df = {term: len(postings) for term, postings in self.index.items()}

        print(f"[Indexer] ✅ Index is ready — {len(self.index)} unique terms")

   
    def get_postings(self, term: str) -> dict:
        return self.index.get(term, {})

    def get_df(self, term: str) -> int:
        return self.df.get(term, 0)

    def get_idf(self, term: str, variant: str = "bm25") -> float:
        df = self.get_df(term)
        if df == 0:
            return 0.0
        if variant == "bm25":
            return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        else:  # tfidf
            return math.log(self.doc_count / df) + 1

    def get_doc_tokens(self, doc_id: str) -> list:
        return self.doc_tokens.get(doc_id, [])

    def get_doc_length(self, doc_id: str) -> int:
        return self.doc_lengths.get(doc_id, 0)

    def get_doc_text(self, doc_id: str) -> str:
        return self.doc_texts.get(doc_id, "")

    def get_candidates_for_query(self, query_tokens: list) -> set:
        candidates = set()
        for term in query_tokens:
            candidates.update(self.index.get(term, {}).keys())
        return candidates

    def get_all_terms(self) -> list:
        return list(self.index.keys())
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": dict(self.index),
                    "doc_lengths": self.doc_lengths,
                    "doc_count": self.doc_count,
                    "avg_doc_length": self.avg_doc_length,
                    "doc_ids": self.doc_ids,
                    "df": self.df,
                    "doc_texts": self.doc_texts,
                    "doc_tokens": self.doc_tokens,
                },
                f,
            )
        print(f"[Indexer] 💾 Saved successfully to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = defaultdict(dict, data["index"])
        self.doc_lengths = data["doc_lengths"]
        self.doc_count = data["doc_count"]
        self.avg_doc_length = data["avg_doc_length"]
        self.doc_ids = data["doc_ids"]
        self.df = data.get("df", {term: len(p) for term, p in self.index.items()})
        self.doc_texts = data.get("doc_texts", {})
        self.doc_tokens = data.get("doc_tokens", {})
        print(f"[Indexer] 📂 Loaded from {path} — {self.doc_count} documents")


_index_instance = None


def get_index(dataset_key: str = "dataset2") -> InvertedIndex:
 
    global _index_instance
    if _index_instance is None:
        _index_instance = load_index(dataset_key)
    return _index_instance


def build_and_save_index(dataset_key: str = "dataset2"):
    docs_path = config.DATASET2_DOCS
    index_dir = config.INDEX2_DIR
    index_path = os.path.join(index_dir, "inverted_index.pkl")

    print(f"[Indexer] Loading documents for {dataset_key}...")
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        documents = [{"doc_id": str(k), "text": str(v)} for k, v in data.items()]
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError(f"Unexpected format: {type(data)}")

    idx = InvertedIndex()
    idx.build(documents)
    idx.save(index_path)
    return idx


def load_index(dataset_key: str = "dataset2") -> InvertedIndex:
    index_dir = config.INDEX2_DIR
    index_path = os.path.join(index_dir, "inverted_index.pkl")

    if not os.path.exists(index_path):
        print(f"[Indexer] ⚠️ Index not found, building...")
        return build_and_save_index(dataset_key)

    idx = InvertedIndex()
    idx.load(index_path)

    if not idx.doc_tokens:
        print("[Indexer] ⚠️ Old index without doc_tokens, rebuilding...")
        return build_and_save_index(dataset_key)

    return idx


if __name__ == "__main__":
    build_and_save_index("dataset2")