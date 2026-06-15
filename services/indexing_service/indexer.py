"""
Indexing Service
Builds an Inverted Index for each dataset and saves/loads it from disk
"""

import os
import json
import pickle
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
            total_length += len(tokens)

            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq

        self.doc_count = len(self.doc_ids)
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
        print(f"[Indexer] ✅ Index is ready — {len(self.index)} unique terms")

    def get_postings(self, term: str) -> dict:
        return self.index.get(term, {})

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
        print(f"[Indexer] 📂 Loaded from {path} — {self.doc_count} documents")


def build_and_save_index(dataset_key: str):
    docs_path = (
        config.DATASET1_DOCS if dataset_key == "dataset1" else config.DATASET2_DOCS
    )
    index_dir = config.INDEX1_DIR if dataset_key == "dataset1" else config.INDEX2_DIR
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


def load_index(dataset_key: str) -> InvertedIndex:
    index_dir = config.INDEX1_DIR if dataset_key == "dataset1" else config.INDEX2_DIR
    index_path = os.path.join(index_dir, "inverted_index.pkl")

    if not os.path.exists(index_path):
        print(f"[Indexer] ⚠️ Index not found, building it now...")
        return build_and_save_index(dataset_key)

    idx = InvertedIndex()
    idx.load(index_path)
    return idx


if __name__ == "__main__":
    for ds in ["dataset1", "dataset2"]:
        build_and_save_index(ds)
