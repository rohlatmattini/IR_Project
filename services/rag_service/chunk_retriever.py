import os
import sys
import numpy as np
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from services.rag_service.chunker import chunk_text
from services.retrieval_service.embedding_model import get_shared_sentence_model


def _get_model():
    return get_shared_sentence_model(config.EMBEDDING["model_name"])


def build_chunk_index(documents: list, chunk_size: int = 150, overlap: int = 40):

    model = _get_model()
    all_chunks = []

    for doc in documents:
        text = (doc.get("full_text") or doc.get("snippet") or "").strip()
        if not text:
            continue

        doc_id = doc.get("doc_id")
        title = doc.get("title", f"Document {doc_id}")
        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "doc_id": doc_id,
                "title": title,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": piece,
            })

    if not all_chunks:
        return None, []

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"[RAG-Chunker] 🔎 Built vector store: {len(all_chunks)} chunks من {len(documents)} وثيقة")
    return index, all_chunks


def search_chunks(query: str, index, chunks: list, top_k: int = 5) -> list:
    """يبحث عن أقرب chunks للاستعلام داخل الـ vector store"""
    if index is None or not chunks:
        return []

    model = _get_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        c = chunks[idx]
        results.append({
            "doc_id": c["doc_id"],
            "title": c["title"],
            "chunk_id": c["chunk_id"],
            "snippet": c["text"],
            "full_text": c["text"],
            "score": round(float(score), 4),
        })

    return results