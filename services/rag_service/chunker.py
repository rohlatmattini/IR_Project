def chunk_text(text: str, chunk_size: int = 150, overlap: int = 40) -> list:

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= len(words):
            break
        start = end - overlap

    return chunks