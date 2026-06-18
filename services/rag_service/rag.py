
import sys, os, requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def generate_answer(query: str, retrieved_docs: list, max_docs: int = 5) -> dict:
    if not GROQ_API_KEY:
        return {
            "success": False,
            "answer": "⚠️ Error: GROQ_API_KEY is not set. Please add it to the .env file.",
            "sources_used": [],
        }

    context_parts = []
    for i, doc in enumerate(retrieved_docs[:max_docs], 1):
        title = doc.get("title", f"Document {doc.get('doc_id', i)}")
        snippet = doc.get("snippet", "")
        context_parts.append(f"[{i}] {title}:\n{snippet}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are an intelligent assistant for an Information Retrieval system.
Based ONLY on the following retrieved documents, answer the user's question clearly and concisely.
If the documents don't contain enough information, say so honestly.

Retrieved Documents:
{context}

User Question: {query}

Provide a clear, direct answer. Cite documents used by their number [1], [2], etc."""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            answer_text = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "answer": answer_text,
                "sources_used": retrieved_docs[:max_docs],
                "model": "llama-3.3-70b-versatile",
            }
        else:
            return {
                "success": False,
                "answer": f"API Error: {response.status_code} - {response.text}",
                "sources_used": [],
            }

    except Exception as e:
        return {
            "success": False,
            "answer": f"Error: {str(e)}",
            "sources_used": [],
        }