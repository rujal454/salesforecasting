import os
import requests
from app.backend.rag import retrieval

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistralai/mistral-7b-instruct")

def chat_with_rag(query):
    context_docs = retrieval.search_chroma(query, n_results=3)
    context = "\n".join([doc['document'] for doc in context_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful sales assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]