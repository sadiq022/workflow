import requests

res = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3-local",
        "messages": [{"role": "user", "content": "Explain RAG"}]
    }
)

print(res.json()["message"]["content"])