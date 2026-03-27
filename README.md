# Tiny AI

A lightweight Python library for embedding a chatbot into your web app with a simple function call.

## What it is
- Wraps the Qwen2.5-0.5B-Instruct-GGUF model.
- Runs inference in about 330 MB of memory on-device.
- Avoids heavy RAG pipelines by accepting documents directly through its parameters.

## Why use it
- Minimal setup and small footprint.
- Focus on your app logic instead of infrastructure.

# tiny-rag-ai

A fully local RAG chatbot library. No API keys, no external servers, just 2 lines of code. Runs entirely on-device in ~500MB of memory.

## Why use it
- No API keys needed
- No external servers
- Fully offline after first model download
- ~500MB memory footprint
- Works with any PDF or TXT documents

## Installation

### Step 1 — Install llama-cpp-python (pre-built, no compilation)
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Step 2 — Install tiny-rag-ai
```bash
pip install tiny-rag-ai
```

## Quick Start
```python
import tiny_ai

tiny_ai.index("./docs")

answer = tiny_ai.chat("What is your return policy?", use_case="customer support bot")
print(answer)
```

## Usage in Flask
```python
from flask import Flask, request, jsonify
import tiny_ai

app = Flask(__name__)
tiny_ai.index("./docs")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    answer = tiny_ai.chat(data.get("message"), use_case="customer support bot")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
```

## Usage in FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel
import tiny_ai

app = FastAPI()

@app.on_event("startup")
def startup():
    tiny_ai.index("./docs")

class ChatRequest(BaseModel):
    message: str
    use_case: str = "customer support bot"

@app.post("/chat")
def chat(req: ChatRequest):
    answer = tiny_ai.chat(req.message, use_case=req.use_case)
    return {"answer": answer}
```

## Usage in Django
```python
# apps.py
from django.apps import AppConfig

class MyAppConfig(AppConfig):
    name = "myapp"

    def ready(self):
        import tiny_ai
        tiny_ai.index("./docs")
```
```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import tiny_ai

@csrf_exempt
def chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        answer = tiny_ai.chat(data.get("message"), use_case="customer support bot")
        return JsonResponse({"answer": answer})
```
```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("chat/", views.chat),
]
```

## On Render / Production Server

Set this environment variable to persist models across deploys:
```
TINY_AI_CACHE_DIR=/data/models
```

Mount a persistent disk at `/data` with minimum 1GB.

## How it Works
```
Your Docs (PDF/TXT)
      ↓ index()
Chunk → Embed → FAISS index saved to disk
                        ↓ chat()
           Search → Build Prompt → Qwen LLM → Answer
```

## Stack
- LLM: Qwen2.5 0.5B via llama-cpp-python
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector store: FAISS
- PDF loading: PyMuPDF

## License
MIT