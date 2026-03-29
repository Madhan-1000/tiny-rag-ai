# Tiny RAG AI

<img src="assets/logo.jpg" alt="TINY RAG AI" width="100" />

A fully local RAG chatbot library. No API keys, no external servers, just 2 lines of code.  
Runs entirely on-device in ~500MB of memory using the Qwen2.5-0.5B model.

## What it is
- Wraps the Qwen2.5-0.5B-Instruct-GGUF model.
- Runs inference in about 330 MB of memory on-device.
- Avoids heavy RAG pipelines by accepting documents directly through its parameters.

## Why use it
- Minimal setup and small footprint.
- Focus on your app logic instead of infrastructure.



 
## Installation
 
### Step 1 — Install llama-cpp-python (pre-built, no compilation needed)
 
Pick the version that matches your hardware:
 
```bash
# CPU only
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
 
# CUDA 12.1 (NVIDIA GPU)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
 
# Metal (macOS Apple Silicon)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```
 
### Step 2 — Install tiny-rag-ai
 
```bash
pip install tiny-rag-ai
```
 
---
 
## Quick Start
 
```python
import tiny_rag_ai
 
tiny_rag_ai.index("./docs")
 
answer = tiny_rag_ai.chat("What is your return policy?", use_case="customer support bot")
print(answer)
```
 
`index()` only needs to run once. After that, the FAISS index is saved to disk and reloaded automatically on the next run.
 
---
 
## Framework Examples
 
### Flask
 
```python
import tiny_rag_ai
tiny_rag_ai.index("./docs")
 
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    return jsonify({"answer": tiny_rag_ai.chat(data["message"], use_case="support bot")})
```
 
### FastAPI
 
```python
import tiny_rag_ai
tiny_rag_ai.index("./docs")
 
@app.post("/chat")
def chat(req: ChatRequest):
    return {"answer": tiny_rag_ai.chat(req.message, use_case="support bot")}
```
 
### Django
 
```python
# Call index() in AppConfig.ready(), then use tiny_rag_ai.chat() in your view as normal.
import tiny_rag_ai
answer = tiny_rag_ai.chat(request.POST["message"], use_case="support bot")
```
 
---
 
## Deploying to Render (or any cloud server)
 
Set this environment variable to persist downloaded models across deploys:
 
```
TINY_AI_CACHE_DIR=/data/models
```
 
Mount a persistent disk at `/data` with a minimum of 1GB.
 
---
 
## API Reference
 
### `tiny_rag_ai.index(folder_path, save_path, n_ctx, threads)`
 
| Parameter | Default | Description |
|---|---|---|
| `folder_path` | required | Path to your documents folder (PDF/TXT) |
| `save_path` | `./tiny_ai_data` | Where to save the FAISS index and chunks |
| `n_ctx` | `2048` | Context window size for the LLM |
| `threads` | `8` | Number of CPU threads for inference |
 
### `tiny_rag_ai.chat(query, use_case)`
 
| Parameter | Default | Description |
|---|---|---|
| `query` | required | The user's question |
| `use_case` | required | Describes the bot's role e.g. `"customer support bot"` |
 
---

## Stack
- LLM: Qwen2.5 0.5B via llama-cpp-python
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector store: FAISS
- PDF loading: PyMuPDF

## License
MIT