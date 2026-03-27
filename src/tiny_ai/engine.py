import faiss
import pickle
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from pathlib import Path

_embed_model = None
_llm = None
_index = None
_chunks = None
_save_path = None

def set_save_path(path: str):
    global _save_path
    _save_path = path

def _load_models():
    global _embed_model, _llm, _index, _chunks
    if _embed_model is None:
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if _llm is None:
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )
        _llm = Llama(model_path=model_path, n_ctx=4096, n_threads=16)

    if _index is None:
        _index = faiss.read_index(str(Path(_save_path) / "faiss.index"))  # ← fixed

    if _chunks is None:
        try:
            with open(Path(_save_path) / "chunks.pkl", "rb") as data_file:  # ← fixed
                _chunks = pickle.load(data_file)
        except FileNotFoundError:
            print(f"Error: The file with data is not found")
        except Exception as e:
            print(f"An error occurred: {e}")


def search(query: str, k=6):
    _load_models()
    query_vector = _embed_model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = _index.search(query_vector, k)
    results = [_chunks[i] for i in indices[0]]
    return "\n".join(results)


def build_prompt(query: str, context, use_case: str):
    prompt = f"""<|im_start|>system
You are a helpful chatbot assistant for {use_case} and you must say I don't know or I can't answer it with a reason why if anything outside of this context.
Use ONLY the context below to answer.
CONTEXT:
{context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def generate(prompt:str,max_tokens:int=650):
    _load_models()
    answer_output=_llm(prompt,max_tokens=max_tokens,echo=False,stop=["<|im_end|>"])
    return answer_output["choices"][0]["text"].strip()

def chat(user_query:str,usecase:str):
    context=search(query=user_query)
    
    prompt=build_prompt(user_query,context=context,use_case=usecase)
    
    answer=generate(prompt=prompt)
    return answer