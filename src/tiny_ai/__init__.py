from . import engine,indexer
import faiss
import numpy as np


def index(folder_path, save_path="./tiny_ai_data"):
    """
    This fucntion does many things to initialize the setup.
    -> First loads all the documents
    -> Second the seperates them into chunks 
    -> Third Loads all the AI models
    -> Then Embeddes the chunks into vectors
    -> Then it will store them in a faiss index
    -> After that the path for saving the indexes is done 
    """
    
    documents=indexer.load_documenmts(folder_path=folder_path)
    chunks=[]
    for doc in documents:
        chunks.extend(indexer.chunk_text(text=doc,chunk_size=500,overlap=50 ))
    engine._load_models()
    embeddings=indexer.embed_chunks(chunks=chunks, embed_model=engine._embed_model)
    dimensions=embeddings.shape[1]
    faiss_index=faiss.IndexFlatL2(dimensions)
    engine.set_save_path(save_path) 
    faiss_index.add(embeddings)
    indexer.save_index(faiss_index,chunks=chunks,save_path=save_path)
    
    
def chat(query, use_case):  
    """
    This fucntion does exactly what is in its name it just calls the ai fucntion.
    """
    return engine.chat(user_query=query,usecase=use_case)