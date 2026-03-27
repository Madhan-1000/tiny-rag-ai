import faiss
import pickle
import fitz
from pathlib import Path

def load_documenmts(folder_path: str) -> list:
    """
    Returns all the loaded documents in the path.

    Parameters
    ----------
    path : str
        path for the folder of the web application or any application detailed documents either in .txt files or .pdf files.

    Returns
    -------
    returns all the text in the documents in form of array of each document in a string. 
    """
    docs=[]
    folder=Path(folder_path)
    for pdf_file in folder.glob("**/*.pdf"):
        
        pdf_content=load_pdf(pdf_file_path=pdf_file)
        docs.append(pdf_content)
    
    for txt_file in folder.glob("**/*.txt"):
        
        txt_file_content=load_txt(txt_file_path=txt_file)
        docs.append(txt_file_content)
    
    return docs


def load_txt(txt_file_path: str) -> str:
    """
    Returns all the content in the txt file.

    Parameters
    ----------
    txt_file_path : str
        path for the txt file.
    
    Returns
    -------
    returns the text file content
    """
    text=""
    with open(txt_file_path,"r") as file:
        text=file.read()
        
    return text


def load_pdf(pdf_file_path: str) -> str:
    """
    Returns all the content in the pdf file.

    Parameters
    ----------
    pdf_file_path : str
        path for the pdf file.
    
    Returns
    -------
    returns the pdf file content
    """
    try:
        
        doc=fitz.open(pdf_file_path)
        text=""
        
        for page in doc:
            text+=page.get_text()
        doc.close()
        
    except Exception as e:
        print(f"Error processing file {pdf_file_path.name}: {e}")        
    return text



def chunk_text(text :str , chunk_size:int , overlap:int)-> list :

    """
    Returns the chunked text with size of 'chunk_size' and with overlap of 'overlap'.

    Parameters
    ----------
    text : str
          the text of all the documents 
    chunk_size : int
          the variable for chunking the data 
    overlap : int
          the variable for overlapping the chunked data
    
    Returns
    -------
    returns all the text in the documents in form of array of each document in a string. 
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks:list,embed_model):
    """
    Returns the embedded chunks of data.

    Parameters
    ----------
    chunks : list
          the text of all the documents 
    
    embed_model
          the model with transformer
    Returns
    -------
    array of float32 values which are embedded using the model. 
    """
    embeddings_array=embed_model.encode(chunks).astype("float32")
    return embeddings_array



def save_index(index, chunks: list, save_path: str):
    """
    Saves the FAISS index and chunks to disk.
    """
    path = Path(save_path)
    
    
    faiss.write_index(index, str(path / "faiss.index"))
    
    
    with open(path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks,f)
