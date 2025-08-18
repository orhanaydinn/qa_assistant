import os
import json
import faiss

def load_rag_index():
    """
    Load pre-built FAISS index and text chunks for dataset RAG.
    """
    base_path = r"C:/Users/Orhan/Desktop/Software Project/Artificial Intelligence/Projects/deneme/rag_index"
    index_path = os.path.join(base_path, "index.faiss")
    text_path = os.path.join(base_path, "texts.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at {index_path}")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Texts file not found at {text_path}")

    index = faiss.read_index(index_path)
    with open(text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    return index, texts
