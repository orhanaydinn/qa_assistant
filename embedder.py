from sentence_transformers import SentenceTransformer

# Tek embedder modeli – hem PDF hem OCR hem dataset için
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = SentenceTransformer(_MODEL_NAME)

def embed_chunks(chunks):
    """
    Embed a list of text chunks into vector representations.
    """
    if not chunks:
        return []
    return _embedder.encode(chunks, convert_to_numpy=True)
