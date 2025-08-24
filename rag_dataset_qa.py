from pathlib import Path
import json
import faiss

def load_rag_index():
    """
    rag_index/index.faiss + rag_index/texts.json yükler.
    Dönüş: (faiss_index, texts_list)
    """
    base = Path(__file__).resolve().parent
    rag_dir = base / "rag_index"
    idx_path = rag_dir / "index.faiss"
    txt_path = rag_dir / "texts.json"

    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"texts.json not found: {txt_path}")

    index = faiss.read_index(str(idx_path))
    with open(txt_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    # Basit doğrulama
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts.json içerik listesi boş veya hatalı.")
    if index.ntotal == 0:
        raise ValueError("FAISS index boş (ntotal=0).")

    return index, texts
