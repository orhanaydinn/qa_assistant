# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 03:56:49 2025

@author: Orhan
"""

import faiss
import numpy as np
from typing import Sequence, Any

try:
    # Sadece ham text verilirse kullanılır
    from embedder import embed_chunks
except Exception:
    embed_chunks = None


def _is_texty(seq: Sequence[Any]) -> bool:
    if not isinstance(seq, (list, tuple)):
        return False
    if len(seq) == 0:
        return True
    s0 = seq[0]
    return isinstance(s0, str) or (isinstance(s0, dict) and ("text" in s0))


def _to_float32_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:  # tek vektör gelmişse (dim,) -> (1, dim)
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n_items, dim). Got shape={arr.shape}")
    return arr


def create_faiss_index(data, *, assume_embeddings: bool = False, metric: str = "l2"):
    """
    data:
      - Önceden hesaplanmış embeddingler (önerilen): np.ndarray (N, D) veya list[list[float]]
      - Alternatif: ham OCR text chunk listesi (str/dict); bu durumda içeride embed eder

    metric: "l2" (varsayılan) veya "ip" (cosine için normalize + inner product)
    """
    if data is None:
        return None

    if not assume_embeddings and _is_texty(data):
        if embed_chunks is None:
            raise RuntimeError("embedder.embed_chunks not available but text chunks were provided.")
        # Temizle
        texts = []
        for c in data:
            if isinstance(c, str):
                t = c.strip()
            elif isinstance(c, dict) and "text" in c:
                t = str(c["text"]).strip()
            else:
                t = str(c).strip()
            if t:
                texts.append(t)
        if len(texts) == 0:
            return None
        vecs = _to_float32_2d(embed_chunks(texts))
    else:
        vecs = _to_float32_2d(data)

    # Boşluk kontrolü
    if vecs.size == 0 or vecs.shape[0] == 0:
        return None

    n, d = vecs.shape
    if metric.lower() == "ip":
        faiss.normalize_L2(vecs)      # cosine benzerlik için
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(vecs)
    return index


# Geriye dönük uyumluluk: app.py 'create_ocr_index' çağırıyorsa
def create_ocr_index(data, **kwargs):
    return create_faiss_index(data, **kwargs)
