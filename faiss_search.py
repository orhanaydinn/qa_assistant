# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:18:11 2025

@author: Orhan
"""

import faiss
import numpy as np
from typing import Sequence, Any

try:
    # Optional: only used when raw text chunks are provided
    from embedder import embed_chunks
except Exception:
    embed_chunks = None

def _is_texty(seq: Sequence[Any]) -> bool:
    if not isinstance(seq, (list, tuple)):
        return False
    if len(seq) == 0:
        return True
    s0 = seq[0]
    if isinstance(s0, str):
        return True
    if isinstance(s0, dict) and ("text" in s0):
        return True
    return False

def _to_float32_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        # Single vector -> make it 2D (1, dim)
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n_items, dim). Got shape={arr.shape}")
    return arr

def create_faiss_index(data, *, assume_embeddings: bool = False, metric: str = "l2"):
    """
    Build a FAISS index.

    Parameters
    ----------
    data : list[str] | list[dict] | list[list[float]] | np.ndarray
        Either raw text chunks (strings / dicts with 'text') or precomputed embeddings.
    assume_embeddings : bool
        If True, 'data' is treated as embeddings even if it looks like text.
    metric : {"l2", "ip"}
        Distance metric. "l2" recommended (your downstream code expects smaller=better).

    Returns
    -------
    faiss.Index or None
    """
    if data is None:
        return None

    # Decide: text -> embed, else assume embeddings
    if not assume_embeddings and _is_texty(data):
        if embed_chunks is None:
            raise RuntimeError("embedder.embed_chunks is not available but text chunks were provided.")
        # normalize/clean simple text inputs
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
        # Precomputed embeddings path
        vecs = _to_float32_2d(data)

    n, d = vecs.shape
    if metric.lower() == "ip":
        # cosine (IP) kullanmak istersen: önce normalize et
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(vecs)
    return index
