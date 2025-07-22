# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:18:11 2025

@author: Orhan
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search_similar_chunk(question, index, chunks, top_k=1):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return chunks[indices[0][0]]
