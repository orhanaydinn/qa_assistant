# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:18:11 2025

@author: Orhan
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_faiss_index(embeddings):
    return embeddings  # FAISS yerine doğrudan numpy array

def search_similar_chunk(index, query_embedding, top_k=5):
    similarities = cosine_similarity([query_embedding], index)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]
