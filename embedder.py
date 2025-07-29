# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:18:01 2025

@author: Orhan
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks, convert_to_numpy=True)
