# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 16:09:34 2025

@author: Orhan
"""

import io
import streamlit as st
from PIL import Image

# Projendeki mevcut modüller
from embedder import embed_chunks as embed_any
from pdf_parser import extract_text_chunks as extract_pdf_chunks
from faiss_search import create_faiss_index as create_pdf_index
from ocr_utils import extract_ocr_chunks
from ocr_faiss import create_faiss_index as create_ocr_index

def process_upload(uploaded_file):
    """PDF/IMG yükler; parse eder; embedding + FAISS index hazırlar; state'e yazar."""
    file_bytes = uploaded_file.getvalue()
    cache_key = f"{uploaded_file.name}:{len(file_bytes)}"
    is_new_file = (st.session_state.get("_UPLOADED_CACHE_KEY") != cache_key)
    if not is_new_file:
        return

    with st.spinner("Processing file..."):
        file_type = uploaded_file.type or ""
        bio = io.BytesIO(file_bytes)

        if file_type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            raw_chunks = extract_pdf_chunks(bio)
            norm_chunks = []
            for c in raw_chunks:
                if isinstance(c, str): norm_chunks.append(c)
                elif isinstance(c, dict) and "text" in c: norm_chunks.append(str(c["text"]))
                elif c is not None: norm_chunks.append(str(c))
            chunks = [c.strip() for c in norm_chunks if isinstance(c, str) and c.strip()]
            if not chunks:
                st.error("PDF parsed but produced no text.")
                st.session_state.update({"doc_chunks": None, "faiss_index": None, "last_upload_type": None})
            else:
                emb = embed_any(chunks)
                index = create_pdf_index(emb)
                st.session_state.update({
                    "doc_chunks": chunks,
                    "faiss_index": index,
                    "last_upload_type": "pdf",
                })
        else:
            img = Image.open(bio).convert("RGB")
            raw_chunks = extract_ocr_chunks(img)
            chunks = []
            for c in raw_chunks:
                if isinstance(c, str): chunks.append(c.strip())
                elif isinstance(c, dict) and "text" in c: chunks.append(str(c["text"]).strip())
                elif c: chunks.append(str(c).strip())
            chunks = [c for c in chunks if c]
            if not chunks:
                st.error("OCR produced no text.")
                st.session_state.update({"doc_chunks": None, "faiss_index": None, "last_upload_type": None})
            else:
                emb = embed_any(chunks)
                index = create_ocr_index(emb)
                st.session_state.update({
                    "doc_chunks": chunks,
                    "faiss_index": index,
                    "last_upload_type": "image",
                })

        st.session_state["_UPLOADED_CACHE_KEY"] = cache_key
        st.session_state["_UPLOADED_NAME"] = uploaded_file.name
