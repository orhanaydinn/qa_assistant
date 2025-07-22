# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:17:18 2025

@author: Orhan
"""

import streamlit as st
from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from model_utils import generate_zephyr_answer

st.set_page_config(page_title="Zephyr PDF QA", layout="wide")
st.title("📄💬 Chat with your PDF (Zephyr 7B Beta)")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None

# Upload PDF
uploaded_file = st.file_uploader("📎 Upload your PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text and computing embeddings..."):
        chunks = extract_text_chunks(uploaded_file)
        embeddings = embed_chunks(chunks)
        index = create_faiss_index(embeddings)
        st.session_state.pdf_chunks = chunks
        st.session_state.pdf_index = index
    st.success("✅ PDF loaded and ready to chat!")

st.divider()

# User input
user_input = st.text_input("💬 Ask a question or say something:")

if user_input:
    context = ""
    if st.session_state.pdf_chunks and st.session_state.pdf_index:
        best_chunk = search_similar_chunk(user_input, st.session_state.pdf_index, st.session_state.pdf_chunks)
        context = best_chunk

    # Prompt + context + history
    history = st.session_state.chat_history[-3:]
    answer = generate_zephyr_answer(context, user_input, history)

    # Update history
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display conversation
if st.session_state.chat_history:
    st.subheader("🗨️ Conversation")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
