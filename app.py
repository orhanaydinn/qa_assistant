import streamlit as st
import time
import json
import csv
import os
import pandas as pd

from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from llm_response import generate_zephyr_answer
from image_gen import generate_image_from_prompt
from translation_utils import (
    smart_detect_language,
    translate_to_en,
    translate_from_en,
    extract_target_language_instruction,
)

translation_debug = True

st.set_page_config(page_title="AI Assistant – PDF + Image", layout="centered")

st.title("Chat with your PDF AI – Zephyr Enhanced")
st.markdown(
    """
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px; font-size: 0.9em; color: gray;'>
        Developed by <strong>Orhan Aydın</strong> – with web-enhanced answers
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    div.stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.75em 2em;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #e64545;
    }
    </style>
""", unsafe_allow_html=True)

mode = st.radio("Select Mode", ["Chat (PDF QA)", "Image Generator", "Performance Test", "User Simulation Test"], horizontal=True)

language_selection = st.selectbox("Select Language", ["English", "Turkish", "Auto"], key="language_option")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "image_history" not in st.session_state:
    st.session_state.image_history = []

# ------------------------------
# Chat Mode
# ------------------------------
if mode == "Chat (PDF QA)":
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            with st.spinner("Parsing PDF..."):
                chunks = extract_text_chunks(uploaded_file)
                embeddings = embed_chunks(chunks)
                index = create_faiss_index(embeddings)
                st.session_state.doc_chunks = chunks
                st.session_state.faiss_index = index
            st.success("PDF parsed and ready!", icon="✅")

    st.markdown("### Ask a question")
    user_input = st.text_input("Your question:", key="user_input")

    if st.button("Send", type="primary", use_container_width=True) and user_input:
        original_input = user_input

        user_lang = smart_detect_language(user_input)
        target_lang = extract_target_language_instruction(original_input)

        if st.session_state.language_option == "English":
            target_lang = "en"
        elif st.session_state.language_option == "Turkish":
            target_lang = "tr"
        else:
            target_lang = target_lang or user_lang

        if translation_debug:
            st.info(f"[DEBUG] Seçilen dil: **{st.session_state.language_option}**, Kullanıcı dili: **{user_lang}**, hedef: **{target_lang}**")

        if user_lang != "en" and target_lang == "en":
            user_input = translate_to_en(user_input, user_lang)
            if translation_debug:
                st.info(f"[DEBUG] Soru İngilizceye çevrildi: `{user_input}`")

        context = ""
        if st.session_state.doc_chunks and st.session_state.faiss_index is not None:
            similar_chunks = search_similar_chunk(
                user_input, st.session_state.faiss_index, st.session_state.doc_chunks
            )
            context = "\n".join(similar_chunks)

        answer, status_message_preview = generate_zephyr_answer(context, user_input, st.session_state.chat_history)

        with st.spinner(status_message_preview):
            pass

        answer_lang = smart_detect_language(answer)

        if translation_debug:
            st.info(f"[DEBUG] Yanıt dili: **{answer_lang}**")
            st.info(f"[DEBUG] Ham LLM yanıtı: {answer}")

        if target_lang != "en" and answer_lang != target_lang:
            if translation_debug:
                st.info(f"→ Cevap {answer_lang}, hedef {target_lang} → çeviriliyor.")
            answer = translate_from_en(answer, target_lang)

        st.session_state.chat_history.append({
            "user": original_input,
            "bot": answer
        })

    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for turn in st.session_state.chat_history[::-1]:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["bot"])

# ------------------------------
# Image Mode
# ------------------------------
elif mode == "Image Generator":
    st.markdown("### Describe the image you want to generate")

    prompt = st.text_input(
        "📝 Prompt:",
        placeholder="A futuristic cyberpunk city at night with neon lights",
        key="image_prompt"
    )

    if st.button("Generate Image", use_container_width=True) and prompt:
        original_prompt = prompt
        user_lang = smart_detect_language(prompt)

        if st.session_state.language_option == "English":
            target_lang = "en"
        elif st.session_state.language_option == "Turkish":
            target_lang = "tr"
        else:
            target_lang = user_lang

        if target_lang != "en":
            prompt = translate_to_en(prompt, target_lang)

        with st.spinner("Generating image..."):
            try:
                image = generate_image_from_prompt(prompt)
                st.session_state.image_history.append({
                    "prompt": original_prompt,
                    "image": image
                })
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.image_history:
        st.markdown("### Image History")
        for item in st.session_state.image_history[::-1]:
            with st.chat_message("user"):
                st.markdown(item["prompt"])
            with st.chat_message("assistant"):
                st.image(item["image"], use_container_width=True)

