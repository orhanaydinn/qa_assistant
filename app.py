import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from main_ui import HERO_HTML, CHAT_CSS, TOGGLE_JS, FOOTER_HTML, render_history
from main_data import process_upload
from main_pipeline import preload_small_rag, handle_user_message

st.set_page_config(page_title="AI Assistant – RAG/Web/ImageGen", layout="centered")

# --- session_state defaults (küçük ve lokal tutuyoruz) ---
DEFAULTS = {
    "chat_history": [],           # [{role, content, sources?, uid?, image_b64?, mime?}]
    "doc_chunks": None,
    "faiss_index": None,
    "_DATASET_LOADED": False,
    "_DATASET_INDEX": None,
    "_DATASET_TEXTS": None,
    "temp_input": "",
    "pending_question": None,
    "clear_input_flag": False,
    "last_upload_type": None,     # "pdf" | "image"
    "_UPLOADED_CACHE_KEY": None,
    "_UPLOADED_NAME": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# --- UI başlık/tema ---
st.markdown(HERO_HTML, unsafe_allow_html=True)
st.markdown(CHAT_CSS, unsafe_allow_html=True)
st.markdown(TOGGLE_JS, unsafe_allow_html=True)

# --- Dil seçimi ve ana alanlar ---
language_selection = st.selectbox("Select Language", ["English", "Turkish", "Auto"], key="language_option")
chat_area = st.container()
input_area = st.container()

# --- küçük dataset RAG preload ---
preload_small_rag()

# --- Sidebar: dosya yükleme ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        process_upload(uploaded_file)
    if st.session_state.get("doc_chunks"):
        st.info(
            f"Loaded {len(st.session_state['doc_chunks'])} doc chunks "
            f"(Source: {st.session_state.get('_UPLOADED_NAME')})"
        )
        st.success(f"File parsed and ready! ✅  (Focus: {st.session_state.get('last_upload_type')})")

# --- Input handling ---
def _commit_message():
    msg = st.session_state.temp_input.strip()
    if msg:
        st.session_state.chat_history.append({"role": "user", "content": msg})
        st.session_state.pending_question = msg

if st.session_state.get("clear_input_flag"):
    st.session_state["temp_input"] = ""
    st.session_state["clear_input_flag"] = False

with input_area:
    c1, c2 = st.columns([12, 1])
    with c1:
        st.text_input(
            "Type your question...",
            key="temp_input",
            label_visibility="collapsed",
            placeholder="Type your question...",
            on_change=_commit_message,
        )
    with c2:
        st.button("➤", on_click=_commit_message)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)

# --- Chat alanı: geçmiş + yeni mesajı işleme ---
with chat_area:
    render_history()
    if st.session_state.pending_question:
        handle_user_message(st.session_state.pending_question, language_selection)
        st.session_state.pending_question = None
        st.session_state.clear_input_flag = True
