import streamlit as st
from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from llm_response import generate_zephyr_answer

# Sayfa ayarları
st.set_page_config(page_title="📄 Chat with PDF - Zephyr", layout="wide")
st.markdown("<h2 style='text-align:center;'>💬 Chat with your PDF (Zephyr 7B)</h2>", unsafe_allow_html=True)

# Session state başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None
if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = None
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None

# Yardımcı: ChatGPT benzeri mesajları göster
def render_chat(user, bot):
    st.markdown(f"""
    <div style='background-color:#DCF8C6; padding:10px 15px; border-radius:10px; margin-bottom:5px; max-width:80%;'>
        <b>You:</b> {user}
    </div>
    <div style='background-color:#F1F0F0; padding:10px 15px; border-radius:10px; margin-bottom:20px; max-width:80%;'>
        <b>Assistant:</b> {bot}
    </div>
    """, unsafe_allow_html=True)

# PDF yükleme
uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("🔎 Processing PDF..."):
        chunks = extract_text_chunks(uploaded_file)
        embeddings = embed_chunks(chunks)
        index = create_faiss_index(embeddings)

        st.session_state.pdf_chunks = chunks
        st.session_state.pdf_embeddings = embeddings
        st.session_state.pdf_index = index
    st.success("✅ PDF is ready for chat!")

st.markdown("---")

# Kullanıcı girişi
user_input = st.text_input("💬 Ask a question or say something:")

if user_input:
    # History + context
    history = st.session_state.chat_history[-3:] if len(st.session_state.chat_history) > 3 else st.session_state.chat_history
    context = ""

    if st.session_state.pdf_chunks and st.session_state.pdf_index:
        context = search_similar_chunk(user_input, st.session_state.pdf_index, st.session_state.pdf_chunks)

    with st.spinner("🤖 Generating answer..."):
        answer = generate_zephyr_answer(context, user_input, history)

    # Geçmişe ekle
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Sohbet geçmişini yazdır
if st.session_state.chat_history:
    st.markdown("### 🗨️ Conversation", unsafe_allow_html=True)
    for chat in st.session_state.chat_history[::-1]:
        render_chat(chat["user"], chat["bot"])
