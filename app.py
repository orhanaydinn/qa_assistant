import streamlit as st
from pdf_parser import extract_text_chunks
from embedder import get_doc_embeddings
from faiss_search import find_similar_chunks
from llm_response import generate_zephyr_answer

st.set_page_config(page_title="PDF QA Chatbot", layout="centered")

# Başlık + Geliştirici Notu
st.title("📄 Chat with your PDF AI – Zephyr Enhanced")
st.markdown(
    """
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px; font-size: 0.9em; color: gray;'>
        Developed by <strong>Orhan Aydın</strong> – with web-enhanced answers
    </div>
    """,
    unsafe_allow_html=True
)

# Oturum değişkenleri
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

# PDF yükleme alanı
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Parsing PDF..."):
            chunks = extract_text_chunks(uploaded_file)
            embeddings = get_doc_embeddings(chunks)
            st.session_state.doc_chunks = chunks
            st.session_state.doc_embeddings = embeddings
        st.success("PDF parsed and ready!", icon="✅")

# Soru alanı
st.markdown("### Ask a question")
user_input = st.text_input("Your question:", key="user_input")

# Kullanıcı soru sorduğunda işlem
if st.button("Send", type="primary", use_container_width=True) and user_input:
    context = ""

    if st.session_state.doc_chunks and st.session_state.doc_embeddings is not None:
        similar_chunks = find_similar_chunks(user_input, st.session_state.doc_embeddings, st.session_state.doc_chunks)
        context = "\n".join(similar_chunks)

    # LLM çağrısı + durum mesajı
    answer, status_message = generate_zephyr_answer(context, user_input, st.session_state.chat_history)

    # Yükleniyor göstergesi
    with st.spinner(status_message):
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": answer
        })

# Chat geçmişini göster
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for turn in st.session_state.chat_history[::-1]:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["bot"])
