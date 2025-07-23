import streamlit as st
from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from llm_response import generate_zephyr_answer

st.set_page_config(page_title="PDF QA Chatbot", layout="centered")

# Başlık ve açıklama
st.title("Chat with your PDF AI – Zephyr Enhanced")
st.markdown(
    """
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px; font-size: 0.9em; color: gray;'>
        Developed by <strong>Orhan Aydın</strong> – with web-enhanced answers
    </div>
    """,
    unsafe_allow_html=True
)

# Oturum içi değişkenler
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# PDF Yükleme
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

# Kullanıcı girişi
st.markdown("### Ask a question")
user_input = st.text_input("Your question:", key="user_input")

# Gönder butonu
if st.button("Send", type="primary", use_container_width=True) and user_input:
    context = ""

    if st.session_state.doc_chunks and st.session_state.faiss_index is not None:
        similar_chunks = search_similar_chunk(
            user_input, st.session_state.faiss_index, st.session_state.doc_chunks
        )
        context = "\n".join(similar_chunks)

    # Önce status_message alınır
    _, status_message_preview = generate_zephyr_answer(context, user_input, st.session_state.chat_history, preview=True)

    # Ardından gerçek yanıt alınır
    with st.spinner(status_message_preview):
        answer, _ = generate_zephyr_answer(context, user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": answer
        })

# Sohbet geçmişi
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for turn in st.session_state.chat_history[::-1]:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["bot"])
