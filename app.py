import streamlit as st
from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from llm_response import generate_zephyr_answer

# Sayfa başlığı ve tema
st.set_page_config(page_title="Chat with PDF + AI", layout="wide")
st.markdown("<h2 style='text-align:center;'>Chat with your PDF AI – Zephyr Enhanced</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>Developed by <strong>Orhan Aydin</strong> – with web-enhanced answers</p>",
    unsafe_allow_html=True
)

# Session state başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None
if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = None
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None

# Yardımcı: Chat baloncukları
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
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF..."):
        chunks = extract_text_chunks(uploaded_file)
        embeddings = embed_chunks(chunks)
        index = create_faiss_index(embeddings)

        st.session_state.pdf_chunks = chunks
        st.session_state.pdf_embeddings = embeddings
        st.session_state.pdf_index = index
    st.success("✅ PDF is ready for chat!")

st.markdown("---")

# Kullanıcı girişi
user_input = st.text_input("Ask a question or say something:")

if user_input:
    # Sohbet geçmişi
    history = st.session_state.chat_history[-3:] if len(st.session_state.chat_history) > 3 else st.session_state.chat_history
    context = ""

    # PDF içinden içerik eklenmesi gerekiyorsa
    if st.session_state.pdf_chunks and st.session_state.pdf_index:
        context = search_similar_chunk(user_input, st.session_state.pdf_index, st.session_state.pdf_chunks)

    # Yanıtı al ve durumu göster
    answer, status_message = generate_zephyr_answer(context, user_input, history)
    st.info(status_message)
    st.write(answer)

    # Sohbete kaydet
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Sohbet geçmişini yazdır
if st.session_state.chat_history:
    st.markdown("### Conversation", unsafe_allow_html=True)
    for chat in st.session_state.chat_history[::-1]:
        render_chat(chat["user"], chat["bot"])
