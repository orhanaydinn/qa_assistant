import streamlit as st
from PIL import Image
from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from llm_response import generate_zephyr_answer
from image_gen import generate_image_from_prompt

# Sayfa ayarları
st.set_page_config(page_title="AI Assistant – PDF + Image", layout="centered")

# Başlık
st.title("Chat with your PDF AI – Zephyr Enhanced")
st.markdown("""
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px; font-size: 0.9em; color: gray;'>
        Developed by <strong>Orhan Aydın</strong> – with web-enhanced answers
    </div>
""", unsafe_allow_html=True)

# Kırmızı buton stili
st.markdown("""
    <style>
    div.stButton > button:first-child {
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

    div.stButton > button:first-child:hover {
        background-color: #e64545;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Mod seçimi
mode = st.radio("Select Mode", ["Chat (PDF QA)", "Image Generator"], horizontal=True)

# Dosya yükleme (PDF)
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file and mode == "Chat (PDF QA)":
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Parsing PDF..."):
                chunks = extract_text_chunks(uploaded_file)
                embeddings = embed_chunks(chunks)
                index = create_faiss_index(embeddings)
                st.session_state.doc_chunks = chunks
                st.session_state.faiss_index = index
            st.success("PDF parsed and ready!", icon="✅")

# Session state başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "image_history" not in st.session_state:
    st.session_state.image_history = []

# ------------------------------
# Chat (PDF QA) Sekmesi
# ------------------------------
if mode == "Chat (PDF QA)":
    st.markdown("### Ask a question about the document")
    question = st.text_input("Your question:")

    if st.button("Send") and question:
        # Önce spinner mesajı alınır
        _, status_message = generate_zephyr_answer(
            context="",
            question=question,
            chat_history=st.session_state.chat_history,
            preview=True
        )

        with st.spinner(status_message):
            context = ""
            if st.session_state.faiss_index and st.session_state.doc_chunks:
                similar_chunks = search_similar_chunk(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.doc_chunks
                )
                context = "\n".join(similar_chunks)

            answer, _ = generate_zephyr_answer(
                context=context,
                question=question,
                chat_history=st.session_state.chat_history
            )

            st.session_state.chat_history.append({"user": question, "bot": answer})
            st.markdown(f"**Answer:** {answer}")

    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for turn in st.session_state.chat_history[::-1]:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["bot"])

# ------------------------------
# 🖼️ Image Generator Sekmesi
# ------------------------------
elif mode == "Image Generator":
    st.markdown("### Describe the image you want to generate")

    prompt = st.text_input("📝 Enter your prompt (e.g. 'a cat in watercolor style')")

    if st.button("Generate Image") and prompt:
        with st.spinner("Generating image..."):
            try:
                result = generate_image_from_prompt(prompt)

                # Geçmişe ekle
                st.session_state.image_history.append({
                    "prompt": prompt,
                    "image": result
                })

            except Exception as e:
                st.error(f"❌ Error generating image: {e}")

    # Görsel geçmişi (sadece burada gösterilir)
    if st.session_state.image_history:
        st.markdown("### Image History")
        for item in st.session_state.image_history[::-1]:
            with st.chat_message("user"):
                st.markdown(item["prompt"])
            with st.chat_message("assistant"):
                st.image(item["image"], use_column_width=True)
