import streamlit as st
from pdf_parser import extract_text_chunks
from embedder import embed_chunks
from faiss_search import create_faiss_index, search_similar_chunk
from llm_response import generate_zephyr_answer
from image_gen import generate_image_from_prompt

st.set_page_config(page_title="AI Assistant – PDF + Image", layout="centered")

# Başlık
st.title("Chat with your PDF AI – Zephyr Enhanced")
st.markdown(
    """
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px; font-size: 0.9em; color: gray;'>
        Developed by <strong>Orhan Aydın</strong> – with web-enhanced answers
    </div>
    """,
    unsafe_allow_html=True
)

#  Kırmızı buton stili
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

# Mod seçici
mode = st.radio("Select Mode", ["Chat (PDF QA)", "Image Generator"], horizontal=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "image_history" not in st.session_state:
    st.session_state.image_history = []

# Dosya yükleme
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and mode == "Chat (PDF QA)":
        with st.spinner("Parsing PDF..."):
            chunks = extract_text_chunks(uploaded_file)
            embeddings = embed_chunks(chunks)
            index = create_faiss_index(embeddings)

            st.session_state.doc_chunks = chunks
            st.session_state.faiss_index = index

        st.success("PDF parsed and ready!", icon="✅")

# ------------------------------
# 📄 Chat (PDF QA)
# ------------------------------
if mode == "Chat (PDF QA)":
    st.markdown("### Ask a question")
    user_input = st.text_input("Your question:", key="user_input")

    if st.button("Send", type="primary", use_container_width=True) and user_input:
        context = ""

        if st.session_state.doc_chunks and st.session_state.faiss_index is not None:
            similar_chunks = search_similar_chunk(
                user_input, st.session_state.faiss_index, st.session_state.doc_chunks
            )
            context = "\n".join(similar_chunks)

        # Spinner mesajı
        _, status_message_preview = generate_zephyr_answer(context, user_input, st.session_state.chat_history, preview=True)

        with st.spinner(status_message_preview):
            answer, _ = generate_zephyr_answer(context, user_input, st.session_state.chat_history)
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": answer
            })

    # Geçmiş
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for turn in st.session_state.chat_history[::-1]:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["bot"])

# ------------------------------
# 🖼️ Image Generator
# ------------------------------
elif mode == "Image Generator":
    st.markdown("### Describe the image you want to generate")

    # 🎯 Prompt + örnek içerik
    prompt = st.text_input(
        "📝 Prompt:",
        placeholder="A futuristic cyberpunk city at night with neon lights",
        key="image_prompt"
    )

    if st.button("Generate Image", use_container_width=True) and prompt:
        with st.spinner("Generating image..."):
            try:
                image = generate_image_from_prompt(prompt)

                st.session_state.image_history.append({
                    "prompt": prompt,
                    "image": image
                })

            except Exception as e:
                st.error(f"Error: {e}")

    # Görsel geçmişi
    if st.session_state.image_history:
        st.markdown("### Image History")
        for item in st.session_state.image_history[::-1]:
            with st.chat_message("user"):
                st.markdown(item["prompt"])
            with st.chat_message("assistant"):
                st.image(item["image"], use_column_width=True)
