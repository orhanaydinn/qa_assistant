# 🤖 ChatRAG Assistant

ChatRAG Assistant is a powerful multimodal AI assistant that combines the capabilities of Retrieval-Augmented Generation (RAG), OCR, PDF parsing, web search, and image generation into a unified conversational system — much like ChatGPT but with expanded input and retrieval capabilities.

---

## 🧠 Project Overview

![Image](https://github.com/user-attachments/assets/a0f45bd9-eda5-4cf3-b241-d7ae09e1c5d1)

This assistant is designed to process and respond using a variety of data sources:

- Custom documents (PDFs)
- Images (with text extraction)
- Real-time web results
- LLM-powered responses
- Image generation from prompts

---

## 🗃️ Dataset & Input Types

This project works with:
- Uploaded **PDF files**
- Uploaded **images** (JPG, PNG, etc.) — processed via **OCR**
- User queries that can optionally trigger **web searches**

These inputs are processed and embedded using FAISS for retrieval in context.

---

## 🔄 Project Flow

### Step 1: Input Collection
- Accepts input from PDFs, images, or user chat.
- Optionally performs web search for relevant context.

### Step 2: Preprocessing & Parsing
- PDF and image content is parsed and cleaned.
- OCR is applied to extract text from images.
- Embeddings are generated using language models.

### Step 3: Retrieval & Response
- FAISS is used to retrieve relevant context.
- LLM generates a final response based on context and prompt.

### Step 4: Image Generation (Optional)
- When prompted, the assistant can create images using a generative model.

---

## 🧪 Modules & Structure

```
app/
├── main_ui.py              # UI & interaction flow
├── main_pipeline.py        # Core orchestration logic
├── main_data.py            # Configuration & shared data logic
├── faiss_search.py         # FAISS vector search implementation
├── pdf_parser.py           # PDF reading and text extraction
├── ocr_utils.py            # OCR helper functions
├── ocr_faiss.py            # OCR + FAISS integration
├── image_gen.py            # Image generation logic
├── llm_response.py         # LLM response generation
├── rag_dataset_qa.py       # Dataset formatting for RAG
```

---

## 📈 Output & Visualization

- Console-based chat or UI-based interaction
- Printout of selected context before answering
- Generated images displayed inline
- Future: Confusion matrix / eval for retrieval precision

---

## ⏱️ Performance & Optimization

- Efficient FAISS-based retrieval for real-time responses
- Modular structure for easy scaling or switching components (e.g. Pinecone, Claude, etc.)

---

## 🛠️ Technologies Used

- Python 3.8+
- OpenAI (LLM + Image Gen)
- FAISS (Facebook AI Similarity Search)
- PyMuPDF (PDF parsing)
- Tesseract OCR
- Streamlit / CLI (for UI)
- Requests / SerpAPI (for web search)

---

## ▶️ How to Run

1. Clone the repo:

```bash
git clone https://orhanaydin-ai-asistant-v1.streamlit.app/
cd chatrag-assistant
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the assistant:

```bash
python app/main_ui.py
```

---

## 👤 Author

**Orhan Aydin**  
*MSc Data Science and Artificial Intelligence*

---
