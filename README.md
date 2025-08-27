# ChatRAG Assistant

This project is a multimodal AI assistant that integrates Retrieval-Augmented Generation (RAG), OCR, PDF parsing, web search, and image generation into a unified conversational interface. It supports complex queries using both text and visual inputs.

---

## Project Overview

![Image](https://github.com/user-attachments/assets/0209eb0a-aac3-4ebc-b4b1-3677446c2e64)

This assistant is designed to process and respond using a variety of data sources:

- Custom documents (PDFs)
- Images (with text extraction)
- Real-time web results
- LLM-powered responses
- Image generation from prompts

---

## Dataset & Input Types

This project works with:
- Uploaded **PDF files**
- Uploaded **images** (JPG, PNG, etc.) — processed via **OCR**
- User queries that can optionally trigger **web searches**

These inputs are processed and embedded using FAISS for retrieval in context.

---

## Project Flow

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

## Modules & Structure

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

## Output & Visualization

- Console-based chat or UI-based interaction
- Printout of selected context before answering
- Generated images displayed inline
- Future: Confusion matrix / eval for retrieval precision

---

## Performance & Optimization

- Efficient FAISS-based retrieval for real-time responses
- Modular structure for easy scaling or switching components (e.g. Pinecone, Claude, etc.)

---

## Technologies Used

- **Python 3.10**
- **FAISS** (Facebook AI Similarity Search) – for vector-based retrieval in RAG
- **sentence-transformers** (Hugging Face) – for text embeddings
- **EasyOCR** – for extracting text from images
- **Pillow (PIL)** – for image loading and processing
- **Streamlit** – for building the user interface
- **HuggingFaceH4/zephyr-7b-beta** – as the LLM for generating text responses
- **stabilityai/stable-diffusion-xl-base-1.0** – for generating images from prompts

---

## Author

**Orhan Aydin**  
*MSc Data Science and Artificial Intelligence*

---
