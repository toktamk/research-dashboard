# 🧠 Research Q&A Chatbot Architecture

This document provides an updated explanation of the **Research Q&A Chatbot** integrated into the Research Dashboard. The chatbot combines **retrieval-augmented generation (RAG)** with a **mixture-of-experts (MoE)** question-answering strategy and optional LLM-based refinement to deliver evidence-based answers from your research publications.

---

## 1. Overview

The chatbot helps users query your personal archive of scientific papers by using a multi-model approach that enhances reliability and precision. It retrieves relevant content from your documents, queries multiple QA models, and selects or refines the best answer using LLM reasoning or voting.

---

## 2. Core Components

### 2.1 Mixture of QA Experts

- **Three QA pipelines** are used in parallel:
  - `distilbert-base-cased-distilled-squad`
  - `deepset/roberta-base-squad2`
  - `bert-large-uncased-whole-word-masking-finetuned-squad`
- Each model answers independently based on the same retrieved context.

### 2.2 Retrieval System

- Uses **FAISS** (Facebook AI Similarity Search) for efficient dense vector search.
- PDF files are loaded and chunked using `LangChain` utilities.
- Embeddings are generated via `all-MiniLM-L6-v2` (from SentenceTransformers) to enable semantic search.

### 2.3 LLM Refinement

- A **refinement step** is used to choose or summarize answers:
  - First, a **majority voting** mechanism checks for agreement among experts.
  - If no consensus, OpenAI’s **`gpt-3.5-turbo`** or a **local GPT-2 model** synthesizes a final, concise answer.
- Optional cleanup removes boilerplate or redundant phrasing.

---

## 3. Updated Answer Generation Pipeline

```text
                   ┌────────────────────────┐
                   │      User Query        │
                   └──────────┬─────────────┘
                              │
                ┌────────────▼────────────┐
                │   Semantic Search via   │
                │    FAISS + Embeddings   │
                └────────────┬────────────┘
                             │
                 ┌──────────▼───────────┐
                 │ Top-k Context Chunks │
                 └────┬──────┬──────────┘
                      │      │
        ┌─────────────▼──────▼─────────────┐
        │    QA Expert Models (BERTs)      │
        └────┬──────────┬──────────┬───────┘
             │          │          │
      ┌──────▼────┐┌────▼────┐┌────▼────┐
      │ Expert 1  ││Expert 2 ││Expert 3 │
      └──────┬────┘└────┬────┘└────┬────┘
             │          │          │
        ┌────▼──────────▼──────────▼─────┐
        │     Majority Vote or LLM       │
        └────────────┬───────────────────┘
                     │
           ┌─────────▼─────────┐
           │  Final Answer     │
           └───────────────────┘
````

---

## 4. Fallback Mechanism

If the OpenAI API is unavailable (due to quota, network issues, or lack of credentials), the chatbot automatically falls back to a **local GPT-2 pipeline** for final answer synthesis, ensuring uninterrupted operation.

---

## 5. Implementation Details

* All logic resides in `chatbot.py`, with optional frontend integration via `app.py`.
* PDFs are loaded from the `papers/` folder and indexed using FAISS.
* Embedding and inference are done locally for performance and privacy.
* Cleanup of repeated phrases is handled with custom regex-based logic.

---

## 6. Usage

### ⚙️ Setup

Install required packages:

```bash
pip install torch transformers faiss-cpu sentence-transformers langchain pymupdf streamlit openai
```

### 🔑 OpenAI API Key

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Or define it in `st.secrets["OPENAI_API_KEY"]` for Streamlit Cloud use.

### ▶️ Running the Bot

```bash
python chatbot.py
```

To launch the Streamlit interface:

```bash
streamlit run app.py
```

---

## 7. Design Benefits

* **Robustness**: Using multiple QA models improves fault tolerance.
* **Accuracy**: Combining retrieved document knowledge with large LLM reasoning ensures more precise answers.
* **Transparency**: Displays intermediate outputs for debugging and educational value.

---

## 8. Future Improvements

* Support newer and larger open-source LLMs (e.g., Mistral, Phi-2) for local refinement.
* Extend support to non-PDF sources (e.g., LaTeX, DOCX).
* Allow user scoring of answers to fine-tune expert weighting.

---

## 9. Contact

For questions, issues, or suggestions, please contact:

**Toktam Khatibi**
📧 [toktamk@gmail.com](mailto:toktamk@gmail.com)

---

*Document last updated: 2025-05-21*

