# Research Q&A Chatbot Architecture

This document provides an in-depth explanation of the **Research Q&A Chatbot** integrated into the Research Dashboard. The chatbot combines **large language models (LLMs)** with **retrieval-augmented generation (RAG)** techniques to provide accurate, context-aware answers based on my scientific publications.

---

## 1. Overview

The chatbot answers user queries about my research by merging general language understanding with specific evidence retrieved from my own papers. It uses a **multi-step reasoning algorithm** that ensures both relevance and accuracy.

---

## 2. Core Components

### 2.1 Large Language Models (LLMs)

- The primary model for generating responses is OpenAI's **`gpt-3.5-turbo`**.  
- When OpenAI services are unavailable, the system falls back to a local pipeline based on **GPT-2** models, ensuring uninterrupted service.

### 2.2 Retrieval System

- Uses **FAISS** (Facebook AI Similarity Search) for efficient vector similarity search.  
- The documents (PDFs of my papers) are converted to embeddings using OpenAI's **`text-embedding-ada-002`**.  
- Given a query, the system searches the vector store to find the most relevant text chunks.

### 2.3 Question Answering (QA) Model

- A fine-tuned **DistilBERT** model performs context-aware QA on the retrieved text to extract precise answers.

---

## 3. Multi-step Reasoning Pipeline

The chatbot follows these steps when answering queries:

1. **General LLM Response:**  
   The query is sent to the LLM (preferably GPT-3.5) to generate an initial, broad answer.

2. **RAG-based Retrieval and QA:**  
   The query is embedded and used to search the FAISS vector store. The top relevant text chunks are extracted, and the QA model provides a focused answer from these documents.

3. **LLM Consensus and Merging:**  
   Both answers are combined using an LLM to produce a final, coherent, and comprehensive response.

---

## 4. Fallback Mechanism

- If the OpenAI API is unreachable (due to network errors, quota limits, or missing API keys), the chatbot **automatically switches** to a fully local inference pipeline.  
- The local pipeline uses GPT-2-based models for all reasoning steps and still provides a functional user experience without requiring any API keys.

---

## 5. Workflow Diagram

![Chatbot Architecture](chatbot-architecture.png)

---

## 6. Implementation Details

- The chatbot logic is encapsulated in `chatbot.py`.  
- The vector store is built and updated when new PDFs are added to the `papers/` folder.  
- Embeddings are cached to speed up repeated queries.

---

## 7. Usage Notes

- The chatbot interface is integrated into the Streamlit dashboard (`app.py`).  
- Intermediate reasoning steps (initial LLM answer, RAG answer, merged answer) are shown to enhance transparency.  
- Users can freely ask questions about the publications, and the chatbot provides evidence-backed answers.

---

## 8. Future Improvements

- Integrate larger local LLMs to improve fallback quality.  
- Support additional document types beyond PDFs.  
- Add user feedback loop to continuously improve answer accuracy.

---

## 9. Contact

For further questions or contributions, please reach out at **toktamk@gmail.com**.

---

*Document last updated: 2025-05-20*
