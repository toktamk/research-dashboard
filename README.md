# 📊 Toktam Khatibi's Research Dashboard

An interactive Streamlit dashboard that visualizes my publications and research projects in medical AI, NLP, and healthcare data science.

🔗 **Live App:** Coming soon...

---

## 📂 Features

- Interactive table of publications with metadata
- Bar charts showing publications per year, research topics, publication types, and data modalities
- Easily expandable dashboard (e.g., filtering by topic, linking to full papers)
- **Robust Research Q\&A Chatbot** that uses a multi-step LLM + Retrieval-Augmented Generation (RAG) reasoning approach with automatic fallback

---

## 🚀 Run It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```


---

## 🤖 Research Q\&A Chatbot with LLM + RAG Consensus Reasoning and Automatic Fallback

This project features an advanced research-focused chatbot that answers questions about my scientific publications using a **three-step reasoning algorithm** combining large language models (LLMs) and retrieval over my own papers.

### 💡 How It Works

1. **General LLM Response**
Generates an initial answer using a base LLM. The system tries to use OpenAI's GPT-3.5 API first for high-quality responses.
2. **RAG-based Answer from My Publications**
Performs semantic vector search over uploaded PDFs to find relevant sections. A QA model (DistilBERT) then produces a context-aware answer based on retrieved text.
3. **LLM Consensus \& Merging**
Combines both answers into a final, accurate, and complete response using an LLM.

### 🔄 Automatic Fallback Logic

- If the OpenAI API is unavailable or fails (due to network issues, quota, or invalid API key), the chatbot **automatically switches** to fully local models (GPT-2 based pipelines) for all steps, ensuring uninterrupted functionality without requiring OpenAI credentials.

---

## 🛠️ Features

- Multi-source answer generation and reasoning combining general language understanding and domain-specific retrieval
- Semantic search over my own publications (PDF-based) for evidence-backed answers
- Streamlit-based interactive UI for easy access and exploration
- Displays intermediate reasoning steps for transparency: general LLM answer, RAG answer, and final merged answer
- Robust fallback mechanism to local models when OpenAI is not accessible

---

## 📂 Project Structure

```
├── app.py                # Streamlit UI with chatbot integration and fallback handling
├── chatbot.py            # Unified chatbot logic with OpenAI + local fallback
├── requirements.txt      # Python dependencies
├── papers/               # Folder to store PDF research papers
└── data/publications.csv # Metadata CSV used for dashboard visualizations
```


---

## 📖 Usage Overview

- The dashboard loads publication metadata and displays interactive tables and charts.
- The chatbot indexes my PDF papers using embeddings and FAISS vector search.
- When a question is asked, the chatbot attempts to answer using OpenAI GPT-3.5. If unsuccessful, it falls back to local models seamlessly.
- Answers are presented in three steps for clarity and trustworthiness.

---

## 📬 Contact

For questions or contributions, please contact toktamk@gmail.com.

