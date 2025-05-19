# 📊 Toktam Khatibi's Research Dashboard

An interactive Streamlit dashboard that visualizes my publications and research projects in medical AI, NLP, and healthcare data science.

🔗 **Live App:** Coming soon...

## 📂 Features
- Interactive table of publications
- Bar chart of publications per year
- Easily expandable (e.g., filter by topic, link to full papers)
- chatbot application that uses a multi-step LLM + RAG reasoning approach

## 🚀 Run It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🤖 Research Q&A Chatbot with LLM + RAG Consensus Reasoning

This project is an advanced research-focused chatbot that answers questions about my scientific publications using a **multi-step reasoning algorithm** combining general large language models (LLMs) and Retrieval-Augmented Generation (RAG) over my own papers.

---

### 💡 How It Works

The chatbot answers each user question using a **three-step algorithm**:

1. **General LLM Response**  
   A base LLM (e.g., GPT-2) is used to generate an initial answer purely from language patterns.

2. **RAG-based Answer from My Publications**  
   The same question is answered again using a RAG approach. A semantic vector search retrieves the most relevant sections from your uploaded PDFs, and a QA model (DistilBERT) generates a context-aware answer.

3. **LLM Consensus & Merging**  
   A third prompt is sent to the LLM, asking it to combine both answers (from Step 1 and 2) into a **final, accurate, and complete answer**.

---

### 🛠️ Features

- Multi-source answer generation and reasoning
- Semantic search over my own publications (PDF-based)
- Streamlit-based interactive UI
- Intermediate reasoning trace: displays **all three steps**

---

### 📂 Project Structure
├── app.py # Streamlit UI

├── chatbot.py # LLM + RAG + merging logic

├── requirements.txt # Dependencies

├── papers/ # Folder to store PDF research papers

└── data/publications.csv # Metadata CSV used for the dashboard
