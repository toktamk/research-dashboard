# 📊 Toktam Khatibi's Research Dashboard

An interactive Streamlit dashboard that visualizes my publications and research projects in medical AI, NLP, and healthcare data science.

🔗 **Live App:** [Research Dashboard Web Application](https://share.streamlit.io/app/toktamkhatibi-publicationsdashboard/)

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
- 📚 Interactive publication explorer with filtering by topic, year, and type  
- 📈 Visual analytics: publications per year, modality, research domain, and more  
- 🤖 **Smart Q&A Chatbot**: answers questions using multi-step LLM + RAG architecture  
- 🧠 Semantic search over your own papers (PDFs) with fallback to local models if OpenAI is unavailable  
- 📄 Transparent reasoning pipeline with intermediate results for traceability 
- Multi-source answer generation and reasoning combining general language understanding and domain-specific retrieval
- Semantic search over my own publications (PDF-based) for evidence-backed answers
- Streamlit-based interactive UI for easy access and exploration
- Displays intermediate reasoning steps for transparency: general LLM answer, RAG answer, and final merged answer
- Robust fallback mechanism to local models when OpenAI is not accessible

---

## 🔍 Chatbot Architecture (Summary)

This project includes a **Research Q&A Chatbot** using a **Retrieval-Augmented Generation (RAG)** approach for high-fidelity, context-aware answers from your own scientific work.

### 🧠 Core Components:
- **LLM**: `gpt-3.5-turbo` for final answer synthesis  
- **Retrieval**: FAISS vector store + `text-embedding-ada-002` embeddings  
- **Fallback**: Seamless transition to local GPT-2-based pipeline when OpenAI is unavailable  

### 🧬 Workflow Overview: 
[Workflow of Chatbot](process.png)

```bash

    A[User Query] --> B[Embed Query];
    B --> C[Search FAISS Vector Store];
    C --> D[Retrieve Top-k Documents];
    D --> E[Combine Context + Query];
    E --> F[LLM (gpt-3.5-turbo)];
    F --> G[Answer Output to Dashboard];
```
📖 Full Technical Details →

## 📂 Project Structure

```
├── app.py                # Streamlit UI with chatbot and dashboard
├── chatbot.py            # Unified chatbot logic with OpenAI & local fallback
├── requirements.txt      # Python dependencies
├── papers/               # Folder for PDF research papers
├── data/
│   └── publications.csv  # Publication metadata for dashboard charts
└── docs/
    └── chatbot.md        # Full chatbot architecture explanation
```


---

## 📖 Usage Overview
### 1. Dashboard:

- Loads publication metadata (publications.csv)
- Displays interactive visualizations and searchable tables

### 2. Chatbot:

- Indexes uploaded PDFs using OpenAI embeddings and FAISS
- Answers questions using:
- A general LLM response
- A document-based RAG response
- A final merged answer combining both insights
- If OpenAI is unavailable, it falls back to a local GPT-2 pipeline
### 3. Transparency:

Each query displays its reasoning steps: general LLM → RAG-based answer → merged consensus


---

## 📬 Contact

For questions or contributions, please contact toktamk@gmail.com.

