## 📊 Toktam Khatibi's Research Dashboard

An interactive Streamlit dashboard visualizing publications and research projects in medical AI, NLP, and healthcare data science, now enhanced with a **Mixture-of-Experts RAG-based chatbot**.

### 🔗 Live App

**Research Dashboard Web Application** *(Link placeholder)*

---

### 📂 Features

* Interactive table of publications with searchable metadata
* Visual analytics: publications per year, data modalities, publication types, and topics
* Expandable dashboard with filtering, sorting, and linking to full papers
* **Advanced Research Q\&A Chatbot** powered by a multi-expert LLM + Retrieval-Augmented Generation (RAG) pipeline with fallback to local models

---

### 🚀 Run It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

### 🤖 Mixture-of-Experts Chatbot (RAG + LLM + Fallback)

This project features a next-generation research Q\&A chatbot that uses **multiple QA models** over your own papers and synthesizes the best response via a large language model (LLM).

#### 💡 How It Works

1. **RAG-Based Answers by Experts**: Retrieves relevant document chunks and runs question-answering with **three expert QA models**:

   * DistilBERT
   * RoBERTa (deepset/roberta-base-squad2)
   * BERT (bert-large-uncased-whole-word-masking-finetuned-squad)
2. **LLM-Based Selection (Mixture-of-Experts)**: All three answers are fed into an LLM (GPT-3.5 or GPT-2) to **evaluate and select the best response** based on clarity, accuracy, and completeness.
3. **Transparent Output**: Shows all expert answers and the selected final answer.

---

### 🔄 Fallback Logic

If OpenAI GPT-3.5 is unavailable (due to API errors, lack of credentials, or network issues), the chatbot:

* Switches to local GPT-2 for final answer generation
* Continues using FAISS + SentenceTransformer + local QA pipelines

---

### 🛠️ Project Highlights

* 📚 Semantic search over uploaded PDFs using FAISS and sentence embeddings
* 🤖 Robust chatbot using multi-model QA and LLM refinement
* 📊 Visual and tabular exploration of your research profile
* 🕵️ Transparent reasoning steps displayed to the user

---

### 🔍 Chatbot Architecture Summary

* **Retrieval**: FAISS + `all-MiniLM-L6-v2` sentence embeddings
* **QA Experts**: 3 QA pipelines (DistilBERT, RoBERTa, BERT)
* **Final Answer Selector**: OpenAI GPT-3.5-turbo or local GPT-2
* **Fallback**: Fully local model path when OpenAI is not available

---

### 📄 Project Structure

```
├── app.py                # Streamlit UI for dashboard + chatbot
├── chatbot.py            # Full MoE RAG chatbot logic
├── requirements.txt      # Python dependencies
├── papers/               # Folder containing PDFs of publications
├── data/
│   └── publications.csv  # Metadata file for the dashboard
└── docs/
    └── chatbot.md        # (Optional) Extended architecture explanation
```

---

### 📖 Usage Overview

1. **Dashboard**:

   * Loads and displays research publication data
   * Includes interactive charts and tables
2. **Chatbot**:

   * Uses FAISS + embeddings to retrieve relevant document context
   * Runs 3 QA models over context (DistilBERT, RoBERTa, BERT)
   * LLM ranks and selects best answer
   * Displays all intermediate steps
3. **Fallback**:

   * Seamless use of local models if OpenAI is not available

---

### 📩 Contact

For questions or contributions, please contact: **[toktamk@gmail.com](mailto:toktamk@gmail.com)**

---

### 💼 About

Interactive dashboard for exploring Toktam Khatibi's research projects and publications with a powerful QA assistant built for academic inquiry.

---

### 📊 Tech Stack

* Python
* Streamlit
* Transformers (Hugging Face)
* SentenceTransformers
* FAISS
* OpenAI GPT-3.5 (optional)
* GPT-2 (fallback)

---

**Star this repo if you find it useful!**
