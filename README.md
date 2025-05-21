## 📊 Toktam Khatibi's Research Dashboard

An interactive Streamlit dashboard that visualizes publications and research projects in medical AI, NLP, and healthcare data science. Now upgraded with a **Mixture-of-Experts Retrieval-Augmented Generation (MoE-RAG) Chatbot** for in-depth question answering over your personal research corpus.

---

### 🔗 Live App

👉 **[Try the Dashboard Online](https://toktamkhatibi-publicationsdashboard.streamlit.app/)**

---

### 📂 Key Features

- 🧾 Searchable, sortable table of scientific publications
- 📈 Visual analytics: trends by year, modality, type, and research area
- 📄 Direct links to full papers and external metadata
- 🤖 **Advanced Q&A Chatbot**: powered by RAG + multiple QA models + LLM-based reasoning

---

### 🚀 Quick Start

Install dependencies and launch the dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

### 🤖 MoE-RAG Chatbot: Intelligent Academic Q\&A

The chatbot uses your own papers to answer research questions through a **multi-expert pipeline**, ensuring high factual accuracy and explainability.

#### 💡 Architecture Overview

1. **Semantic Retrieval (RAG)**:

   * FAISS index over PDFs
   * Embeddings via `all-MiniLM-L6-v2` from SentenceTransformers

2. **Parallel QA Models (Experts)**:

   * DistilBERT (fast, lightweight)
   * RoBERTa (robust SQuAD2-trained)
   * BERT-large (high accuracy, context-sensitive)

3. **LLM-Based Answer Selection**:

   * Evaluates the three expert answers
   * Uses GPT-3.5 (or GPT-2/Hugging Face fallback) to synthesize or select the best answer
   * Cleans and finalizes the response for user output

4. **Intermediate Outputs Displayed**:

   * Transparent view of expert answers and final selection

---

### 🔄 Fallback Strategy

If GPT-3.5 is unavailable:

* ✅ Falls back to Hugging Face-hosted LLMs (no API key required)
* ✅ If still unavailable, switches to local GPT-2 inference
* ✅ RAG + QA pipelines remain fully functional offline

---

### 🛠️ Highlights

* 🔍 Semantic search over uploaded research papers
* 🧠 Robust chatbot using ensemble QA and LLM ranking
* 📊 Integrated visual and tabular analytics
* 🕵️ Transparent AI reasoning steps exposed to users
* 🧩 Modular, extensible codebase for future enhancements

---

### 🔍 Chatbot Summary

| Component      | Method/Model                             |
| -------------- | ---------------------------------------- |
| Retrieval      | FAISS + all-MiniLM-L6-v2 embeddings      |
| QA Experts     | DistilBERT, RoBERTa (SQuAD2), BERT-large |
| Final Selector | GPT-3.5-turbo / Hugging Face LLM / GPT-2 |
| Fallback Modes | Hugging Face APIs → Local GPT-2          |

---

### 📁 Project Structure

```
├── app.py                # Streamlit dashboard + chatbot frontend
├── chatbot.py            # Mixture-of-Experts RAG chatbot logic
├── requirements.txt      # Dependencies
├── papers/               # Folder with research PDFs
├── data/
│   └── publications.csv  # Metadata for publications
└── docs/
    └── chatbot.md        # (Optional) Extended technical docs
```

---

### 📖 Usage Overview

1. **Explore Publications**:

   * Search, sort, and filter your research output
   * Visualize insights across years, topics, and types

2. **Ask Research Questions**:

   * Type any question into the chatbot
   * Retrieves relevant text from your papers
   * Runs 3 QA models in parallel
   * Final answer selected or synthesized by an LLM

3. **Offline and Robust**:

   * No OpenAI key? No problem. System falls back automatically

---

### 📩 Contact

Questions or suggestions?
📧 **[toktamk@gmail.com](mailto:toktamk@gmail.com)**

---

### 💼 About

This project supports **academic self-discovery** by combining NLP, information retrieval, and interactive visualization. Built for researchers who want to organize, query, and present their work with intelligence.

---

### ⚙️ Tech Stack

* Python · Streamlit
* Hugging Face Transformers
* SentenceTransformers · FAISS
* OpenAI GPT-3.5 (optional)
* GPT-2 (local fallback)
* LangChain (PDF parsing, chunking)
* Matplotlib / Seaborn (visualizations)

---

⭐️ **Star this repo if you find it useful!**

```

---

Let me know if you’d like to generate a badge set (e.g., license, Python version, model support), include deployment steps (e.g., Docker), or turn this into a `README.ipynb` with live cells.

