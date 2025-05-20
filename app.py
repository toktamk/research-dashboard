import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chatbot import load_documents, build_vectorstore, get_answer_with_steps

# ... (rest of your dashboard code)

with st.spinner("Setting up the chatbot (embedding & indexing)..."):
    docs = load_documents()
    texts, index, embeddings = build_vectorstore(docs)
    st.success("Chatbot is ready!")

question = st.text_input("Ask a question (e.g., 'Do you work with deep learning on cancer imaging?')")

if question:
    with st.spinner("Thinking..."):
        answers = get_answer_with_steps(question, texts, index, embeddings)
        st.subheader("🔍 Step 1: General LLM Answer")
        st.info(answers["answer1_general_llm"])
        st.subheader("📚 Step 2: RAG Answer (From Your Papers)")
        st.info(answers["answer2_rag"])
        st.subheader("✅ Step 3: Final Merged Answer")
        st.success(answers["final_merged_answer"])
