import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chatbot import load_pdfs_from_folder as load_documents, build_faiss_index as build_vectorstore, get_answer_with_steps

st.set_page_config(page_title="Toktam Khatibi Research Dashboard", layout="wide")
st.title("📊 Toktam Khatibi's Research Dashboard")

# Load publications metadata
df = pd.read_csv("data/publications.csv", encoding="ISO-8859-1")

# Display publications table
st.subheader("🧠 Publications Table")
st.dataframe(df[['title', 'year', 'Data Modality', 'PublicationType', 'link']])

# Publications per year
st.subheader("📅 Publications Per Year")
year_counts = df['year'].value_counts().sort_index()
st.bar_chart(year_counts)

# Research topic frequencies
topic_cols = df.columns[2:20]
topic_counts = df[topic_cols].sum().sort_values(ascending=False)

st.subheader("📚 Research Topics Frequency")
st.bar_chart(topic_counts)

# Publication types
st.subheader("📄 Publication Types")
pub_type_counts = df['PublicationType'].value_counts()
st.bar_chart(pub_type_counts)

# Data modalities
st.subheader("🧪 Data Modalities Used")
modality_counts = df['Data Modality'].value_counts()
st.bar_chart(modality_counts)

# QA Bot Section
st.subheader("🤖 Ask Me About My Research")

with st.spinner("Setting up the chatbot (embedding & indexing)..."):
    docs = load_documents()
    texts, index, embeddings = build_vectorstore(docs)
    st.success("Chatbot is ready!")

question = st.text_input("Ask a question (e.g., 'Do you work with deep learning on cancer imaging?')")

if question:
    with st.spinner("Thinking..."):
        st.subheader("Mixture of Experts QA System")
        answers = get_answer_with_steps(question, texts, index, embeddings)

        st.subheader("🔍 Expert 1 Answer")
        st.info(answers["answer_expert1"])

        st.subheader("🔍 Expert 2 Answer")
        st.info(answers["answer_expert2"])

        st.subheader("🔍 Expert 3 Answer")
        st.info(answers["answer_expert3"])

        st.subheader("✅ Final Selected Answer")
        st.success(answers["final_moe_answer"])

        st.subheader("🧠 LLM Used")
        st.info(answers["llm"])
