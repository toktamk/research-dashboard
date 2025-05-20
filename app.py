import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chatbot import load_documents, build_vectorstore, get_answer_with_steps

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

# QA Bot
st.subheader("🤖 Ask Me About My Research")

with st.spinner("Setting up the chatbot (embedding & indexing)..."):
    docs = load_documents()
    texts, index, embeddings = build_vectorstore(docs)
    st.success("Chatbot is ready!")

question = st.text_input("Ask a question (e.g., 'Do you work with deep learning on cancer imaging?')")

if question:
    with st.spinner("Thinking..."):
        st.subheader("The first option is using open-ai up-to-date LLMs. But if our open-ai API Key does not work, we will return to local traditional LLMs with much less accurate answers")
        answers = get_answer_with_steps(question, texts, index, embeddings)
        st.subheader("LLM used for this analysis is: ")
        st.info(answers["llm"])
        st.subheader("🔍 Step 1: General LLM Answer")
        st.info(answers["answer1_general_llm"])
        st.subheader("📚 Step 2: RAG Answer (From Your Papers)")
        st.info(answers["answer2_rag"])
        st.subheader("✅ Step 3: Final Merged Answer")
        st.success(answers["final_merged_answer"])
