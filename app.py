
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chatbot import load_documents, build_vectorstore, get_answer

st.set_page_config(page_title="Toktam Khatibi Research Dashboard", layout="wide")
st.title("📊 Toktam Khatibi's Research Dashboard")

# Load data
df = pd.read_csv("data/publications.csv", encoding="ISO-8859-1")

# Main display
st.subheader("🧠 Publications Table")
st.dataframe(df[['title', 'year', 'Data Modality', 'PublicationType', 'link']])

# 1. Publications per Year
st.subheader("📅 Publications Per Year")
year_counts = df['year'].value_counts().sort_index()
st.bar_chart(year_counts)

# 2. Research Topic Frequencies
topic_cols = df.columns[2:20]  # All binary topic columns
topic_counts = df[topic_cols].sum().sort_values(ascending=False)

st.subheader("📚 Research Topics Frequency")
st.bar_chart(topic_counts)

# 3. Publication Types
st.subheader("📄 Publication Types")
pub_type_counts = df['PublicationType'].value_counts()
st.bar_chart(pub_type_counts)

# 4. Data Modalities
st.subheader("🧪 Data Modalities Used")
modality_counts = df['Data Modality'].value_counts()
st.bar_chart(modality_counts)

# 5. QA Bot
st.subheader("🤖 Ask Me About My Research")

with st.spinner("Setting up the chatbot (embedding & indexing)..."):
    docs = load_documents()
    texts, index, embeddings = build_vectorstore(docs)
st.success("Chatbot is ready!")

question = st.text_input("Ask a question (e.g., 'Do you work with deep learning on cancer imaging?')")
if question:
    with st.spinner("Thinking..."):
        answer = get_answer(question, texts, index, embeddings)
        st.success(answer)
