# app.py
import streamlit as st
import pandas as pd

st.title("📊 Toktam Khatibi's Research Dashboard")

df = pd.read_csv("publications.csv")

st.write("### 🧠 My Research Publications")
st.dataframe(df)

st.write("### 📅 Publications per Year")
st.bar_chart(df['Year'].value_counts().sort_index())
