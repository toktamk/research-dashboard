import os
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional OpenAI import
try:
    import openai
    import streamlit as st
    OPENAI_AVAILABLE = True
    # Load API key if available
    try:
        openai.api_key = st.secrets["openai_api_key"]
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    OPENAI_AVAILABLE = False

# Local models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm_general = pipeline("text-generation", model="gpt2")
llm_merge = pipeline("text-generation", model="gpt2")

def load_documents():
    docs = []
    for file in os.listdir("papers"):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join("papers", file))
            docs.extend(loader.load())
    return docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return texts, index, embeddings

# --- OpenAI functions ---
def ask_openai(prompt, max_tokens=300, temperature=0.5):
    if not OPENAI_AVAILABLE or not openai.api_key:
        raise RuntimeError("OpenAI API not available")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# --- Local fallback functions ---
def llm_generate_answer(prompt):
    result = llm_general(prompt, max_new_tokens=200, do_sample=True)
    return result[0]['generated_text'].strip()

def merge_answers_with_llm(question, answer1, answer2):
    merge_prompt = (
        f"Question: {question}\n"
        f"Answer 1: {answer1}\n"
        f"Answer 2: {answer2}\n"
        f"Combine both answers to produce a more accurate, complete, and reasoned response:"
    )
    result = llm_merge(merge_prompt, max_new_tokens=200, do_sample=True)
    return result[0]['generated_text'].strip()

# --- Unified answer function with fallback ---
def get_answer_with_steps(question, texts, index, embeddings, top_k=3, token_limit=1500):
    # Try OpenAI path first
    try:
        if OPENAI_AVAILABLE and openai.api_key:
            # Step 1: General LLM response
            prompt_general = f"Answer this research-related question concisely:\n{question}"
            answer1 = ask_openai(prompt_general, max_tokens=200)
            # Step 2: RAG - retrieve context using vector search
            q_embedding = embedder.encode([question])
            distances, indices = index.search(np.array(q_embedding), top_k)
            retrieved = " ".join([texts[i] for i in indices[0]])[:token_limit]
            answer2 = qa_pipeline(question=question, context=retrieved)['answer']
            # Step 3: Merge both answers
            merge_prompt = (
                f"Question: {question}\n"
                f"Answer from general model: {answer1}\n"
                f"Answer from research context: {answer2}\n\n"
                f"Based on both, generate a short, clear, and accurate answer:"
            )
            final_answer = ask_openai(merge_prompt, max_tokens=250)
            return {
                "answer1_general_llm": answer1,
                "answer2_rag": answer2,
                "final_merged_answer": final_answer
            }
        else:
            raise RuntimeError("OpenAI not available")
    except Exception as e:
        # Fallback to local pipeline
        prompt1 = f"Answer the following question:\n{question}"
        answer1 = llm_generate_answer(prompt1)
        q_embedding = embedder.encode([question])
        _, I = index.search(np.array(q_embedding), top_k)
        relevant_context = " ".join([texts[i] for i in I[0]])
        answer2_result = qa_pipeline(question=question, context=relevant_context)
        answer2 = answer2_result['answer']
        final_answer = merge_answers_with_llm(question, answer1, answer2)
        return {
            "answer1_general_llm": answer1,
            "answer2_rag": answer2,
            "final_merged_answer": final_answer
        }
