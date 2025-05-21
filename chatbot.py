import os
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional OpenAI and Streamlit import
try:
    import openai
    import streamlit as st
    OPENAI_AVAILABLE = True
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    OPENAI_AVAILABLE = False

# Initialize multiple RAG QA experts
qa_expert1 = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qa_expert2 = pipeline("question-answering", model="deepset/roberta-base-squad2")
qa_expert3 = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Embedding and LLMs
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Local fallback text generation LLM (lightweight, runs on CPU/GPU locally)
local_llm_models = ["distilgpt2", "gpt2"]
llm_refine = pipeline("text-generation", model=local_llm_models[0], device=-1)  # device=-1 to force CPU

def load_pdfs_from_folder(folder_path="papers"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
    return docs

def build_faiss_index(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return texts, index, embeddings

def ask_openai(prompt, max_tokens=300, temperature=0.5):
    if not OPENAI_AVAILABLE or not openai.api_key:
        raise RuntimeError("OpenAI API not available or API key missing.")

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful research assistant with expertise on Machine Learning, Deep Learning, Medical Image Analysis and Healthcare Data Analytics."
        )
    }

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        raise e

def refine_answers_with_llm(question, answers):
    prompt = (
        f"Question: {question}\n"
        f"Answer A: {answers[0]}\n"
        f"Answer B: {answers[1]}\n"
        f"Answer C: {answers[2]}\n"
        "Based on clarity, accuracy, and completeness, select the best answer and explain why briefly. "
        "Finally, provide a short and concise summary of the best answer."
    )
    if OPENAI_AVAILABLE and openai.api_key:
        try:
            return ask_openai(prompt, max_tokens=150, temperature=0.5)  # reduce max_tokens for brevity
        except Exception:
            print("Falling back to local LLM due to OpenAI failure.")

    generation = llm_refine(prompt, max_new_tokens=150, do_sample=False)  # deterministic output
    return generation[0]['generated_text'].strip()

def get_answer_with_steps(question, texts, index, embeddings, top_k=3, token_limit=1500):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, top_k)
    retrieved_text = " ".join([texts[i] for i in indices[0]])[:token_limit]

    # Get answers from all experts
    answer1 = qa_expert1(question=question, context=retrieved_text)['answer']
    answer2 = qa_expert2(question=question, context=retrieved_text)['answer']
    answer3 = qa_expert3(question=question, context=retrieved_text)['answer']

    # Refine to select best answer
    final_answer = refine_answers_with_llm(question, [answer1, answer2, answer3])

    return {
        "answer_expert1": answer1,
        "answer_expert2": answer2,
        "answer_expert3": answer3,
        "final_moe_answer": final_answer,
        "llm": "OpenAI GPT-3.5" if OPENAI_AVAILABLE and openai.api_key else "Local GPT-2"
    }

if __name__ == "__main__":
    print("Loading documents from 'papers' folder...")
    documents = load_pdfs_from_folder("papers")
    print(f"Loaded {len(documents)} documents.")

    print("Building FAISS index...")
    texts, index, embeddings = build_faiss_index(documents)
    print(f"Indexed {len(texts)} text chunks.")

    question = "What are the key benefits of tomato production?"
    print(f"Answering question: {question}")
    answers = get_answer_with_steps(question, texts, index, embeddings)

    print("\n--- Results ---")
    print("Expert 1:", answers["answer_expert1"])
    print("Expert 2:", answers["answer_expert2"])
    print("Expert 3:", answers["answer_expert3"])
    print("\nFinal Answer:", answers["final_moe_answer"])
    print("\nLLM Used:", answers["llm"])
