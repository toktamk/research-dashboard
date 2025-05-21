import os
import re
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

# Initialize multiple QA experts
qa_expert1 = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qa_expert2 = pipeline("question-answering", model="deepset/roberta-base-squad2")
qa_expert3 = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Embedding and local LLM for fallback
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm_refine = pipeline("text-generation", model="gpt2")

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

def ask_openai(prompt, max_tokens=150, temperature=0.3):
    if not OPENAI_AVAILABLE or not openai.api_key:
        raise RuntimeError("OpenAI API not available or API key missing.")

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful research assistant with expertise in Machine Learning, Deep Learning, "
            "Medical Image Analysis, and Healthcare Data Analytics."
        )
    }

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def clean_repeated_phrases(text):
    # Remove repeated boilerplate
    text = re.sub(r"(please select.*?no explanations\.)", "", text, flags=re.IGNORECASE)

    # Split into sentences and remove exact duplicates
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = set()
    cleaned = []
    for s in sentences:
        norm = s.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            cleaned.append(s.strip())

    # Return only the first non-empty cleaned sentence (to keep it concise)
    cleaned_ouput = ""
    if cleaned:
        cleaned_output= cleaned[0]
          
    return cleaned_output

def refine_answers_with_llm(question, answers):
    prompt = (
        f"Question: {question}\n"
        f"Answer A: {answers[0]}\n"
        f"Answer B: {answers[1]}\n"
        f"Answer C: {answers[2]}\n\n"
        "Please select the best answer and provide a brief, concise final answer only (no explanations)."
    )

    if OPENAI_AVAILABLE and openai.api_key:
        try:
            raw_answer = ask_openai(prompt, max_tokens=100, temperature=0.3)
        except Exception as e:
            print(f"OpenAI API error: {e}. Falling back to local LLM.")
            raw_answer = llm_refine(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text'].strip()
    else:
        raw_answer = llm_refine(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text'].strip()

    # Clean repeated phrases / boilerplate
    cleaned = clean_repeated_phrases(raw_answer)
    if cleaned == "":
        return majority_voting_answer(answers)
    else return cleaned

from collections import Counter

def majority_voting_answer(answers):
    """
    Returns the most common answer (majority vote) from a list of answers.
    If there's a tie, it returns all tied answers concatenated with ' / '.
    """
    if not answers:
        return "No answers provided."

    # Count frequency of each unique answer
    answer_counts = Counter([ans.strip().lower() for ans in answers])
    
    # Find the highest frequency
    max_count = max(answer_counts.values())
    
    # Get all answers with that frequency
    majority_answers = [ans for ans, count in answer_counts.items() if count == max_count]
    
    # Return the most common one, or a joined string if there's a tie
    if len(majority_answers) == 1:
        return majority_answers[0]
    else:
        return " / ".join(majority_answers)

    
def get_answer_with_steps(question, texts, index, embeddings, top_k=3, token_limit=1500):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, top_k)
    retrieved_text = " ".join([texts[i] for i in indices[0]])[:token_limit]

    # Get answers from all experts
    answer1 = qa_expert1(question=question, context=retrieved_text)['answer']
    answer2 = qa_expert2(question=question, context=retrieved_text)['answer']
    answer3 = qa_expert3(question=question, context=retrieved_text)['answer']

    # Refine to select best answer with concise summary
    final_answer = refine_answers_with_llm(question, [answer1, answer2, answer3])

    return {
        "answer_expert1": answer1,
        "answer_expert2": answer2,
        "answer_expert3": answer3,
        "final_moe_answer": final_answer,
        "llm": "OpenAI GPT-3.5" if OPENAI_AVAILABLE else "Local GPT-2"
    }

if __name__ == "__main__":
    print("Loading documents from 'papers' folder...")
    documents = load_pdfs_from_folder("papers")
    print(f"Loaded {len(documents)} documents.")

    print("Building FAISS index...")
    texts, index, embeddings = build_faiss_index(documents)
    print(f"Indexed {len(texts)} text chunks.")

    question = "Explain CNN"
    print(f"Answering question: {question}")
    answers = get_answer_with_steps(question, texts, index, embeddings)

    print("\n--- Results ---")
    print("Expert 1:", answers["answer_expert1"])
    print("Expert 2:", answers["answer_expert2"])
    print("Expert 3:", answers["answer_expert3"])
    print("\nFinal Concise Answer:", answers["final_moe_answer"])
    print("\nLLM Used:", answers["llm"])
