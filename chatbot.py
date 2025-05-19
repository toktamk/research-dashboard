import os
import openai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_documents():
    docs = []
    papers_dir = "papers"
    for file in os.listdir(papers_dir):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(papers_dir, file))
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

def ask_openai(prompt, max_tokens=300, temperature=0.5):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def get_answer_with_steps(question, texts, index, embeddings, top_k=3, token_limit=1500):
    # Step 1: General LLM response
    prompt_general = f"Answer this research-related question concisely:\n{question}"
    answer1 = ask_openai(prompt_general, max_tokens=200)

    # Step 2: RAG - retrieve context using vector search
    q_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)
    retrieved = " ".join([texts[i] for i in indices[0]])
    retrieved = retrieved[:token_limit]  # truncate to avoid prompt overload

    # Use local QA pipeline for context-based answer
    answer2 = qa_pipeline(question=question, context=retrieved)['answer']

    # Step 3: Merge both answers into a final concise answer using OpenAI
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
