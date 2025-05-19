import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

def get_answer(question, texts, index, embeddings, top_k=3):
    q_embedding = embedder.encode([question])
    _, I = index.search(np.array(q_embedding), top_k)
    relevant_context = " ".join([texts[i] for i in I[0]])
    result = qa_pipeline(question=question, context=relevant_context)
    return result['answer']
