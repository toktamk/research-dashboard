import os
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm_general = pipeline("text-generation", model="sshleifer/distil-gpt2")
llm_merge = pipeline("text-generation", model="sshleifer/distil-gpt2")

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

def get_answer_with_steps(question, texts, index, embeddings, top_k=3):
    # Step 1: General LLM answer
    prompt1 = f"Answer the following question:\n{question}"
    answer1 = llm_generate_answer(prompt1)

    # Step 2: RAG-based answer using your papers
    q_embedding = embedder.encode([question])
    _, I = index.search(np.array(q_embedding), top_k)
    relevant_context = " ".join([texts[i] for i in I[0]])
    answer2_result = qa_pipeline(question=question, context=relevant_context)
    answer2 = answer2_result['answer']

    # Step 3: Merged answer using LLM
    final_answer = merge_answers_with_llm(question, answer1, answer2)

    return {
        "answer1_general_llm": answer1,
        "answer2_rag": answer2,
        "final_merged_answer": final_answer
    }
