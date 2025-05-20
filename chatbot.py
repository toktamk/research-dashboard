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
    # Try to load API key securely
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    OPENAI_AVAILABLE = False


# Initialize local models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm_general = pipeline("text-generation", model="gpt2")
llm_refine = pipeline("text-generation", model="gpt2")


def load_pdfs_from_folder(folder_path="papers"):
    """
    Load and return documents from all PDFs in the given folder.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
    return docs


def build_faiss_index(documents, chunk_size=1000, chunk_overlap=150):
    """
    Split documents into chunks, embed them, and build a FAISS index.
    Returns: texts (list of chunks), faiss index, embeddings array
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return texts, index, embeddings


def ask_openai(prompt, max_tokens=300, temperature=0.5):
    """
    Query OpenAI chat completion with a system role for better responses.
    """
    if not OPENAI_AVAILABLE or not openai.api_key:
        raise RuntimeError("OpenAI API not available or API key missing.")

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful research assistant with expertise on Machine Learning, Deep Learning, Medical Image Analysis and Healthcare Data Analytics. "
            "Provide concise, accurate, and well-reasoned answers based on the question and context."
        )
    }

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            system_message,
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


def llm_generate_answer(prompt):
    """
    Generate answer using local GPT-2 model with prompt engineering.
    """
    engineered_prompt = (
        "You are a helpful research assistant with expertise on Machine Learning, Deep Learning, Medical Image Analysis and Healthcare Data Analytics.\n"
        f"{prompt}"
    )
    results = llm_general(engineered_prompt, max_new_tokens=200, do_sample=True)
    return results[0]['generated_text'].strip()


def refine_answers_with_llm(question, answer1, answer2):
    """
    Refine two answers using local GPT-2 model to produce a more accurate response.
    """
    refine_prompt = (
        f"Question: {question}\n"
        f"Answer 1: {answer1}\n"
        f"Answer 2: {answer2}\n"
        "Refine answers and produce a more accurate, complete, and reasoned response:"
    )
    results = llm_refine(refine_prompt, max_new_tokens=200, do_sample=True)
    return results[0]['generated_text'].strip()


def get_answer_with_steps(question, texts, index, embeddings, top_k=3, token_limit=1500):
    """
    Main function to get an answer with stepwise approach:
    1) General LLM answer
    2) RAG answer from retrieved context
    3) Refine answers for final output
    Falls back to local models if OpenAI not available.
    """
    try:
        if OPENAI_AVAILABLE and openai.api_key:
            # Step 1: General LLM response
            prompt_general = f"Answer this research-related question concisely:\n{question}"
            answer1 = ask_openai(prompt_general, max_tokens=200)

            # Step 2: Retrieve relevant context with vector search
            q_embedding = embedder.encode([question], convert_to_numpy=True)
            distances, indices = index.search(q_embedding, top_k)
            retrieved_text = " ".join([texts[i] for i in indices[0]])[:token_limit]

            # Step 3: Extract answer from retrieved context using local QA pipeline
            answer2 = qa_pipeline(question=question, context=retrieved_text)['answer']

            # Step 4: Refine both answers using OpenAI
            refine_prompt = (
                f"Question: {question}\n"
                f"Answer from general model: {answer1}\n"
                f"Answer from research context: {answer2}\n\n"
                "Based on both, generate a short, clear, and accurate answer:"
            )
            final_answer = ask_openai(refine_prompt, max_tokens=250)
            return {
                "answer1_general_llm": answer1,
                "answer2_rag": answer2,
                "final_refined_answer": final_answer,
                "llm": "openai GPT-3.5-turbo"
            }
        else:
            raise RuntimeError("OpenAI not available, using local fallback.")
    except Exception as e:
        # Fallback to local models
        prompt1 = f"Answer the following question:\n{question}"
        answer1 = llm_generate_answer(prompt1)

        q_embedding = embedder.encode([question], convert_to_numpy=True)
        _, indices = index.search(q_embedding, top_k)
        relevant_context = " ".join([texts[i] for i in indices[0]])

        answer2_result = qa_pipeline(question=question, context=relevant_context)
        answer2 = answer2_result['answer']

        final_answer = refine_answers_with_llm(question, answer1, answer2)
        return {
            "answer1_general_llm": answer1,
            "answer2_rag": answer2,
            "final_refined_answer": final_answer,
            "llm": "local GPT-2 and DistilBERT"
        }


if __name__ == "__main__":
    # Example usage
    print("Loading documents from 'papers' folder...")
    documents = load_pdfs_from_folder("papers")
    print(f"Loaded {len(documents)} documents.")

    print("Building FAISS index...")
    texts, index, embeddings = build_faiss_index(documents)
    print(f"Indexed {len(texts)} text chunks.")

    # Example question
    question = "What are the key benefits of tomato production?"

    print(f"Answering question: {question}")
    answers = get_answer_with_steps(question, texts, index, embeddings)

    print("\n--- Results ---")
    print("General LLM answer:")
    print(answers["answer1_general_llm"])
    print("\nRAG answer from context:")
    print(answers["answer2_rag"])
    print("\nFinal refined answer:")
    print(answers["final_refined_answer"])
    print(f"\nUsed LLM: {answers['llm']}")
