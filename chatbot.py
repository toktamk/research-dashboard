# chatbot.py

import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

def build_chatbot():
    # Step 1: Load PDFs
    docs = []
    for file in os.listdir("papers"):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join("papers", file))
            docs.extend(loader.load())

    # Step 2: Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Step 3: Create embeddings using MiniLM (lightweight and fast)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Store vectors in FAISS
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Step 5: Load QA model (lightweight transformer)
    model_name = "distilbert-base-uncased-distilled-squad"
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

    # Step 6: Wrap pipeline in LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Step 7: Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

    return qa
