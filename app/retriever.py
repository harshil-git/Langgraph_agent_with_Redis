
import os
import pickle
import json
import torch
import boto3
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
import os


s3 = boto3.client("s3")
BUCKET_NAME = "faiss-bucket-langgraph"
VECTOR_DIR = '/Users/harshilgohil/python/AWS/Agent_bedrock/VECTOR_DIR'
FAISS_FILE = f"{VECTOR_DIR}/index.faiss"
PICKELE_FILE = f"{VECTOR_DIR}/index.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

DATA_CORPUS = "data_corpus.json"

s3 = boto3.client("s3")

def build_and_upload_faiss(docs, index_name: str):
    # Build FAISS store from docs
    db = FAISS.from_documents(docs, embedding_model)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    # Save FAISS index + docs metadata
    db.save_local(VECTOR_DIR)

    # Upload files to S3
    s3.upload_file(FAISS_FILE, BUCKET_NAME, f"{index_name}/index.faiss")
    s3.upload_file(PICKELE_FILE, BUCKET_NAME, f"{index_name}/index.pkl")

    print("Uploaded to S3!")

    return db

def load_faiss_from_s3(index_name: str):

    os.makedirs(VECTOR_DIR, exist_ok=True)

    # Download from S3
    s3.download_file(BUCKET_NAME, f"{index_name}/index.faiss", FAISS_FILE)
    s3.download_file(BUCKET_NAME, f"{index_name}/index.pkl", PICKELE_FILE)

    # Load vectorstore
    db = FAISS.load_local(VECTOR_DIR, embedding_model, allow_dangerous_deserialization=True)
    
    return db.as_retriever(search_kwargs={"k": 12})

def serialize_docs(docs):
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

def upload_docs_to_s3(docs, index_name: str):
    serializable_docs = serialize_docs(docs)

    # Save corpus
    with open(DATA_CORPUS, "w") as f:
        json.dump(serializable_docs, f)

    # Upload
    s3.upload_file(DATA_CORPUS, BUCKET_NAME, f"{index_name}/data_corpus.json")

def load_docs_from_s3(index_name: str):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"{index_name}/data_corpus.json")
    docs_json = json.loads(obj['Body'].read())
    return [Document(**d) for d in docs_json]

