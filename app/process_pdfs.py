
import boto3
import os
import pickle
import json
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


PDF_STORAGE_PATH = '/Users/harshilgohil/python/AWS/Agent_bedrock/data'


def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdfs_from_directory(path):
    all_docs = []
    loader = PyPDFLoader(path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    all_docs.extend(docs)
    return all_docs


def process_all_pdfs(uploaded_file):
    pdf_path = save_uploaded_file(uploaded_file)
    pdf_docs = load_pdfs_from_directory(pdf_path)
    
    return pdf_docs

