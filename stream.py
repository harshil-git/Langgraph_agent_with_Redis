import streamlit as st
import requests
from app.process_pdfs import process_all_pdfs
from app.retriever import upload_docs_to_s3, build_and_upload_faiss

API_URL = "http://127.0.0.1:8000/" 

st.title("📄 PDF Search")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    docs = process_all_pdfs(uploaded_file)
    upload_docs_to_s3(docs,index_name="bm25")
    build_and_upload_faiss(docs,index_name="pdfindex")
    st.success("File uploaded and vectors db created! Ready to query.")

question = st.text_input("Ask a question:")
if st.button("Submit"):
    if question:
        r = requests.post(f"{API_URL}/query", json={"question": question}) 
        res = r.json()
        if res.get("error"):
            st.error(res["error"])
        elif res.get("Answer"):
            st.write("### cache_level:")
            st.write(res["Cache_level"])
            st.write("### Answer:")
            st.write(res["Answer"])
        else:
            st.warning("No answer found.")
        