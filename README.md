# Langgraph_agent_with_Redis short term memory

## Overview
- - When user uploads pdf file it creates faiss index and uploads to s3.
- upon query similarity is checked if answer is cached then instantly returned to user.
- in case of no similarity cache miss, agent is invoked and relevant context is retrieved through hybrid retrieval(faiss + bm25),
- reranked and generated through LLM call and answer is presented to user.
- Later answer is cached in redis.

# Demo
https://github.com/user-attachments/assets/e9d798a3-8fed-4169-818b-87a4bf0798ab

# Technologies Used
- Langgraph
- Langchain
- AWS BEDROCK
- AWS S3
- RAG
- FAISS
- FASTAPI
- REDIS
- Streamlit
