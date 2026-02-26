import os
from hashlib import md5
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
INDEX_NAME = "semantic-cache"


EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


_vector_store: RedisVectorStore | None = None


def init_cache() -> None:
    """
    Initialize or load the Redis vector index.
    """
    global _vector_store

    if _vector_store is not None:
        return

    _vector_store = RedisVectorStore(
        redis_url= REDIS_URL,
        embeddings=EMBEDDINGS,
        index_name=INDEX_NAME,
    )

    print(f"[SemanticCache] Initialized index: {INDEX_NAME}")


def _cache_key(query: str) -> str:
    return md5(query.encode("utf-8")).hexdigest()


def search_cache(
    query: str,
    k: int = 1,
    max_distance: float = 0.25, 
) -> str | None:
    """
    Returns cached answer if similarity is high enough.
    """

    if _vector_store is None:
        raise RuntimeError("Semantic cache not initialized. Call init_cache().")

    results = _vector_store.similarity_search_with_score(query, k=k)

    if not results:
        return None

    doc, distance = results[0]

    # Redis returns COSINE DISTANCE (0 = identical)
    if distance > max_distance:
        return None

    return doc.metadata.get("answer")


def store_cache(query: str, context: str, answer: str) -> None:
    """
    Store query → answer mapping in semantic cache.
    """
    if _vector_store is None:
        raise RuntimeError("Semantic cache not initialized. Call init_cache().")

    _vector_store.add_texts(
        texts=[query],
        metadatas=[
            {
                "answer": answer,
                "context": context,
                "key": _cache_key(query),
            }
        ],
    )

    print(f"[SemanticCache] Stored: {query}")
