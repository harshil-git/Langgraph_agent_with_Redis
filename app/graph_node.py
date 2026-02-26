import os
from langgraph.graph import StateGraph, END
from app.re_ranker import ranker
from langchain_huggingface import HuggingFaceEmbeddings
from redis import Redis


from app.bedrock import llm_inference,judge_eval
from dotenv import load_dotenv
load_dotenv()
import logging
logger = logging.getLogger(__name__)

from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from app.retriever import load_docs_from_s3, load_faiss_from_s3
from cache_manager_1 import search_cache,store_cache,init_cache

REDIS_URL = os.getenv("REDIS_URL")
redis_client = Redis.from_url(REDIS_URL)


# State definition
class QAState(dict):
    question: str
    documents: list
    answer: str
    metrics: dict
    attempts: int
    cache_level: str


faiss_retriever= load_faiss_from_s3(index_name="pdfindex")


docs= load_docs_from_s3(index_name="bm25")
bm25_retriever = BM25Retriever.from_documents(docs)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)


# Nodes
def retrieve_node(state: QAState):
    print(f"\n[RetrieveNode] Entered for query: {state['question']}")
    #doc_retriever = get_retriever()
    if hybrid_retriever is None:
        raise ValueError("Retriever not initialized! Upload a PDF first.")
    searched_para = hybrid_retriever.invoke(state["question"])
    context_doc = [doc.page_content for doc in searched_para]
    
    print(f"[RetrieveNode] Retrieved {len(context_doc)} docs")
    for i, d in enumerate(context_doc[:3]):
        print(f"Doc {i+1} preview: {str(d)[:200]}...\n")
    return {**state, "documents": context_doc}


def rerank_node(state: QAState):
    print(f"[RerankNode] Received {len(state['documents'])} docs")
    docs = ranker(state["question"], state["documents"])
    return {**state, "documents": docs}


def generate_node(state: QAState):
    print(" ** ENTERED GENERATE NODE **")
    context = "\n\n".join([f"{d}" for d in state["documents"]])

    response = llm_inference(state['question'],context)
    return {**state, "answer": response,"cache_level":"LLM_call"}


def self_check_node(state: QAState):
    judge = judge_eval(state['documents'],state['answer'])
    grounded = "YES" in judge.upper()
    return {**state, "metrics": {"grounded": grounded}, "attempts": state.get("attempts", 0) + 1}

def semantic_cache(state):
    query = state["question"]
    cached_answer = search_cache(query)

    if cached_answer:
        print("[Semantic Cache] HIT found.")
        # FIX 1: Return the update dictionary
        return {"answer": cached_answer, "cache_level": "semantic_cache"}
    else:
        print("[Semantic Cache] MISS.")
        # Return explicit None or empty update to proceed
        return {"answer": None, "cache_level": None}

def persist_cache_node(state):
    if "answer" in state and state.get("cache_level") == "LLM_call":
        query = state["question"]
        answer = state["answer"]
        context = "\n\n".join([str(d) for d in state.get("documents", [])])
        print(f"[Persist Cache] Storing answer for: {query}")
        store_cache(query, context, answer)

    return state

# Graph build
def build_graph():
    init_cache()
    g = StateGraph(QAState)
    
    g.add_node("semantic_cache", semantic_cache)
    
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    #g.add_node("answer_cache", answer_cache_node)

    g.add_node("generate", generate_node)
    g.add_node("self_check", self_check_node)
    g.add_node("persist_cache", persist_cache_node)
    
    g.set_entry_point("semantic_cache")

    g.add_conditional_edges(
    "semantic_cache",
    lambda s: "retrieve" if s.get("answer") is None else END)

    g.add_edge("retrieve", "rerank")
    
    g.add_edge("rerank","generate")
    
    g.add_edge("generate", "self_check")

    g.add_conditional_edges(
    "self_check",
    lambda s: (
        "retrieve"
        if not s.get("metrics", {}).get("grounded", True) and s.get("attempts", 0) < 2
        else "persist_cache"
    ))

    g.add_edge("persist_cache", END)
    
    #redis_conn = Redis.from_url(REDIS_URL)
    checkpointer_cm = RedisSaver.from_conn_string(REDIS_URL)
    store_cm = RedisStore.from_conn_string(REDIS_URL)

    checkpointer = checkpointer_cm.__enter__()
    redis_store = store_cm.__enter__()
    
    graph = g.compile(checkpointer=checkpointer,store=redis_store)
    return graph
