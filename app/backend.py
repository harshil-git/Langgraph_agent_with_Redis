import uuid
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


from app.graph_node import build_graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

fast_app = FastAPI(title="PDF RAG QA API")

try:
    agent_graph = build_graph()
    logger.info("Graph initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize graph: {e}")
    raise e


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None  

@fast_app.post("/query")
def query(payload: QueryRequest, request: Request): 
    
    try:
        
        thread_id = payload.session_id if payload.session_id else str(uuid.uuid4())
        
        logger.info(f"Processing query for thread_id: {thread_id}")

        initial_state = {
            "question": payload.question, 
            "attempts": 0,
            "metrics": {},
            "answer": None,
            "cache_level": None
        }

        
        result = agent_graph.invoke(
            initial_state, 
            config={"configurable": {"thread_id": thread_id}}
        )

        
        return {
            "Answer": result.get("answer", "No answer generated."),
            "Metrics": result.get("metrics", {}),
            "Cache_level": result.get("cache_level", "none"),
            "Thread_id": thread_id 
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))