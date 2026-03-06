#!/usr/bin/env python3
"""
Turkish RAG Server - Port 5001
Loads Turkish data and serves Turkish RAG
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load Turkish config
env_path = Path(__file__).parent / ".env.turkish"
load_dotenv(env_path, override=True)

print(f"🇹🇷 Loading Turkish RAG configuration from {env_path}")
print(f"   Chunks: {os.getenv('CHUNKS_PATH')}")
print(f"   Index: {os.getenv('FAISS_INDEX_PATH')}")
print(f"   Port: 5001\n")

# Import and run the main server with Turkish config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import logging
import time
from contextlib import asynccontextmanager
from models.schemas import ChatRequest, ChatResponse, HealthResponse, StatsResponse
from services.rag_service import RAGService
from config.rag_config import RAGConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG service
rag_service: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global rag_service
    
    # Startup
    logger.info("🇹🇷 Starting Turkish RAG system...")
    
    # Load configuration
    config = RAGConfig.from_env()
    config.validate()
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Initialize RAG service
    rag_service = RAGService(config, openai_client)
    
    logger.info("✅ Turkish RAG system ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Turkish RAG system...")


# Initialize FastAPI app
app = FastAPI(
    title="IZU Turkish RAG System",
    description="Turkish-only RAG with parent-child chunking",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🇹🇷 IZU Turkish RAG System v2.0",
        "language": "Turkish",
        "status": "running",
        "port": 5001
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return HealthResponse(
        status="healthy",
        chunks_loaded=len(rag_service.chunk_loader.child_chunks),
        parents_loaded=len(rag_service.chunk_loader.parent_chunks),
        index_size=rag_service.vector_store.index.ntotal
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """System statistics"""
    return StatsResponse(
        total_children=len(rag_service.chunk_loader.child_chunks),
        total_parents=len(rag_service.chunk_loader.parent_chunks),
        index_vectors=rag_service.vector_store.index.ntotal,
        config={
            "language": "Turkish",
            "embedding_model": rag_service.config.embedding_model,
            "chat_model": rag_service.config.chat_model,
            "reranker": rag_service.config.reranker_type,
            "distance_metric": rag_service.config.distance_metric,
            "max_history": rag_service.config.max_history_length,
            "temperature": rag_service.config.temperature,
            "max_tokens": rag_service.config.max_tokens
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with conversation history support"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_tr_{int(time.time())}"
        
        # Process query
        response = rag_service.query(
            query=request.message,
            conversation_id=conversation_id,
            history=[msg.dict() for msg in request.history] if request.history else None,
            use_parent_retrieval=request.use_parent_retrieval
        )
        
        return ChatResponse(
            answer=response.answer,
            retrieved_chunk_ids=response.chunk_ids,
            num_chunks_retrieved=len(response.chunk_ids),
            contexts=response.contexts,
            status="success",
            conversation_id=response.conversation_id,
            response_time_ms=response.response_time_ms
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    try:
        rag_service.conversation_manager.clear_conversation(conversation_id)
        return {
            "status": "success",
            "message": f"Cleared conversation {conversation_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        history = rag_service.conversation_manager.get_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "message_count": len(history),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in history
            ]
        }
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🇹🇷 IZU TURKISH RAG SYSTEM v2.0 - STARTING SERVER")
    print("="*80)
    print("\nServer will start at: http://localhost:5001")
    print("API docs available at: http://localhost:5001/docs")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
