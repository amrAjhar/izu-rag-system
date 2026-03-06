"""Configuration for RAG system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    
    # Paths
    chunks_path: Path
    faiss_index_path: Path
    embeddings_path: Path
    
    # Models
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Search parameters
    top_k: int = 10
    distance_metric: str = "cosine"  # cosine, l2, ip
    
    # Reranking
    reranker_enabled: bool = True
    reranker_type: str = "crossencoder"  # crossencoder, llm, none
    
    # LLM parameters
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # Conversation history
    max_history_length: int = 6  # Keep last 6 messages
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        load_dotenv()
        
        return cls(
            chunks_path=Path(os.getenv('CHUNKS_PATH', 'data/izu/parent_child_chunks.jsonl')),
            faiss_index_path=Path(os.getenv('FAISS_INDEX_PATH', 'data/izu/faiss_parent_child.index')),
            embeddings_path=Path(os.getenv('EMBEDDINGS_PATH', 'data/izu/parent_child_embeddings.npy')),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            chat_model=os.getenv('CHAT_MODEL', 'gpt-4o-mini'),
            top_k=int(os.getenv('TOP_K', '10')),
            distance_metric=os.getenv('DISTANCE_METRIC', 'cosine'),
            reranker_enabled=os.getenv('RERANKER', 'crossencoder') != 'none',
            reranker_type=os.getenv('RERANKER', 'crossencoder'),
            temperature=float(os.getenv('TEMPERATURE', '0.3')),
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
            max_history_length=int(os.getenv('MAX_HISTORY_LENGTH', '6'))
        )
    
    def validate(self) -> None:
        """Validate configuration"""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        
        if self.distance_metric not in ['cosine', 'l2', 'ip']:
            raise ValueError(f"Invalid distance metric: {self.distance_metric}")
        
        if self.reranker_type not in ['crossencoder', 'llm', 'none']:
            raise ValueError(f"Invalid reranker type: {self.reranker_type}")
