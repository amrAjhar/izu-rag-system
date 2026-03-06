"""Main RAG service orchestrating all components."""

from typing import List, Dict, Tuple
import time
import logging
from dataclasses import dataclass
from services.chunk_loader import ChunkLoader
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.reranker import Reranker, CrossEncoderReranker, LLMReranker, NoReranker
from services.retrieval_engine import RetrievalEngine
from services.answer_generator import AnswerGenerator
from services.conversation_manager import ConversationManager
from config.rag_config import RAGConfig
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG query response"""
    answer: str
    chunk_ids: List[int]
    contexts: List[Dict]
    conversation_id: str
    response_time_ms: float


class RAGService:
    """Main RAG service orchestrating all components"""
    
    def __init__(self, config: RAGConfig, openai_client: OpenAI):
        self.config = config
        self.client = openai_client
        
        # Initialize components
        logger.info("Initializing RAG service...")
        
        # 1. Chunk loader
        self.chunk_loader = ChunkLoader(config.chunks_path)
        self.chunk_loader.load()
        
        # 2. Embedding service
        self.embedding_service = EmbeddingService(config.embedding_model, openai_client)
        
        # 3. Vector store
        self.vector_store = VectorStore(
            config.faiss_index_path,
            config.embeddings_path,
            config.distance_metric
        )
        
        # Try to load existing index, or create new one
        try:
            self.vector_store.load()
        except FileNotFoundError:
            logger.warning("Index not found, creating new one...")
            self._create_index()
        
        # 4. Reranker
        self.reranker = self._create_reranker()
        
        # 5. Retrieval engine
        self.retrieval_engine = RetrievalEngine(
            self.embedding_service,
            self.vector_store,
            self.chunk_loader,
            self.reranker
        )
        
        # 6. Answer generator
        self.answer_generator = AnswerGenerator(
            openai_client,
            config.chat_model,
            config.temperature,
            config.max_tokens
        )
        
        # 7. Conversation manager
        self.conversation_manager = ConversationManager(config.max_history_length)
        
        logger.info("✅ RAG service ready!")
    
    def _create_reranker(self) -> Reranker:
        """Create reranker based on config"""
        if not self.config.reranker_enabled or self.config.reranker_type == 'none':
            return NoReranker()
        elif self.config.reranker_type == 'crossencoder':
            return CrossEncoderReranker(self.config.reranker_model)
        elif self.config.reranker_type == 'llm':
            return LLMReranker(self.client, self.config.chat_model)
        else:
            logger.warning(f"Unknown reranker type: {self.config.reranker_type}, using NoReranker")
            return NoReranker()
    
    def _create_index(self):
        """Create FAISS index from chunks"""
        logger.info("Generating embeddings for all chunks...")
        texts = [chunk.text for chunk in self.chunk_loader.child_chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        
        self.vector_store.create(embeddings)
        self.vector_store.save()
        logger.info("✅ Index created and saved")
    
    def query(
        self,
        query: str,
        conversation_id: str,
        history: List[Dict] = None,
        use_parent_retrieval: bool = True
    ) -> RAGResponse:
        """Process RAG query with conversation history"""
        start_time = time.time()
        
        # Add user message to history
        self.conversation_manager.add_message(conversation_id, "user", query)
        
        # Retrieve relevant contexts
        contexts, chunk_ids = self.retrieval_engine.retrieve(
            query,
            self.config.top_k,
            use_parent_retrieval,
            rerank=self.config.reranker_enabled
        )
        
        # Get conversation history for LLM
        llm_history = None
        if history:
            llm_history = history
        else:
            llm_history = self.conversation_manager.format_history_for_llm(conversation_id)
        
        # Generate answer
        answer = self.answer_generator.generate(query, contexts, llm_history)
        
        # Add assistant response to history
        self.conversation_manager.add_message(conversation_id, "assistant", answer)
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000
        
        # Format contexts for response
        context_dicts = [
            {
                "title": ctx[0].title,
                "url": ctx[0].url,
                "text": ctx[0].text[:200] + "...",
                "score": float(ctx[1])  # Convert numpy.float32 to Python float
            }
            for ctx in contexts[:3]
        ]
        
        return RAGResponse(
            answer=answer,
            chunk_ids=chunk_ids,
            contexts=context_dicts,
            conversation_id=conversation_id,
            response_time_ms=response_time
        )
