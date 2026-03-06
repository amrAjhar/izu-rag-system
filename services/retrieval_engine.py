"""Retrieval engine orchestrating search and reranking."""

from typing import List, Tuple, Dict
import logging
from models.chunk_models import ChildChunk, ParentChunk
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.chunk_loader import ChunkLoader
from services.reranker import Reranker

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Orchestrates search and retrieval"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunk_loader: ChunkLoader,
        reranker: Reranker
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_loader = chunk_loader
        self.reranker = reranker
    
    def search_child_chunks(self, query: str, top_k: int) -> List[Tuple[ChildChunk, float]]:
        """Search child chunks using vector similarity"""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        # Convert to (ChildChunk, score) tuples
        chunks_with_scores = []
        for idx, score in results:
            chunk = self.chunk_loader.get_child_chunk(idx)
            chunks_with_scores.append((chunk, score))
        
        return chunks_with_scores
    
    def retrieve_parents(self, child_chunks: List[Tuple[ChildChunk, float]]) -> List[Tuple[ParentChunk, float]]:
        """Convert child chunks to parent chunks, deduplicate"""
        parent_scores: Dict[str, Tuple[ParentChunk, float]] = {}
        
        for child, score in child_chunks:
            parent_id = child.parent_id
            
            if parent_id not in parent_scores or score > parent_scores[parent_id][1]:
                parent = self.chunk_loader.get_parent_chunk(parent_id)
                parent_scores[parent_id] = (parent, score)
        
        # Convert to list and sort
        parents = [(parent, score) for parent, score in parent_scores.values()]
        parents.sort(key=lambda x: x[1], reverse=True)
        
        return parents
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        use_parent_retrieval: bool = True,
        rerank: bool = True
    ) -> Tuple[List, List[int]]:
        """Full retrieval pipeline"""
        # Step 1: Search child chunks
        child_results = self.search_child_chunks(query, top_k)
        logger.info(f"Retrieved {len(child_results)} child chunks")
        
        # Step 2: Rerank (if enabled)
        if rerank:
            child_results = self.reranker.rerank(query, child_results, min(5, top_k))
            logger.info(f"Reranked to top {len(child_results)} chunks")
        
        # Store chunk IDs before parent retrieval
        chunk_ids = [self.chunk_loader.child_chunks.index(chunk) for chunk, _ in child_results]
        
        # Step 3: Retrieve parents (if enabled)
        if use_parent_retrieval:
            context_results = self.retrieve_parents(child_results)
        else:
            context_results = child_results
        
        return context_results, chunk_ids
