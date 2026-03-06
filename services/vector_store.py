"""Vector store using FAISS."""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages FAISS vector index operations"""
    
    def __init__(self, index_path: Path, embeddings_path: Path, distance_metric: str = "cosine"):
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.distance_metric = distance_metric
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
    
    def load(self) -> None:
        """Load existing FAISS index and embeddings"""
        if self.index_path.exists() and self.embeddings_path.exists():
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            self.embeddings = np.load(str(self.embeddings_path))
            logger.info(f"✓ Loaded {self.index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Index or embeddings not found")
    
    def create(self, embeddings: np.ndarray) -> None:
        """Create new FAISS index from embeddings"""
        self.embeddings = embeddings.astype('float32')
        dimension = embeddings.shape[1]
        
        # Create index based on distance metric
        if self.distance_metric == "cosine":
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(dimension)
        elif self.distance_metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:  # ip
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(self.embeddings)
        logger.info(f"✓ Created FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors"""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or create() first")
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query if using cosine
        if self.distance_metric == "cosine":
            query_vector = query_vector.copy()
            faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to list of (index, score) tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if self.distance_metric == "cosine" or self.distance_metric == "ip":
                score = float(dist)  # Higher is better
            else:  # l2
                score = 1.0 / (1.0 + float(dist))  # Convert distance to similarity
            results.append((int(idx), score))
        
        return results
    
    def save(self) -> None:
        """Save index and embeddings to disk"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        np.save(str(self.embeddings_path), self.embeddings)
        logger.info(f"✓ Saved index to {self.index_path}")
