"""Embedding service using OpenAI API."""

import numpy as np
from typing import List
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embedding using OpenAI API"""
    
    def __init__(self, model: str, openai_client: OpenAI):
        self.model = model
        self.client = openai_client
        logger.info(f"Initialized EmbeddingService with model: {model}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        try:
            response = self.client.embeddings.create(
                input=[text.replace("\n", " ")],
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype='float32')
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                raise
        
        return np.array(all_embeddings, dtype='float32')
