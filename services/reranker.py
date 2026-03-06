"""Reranking services for improving search results."""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from openai import OpenAI
import json
import logging
from models.chunk_models import ChildChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Base reranker class"""
    
    def rerank(self, query: str, chunks: List[Tuple[ChildChunk, float]], top_k: int) -> List[Tuple[ChildChunk, float]]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """CrossEncoder-based reranking"""
    
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        logger.info(f"Initialized CrossEncoderReranker with {model_name}")
    
    def rerank(self, query: str, chunks: List[Tuple[ChildChunk, float]], top_k: int) -> List[Tuple[ChildChunk, float]]:
        if not chunks:
            return []
        
        try:
            # Prepare pairs
            pairs = [[query, chunk.text] for chunk, _ in chunks]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Normalize scores
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [0.5] * len(scores)
            
            # Combine and sort
            reranked = [(chunks[i][0], normalized_scores[i]) for i in range(len(chunks))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}, using original scores")
            return chunks[:top_k]


class LLMReranker(Reranker):
    """LLM-based reranking"""
    
    def __init__(self, openai_client: OpenAI, model: str):
        self.client = openai_client
        self.model = model
        logger.info(f"Initialized LLMReranker with {model}")
    
    def rerank(self, query: str, chunks: List[Tuple[ChildChunk, float]], top_k: int) -> List[Tuple[ChildChunk, float]]:
        if not chunks:
            return []
        
        try:
            candidates = chunks[:10]
            
            prompt = f"""Given the question: "{query}"

Rate the relevance of each text chunk on a scale of 0-10.
Return ONLY a JSON array of scores, e.g., [8, 3, 9, 2, ...]

Chunks:
"""
            for i, (chunk, _) in enumerate(candidates):
                prompt += f"\n{i}. {chunk.text[:300]}..."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a relevance scoring assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            scores = json.loads(response.choices[0].message.content.strip())
            
            reranked = []
            for i, (chunk, orig_score) in enumerate(candidates):
                if i < len(scores):
                    llm_score = scores[i] / 10.0
                    combined = (orig_score * 0.3) + (llm_score * 0.7)
                    reranked.append((chunk, combined))
            
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return chunks[:top_k]


class NoReranker(Reranker):
    """No reranking, just return top_k"""
    
    def rerank(self, query: str, chunks: List[Tuple[ChildChunk, float]], top_k: int) -> List[Tuple[ChildChunk, float]]:
        return chunks[:top_k]
