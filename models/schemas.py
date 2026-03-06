"""Pydantic models for API schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")
    use_parent_retrieval: bool = Field(default=True, description="Use parent chunk retrieval")
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunk_ids: List[int]
    num_chunks_retrieved: int
    contexts: List[Dict]
    status: str
    conversation_id: str
    response_time_ms: float


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int
    parents_loaded: int
    index_size: int


class StatsResponse(BaseModel):
    total_children: int
    total_parents: int
    index_vectors: int
    config: Dict
