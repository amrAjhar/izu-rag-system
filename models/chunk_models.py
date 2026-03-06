"""Data models for RAG chunks."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ChildChunk:
    """Represents a child chunk"""
    parent_id: str
    child_index: int
    text: str
    url: str
    title: str
    university: str
    language: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChildChunk':
        return cls(
            parent_id=data['parent_id'],
            child_index=data['child_index'],
            text=data['text'],
            url=data['url'],
            title=data['title'],
            university=data.get('university', ''),
            language=data.get('language', 'en')
        )


@dataclass
class ParentChunk:
    """Represents a parent chunk (full context)"""
    parent_id: str
    text: str
    url: str
    title: str
    university: str
    language: str
    char_count: int
    
    @classmethod
    def from_children(cls, parent_id: str, children: List['ChildChunk']) -> 'ParentChunk':
        # Sort by child_index and concatenate
        sorted_children = sorted(children, key=lambda x: x.child_index)
        text = ' '.join([c.text for c in sorted_children])
        
        return cls(
            parent_id=parent_id,
            text=text,
            url=sorted_children[0].url,
            title=sorted_children[0].title,
            university=sorted_children[0].university,
            language=sorted_children[0].language,
            char_count=len(text)
        )


@dataclass
class SearchResult:
    """Search result with score"""
    chunk: ChildChunk
    score: float
    rank: int


@dataclass
class ConversationMessage:
    """Single conversation message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None
