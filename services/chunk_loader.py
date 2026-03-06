"""Chunk loader for parent-child chunks."""

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import logging
from models.chunk_models import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


class ChunkLoader:
    """Loads and manages parent-child chunks"""
    
    def __init__(self, chunks_path: Path):
        self.chunks_path = chunks_path
        self.child_chunks: List[ChildChunk] = []
        self.parent_chunks: Dict[str, ParentChunk] = {}
    
    def load(self) -> None:
        """Load chunks from JSONL file"""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        
        logger.info(f"Loading chunks from {self.chunks_path}")
        
        # Group children by parent_id
        children_by_parent = defaultdict(list)
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                child = ChildChunk.from_dict(data)
                self.child_chunks.append(child)
                children_by_parent[child.parent_id].append(child)
        
        # Create parent chunks
        for parent_id, children in children_by_parent.items():
            parent = ParentChunk.from_children(parent_id, children)
            self.parent_chunks[parent_id] = parent
        
        logger.info(f"✓ Loaded {len(self.child_chunks)} child chunks from {len(self.parent_chunks)} parents")
    
    def get_child_chunk(self, index: int) -> ChildChunk:
        """Get child chunk by index"""
        return self.child_chunks[index]
    
    def get_parent_chunk(self, parent_id: str) -> ParentChunk:
        """Get parent chunk by parent_id"""
        return self.parent_chunks[parent_id]
