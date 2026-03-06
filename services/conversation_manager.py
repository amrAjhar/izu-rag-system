"""Conversation history manager."""

from typing import List, Dict, Optional
from models.chunk_models import ConversationMessage
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history"""
    
    def __init__(self, max_history_length: int = 6):
        self.max_history_length = max_history_length
        self.conversations: Dict[str, List[ConversationMessage]] = {}
    
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Add message to conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        self.conversations[conversation_id].append(message)
        
        # Trim to max length
        if len(self.conversations[conversation_id]) > self.max_history_length:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history_length:]
        
        logger.debug(f"Added {role} message to conversation {conversation_id}")
    
    def get_history(self, conversation_id: str) -> List[ConversationMessage]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    def format_history_for_llm(self, conversation_id: str) -> List[Dict[str, str]]:
        """Format history for LLM (OpenAI format)"""
        history = self.get_history(conversation_id)
        return [{"role": msg.role, "content": msg.content} for msg in history]
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
