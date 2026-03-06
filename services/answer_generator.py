"""Answer generation using LLM."""

from typing import List, Tuple, Optional
from openai import OpenAI
import logging
from models.chunk_models import ParentChunk, ChildChunk

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using LLM"""
    
    def __init__(
        self,
        openai_client: OpenAI,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.client = openai_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized AnswerGenerator with {model}")
    
    def generate(
        self,
        query: str,
        contexts: List[Tuple],
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """Generate answer from contexts and optional conversation history"""
        
        # Build context text
        context_text = "\n\n---\n\n".join([
            f"Source: {ctx[0].title}\nURL: {ctx[0].url}\n\n{ctx[0].text}"
            for ctx in contexts
        ])
        
        # System prompt
        system_prompt = """You are a helpful university information assistant. Answer questions based on the provided context.
If the information is not explicitly stated in the context, say so clearly. Always cite sources when possible."""

        #improve the prompt.
        # User prompt
        user_prompt = f"""Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"""
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": user_prompt})
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
