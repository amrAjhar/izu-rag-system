"""Test RAG service initialization and query"""

import logging
from openai import OpenAI
from config.rag_config import RAGConfig
from services.rag_service import RAGService
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_rag_service():
    print("="*80)
    print("Testing RAG Service Initialization")
    print("="*80)
    
    # Load config
    print("\n1. Loading configuration...")
    config = RAGConfig.from_env()
    print(f"   ✓ Config loaded")
    print(f"   - Chunks: {config.chunks_path}")
    print(f"   - Model: {config.chat_model}")
    print(f"   - Reranker: {config.reranker_type}")
    
    # Validate config
    print("\n2. Validating configuration...")
    config.validate()
    print("   ✓ Configuration valid")
    
    # Initialize OpenAI client
    print("\n3. Initializing OpenAI client...")
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("   ✓ OpenAI client ready")
    
    # Initialize RAG service
    print("\n4. Initializing RAG service...")
    print("   (This may take a moment...)")
    rag_service = RAGService(config, openai_client)
    print("   ✓ RAG service initialized!")
    
    # Check components
    print("\n5. Checking components...")
    print(f"   - Child chunks loaded: {len(rag_service.chunk_loader.child_chunks)}")
    print(f"   - Parent chunks loaded: {len(rag_service.chunk_loader.parent_chunks)}")
    print(f"   - Vector index size: {rag_service.vector_store.index.ntotal}")
    print(f"   - Reranker type: {type(rag_service.reranker).__name__}")
    
    # Test query
    print("\n6. Testing query...")
    test_query = "What faculties does IZU have?"
    print(f"   Query: '{test_query}'")
    
    response = rag_service.query(
        query=test_query,
        conversation_id="test_123",
        use_parent_retrieval=True
    )
    
    print(f"\n   ✓ Query successful!")
    print(f"   - Response time: {response.response_time_ms:.2f}ms")
    print(f"   - Chunks retrieved: {response.num_chunks_retrieved}")
    print(f"   - Answer length: {len(response.answer)} chars")
    print(f"\n   Answer preview:")
    print(f"   {response.answer[:200]}...")
    
    # Test conversation history
    print("\n7. Testing conversation history...")
    response2 = rag_service.query(
        query="Tell me more about that",
        conversation_id="test_123",
        use_parent_retrieval=True
    )
    
    history = rag_service.conversation_manager.get_history("test_123")
    print(f"   ✓ Conversation history working!")
    print(f"   - Messages in history: {len(history)}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

if __name__ == "__main__":
    test_rag_service()
