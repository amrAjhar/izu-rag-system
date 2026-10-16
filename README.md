# IZU RAG System

A production-oriented Retrieval-Augmented Generation (RAG) API for answering questions about Istanbul Sabahattin Zaim University (IZU). The system retrieves relevant university content from a FAISS vector index, optionally reranks the results, and uses an OpenAI chat model to generate grounded answers with source context.

It supports parent-child chunk retrieval, conversational history, English and Turkish deployments, health and statistics endpoints, and automatic index creation when a persisted vector index is unavailable.

## Features

- **Retrieval-Augmented Generation:** Answers questions using indexed IZU content instead of relying only on model knowledge.
- **Parent-child chunking:** Searches smaller child chunks while optionally returning their larger parent contexts for better answer quality.
- **FAISS vector search:** Supports cosine similarity, inner-product, and L2 distance metrics.
- **Configurable reranking:** Choose CrossEncoder reranking, LLM-based reranking, or no reranking.
- **Conversation support:** Maintains in-memory conversation history and limits it to a configurable number of messages.
- **Multilingual servers:** Separate English and Turkish entry points with dedicated ports and data configurations.
- **FastAPI interface:** Includes interactive OpenAPI documentation through `/docs`.
- **Operational endpoints:** Provides health checks, runtime statistics, conversation retrieval, and conversation cleanup.
- **Automatic index creation:** If the configured FAISS index is missing, embeddings are generated for the loaded child chunks and a new index is saved.

## Architecture

The request pipeline is organized as follows:

1. `ChunkLoader` loads child chunks from JSONL and reconstructs parent chunks.
2. `EmbeddingService` creates an embedding for the user query through the OpenAI API.
3. `VectorStore` searches the persisted FAISS index.
4. `Reranker` optionally improves the ranking using a CrossEncoder or an LLM.
5. `RetrievalEngine` converts child results into parent contexts when parent retrieval is enabled.
6. `AnswerGenerator` sends the retrieved context and conversation history to the configured OpenAI chat model.
7. `ConversationManager` stores the user and assistant messages for the conversation.

## Repository layout

```text
.
├── config/
│   └── rag_config.py              # Environment-backed RAG configuration and validation
├── data/izu/                      # IZU source chunks, indexes, datasets, and database artifacts
├── models/
│   ├── chunk_models.py             # Child, parent, search-result, and conversation models
│   └── schemas.py                  # Pydantic request and response schemas
├── services/
│   ├── answer_generator.py         # Grounded answer generation with OpenAI
│   ├── chunk_loader.py             # JSONL loading and parent reconstruction
│   ├── conversation_manager.py    # In-memory conversation history
│   ├── embedding_service.py       # OpenAI text embeddings
│   ├── rag_service.py             # Main RAG orchestration service
│   ├── reranker.py                # CrossEncoder, LLM, and no-op rerankers
│   ├── retrieval_engine.py        # Vector retrieval, reranking, and parent retrieval
│   └── vector_store.py            # FAISS index loading, creation, and search
├── rag_all_parent_child.py         # General RAG API server on port 8000
├── rag_english.py                 # English RAG API server on port 5002
├── rag_turkish.py                 # Turkish RAG API server on port 5001
├── test_rag_service.py            # Initialization, query, and conversation smoke test
├── requirements_rag.txt            # Python dependencies
└── .env.example                   # Minimal environment variable example
```

## Requirements

- Python 3.10 or newer is recommended.
- An OpenAI API key.
- The configured chunk JSONL file.
- Persisted FAISS and NumPy embedding artifacts, or permission to create them on first startup.
- `sentence-transformers` is required when using the default CrossEncoder reranker.
- `faiss-cpu` is required by the FAISS-backed vector store.

Install the declared dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements_rag.txt
pip install faiss-cpu sentence-transformers
```

## Configuration

Create a `.env` file in the repository root. The minimum required secret is:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The application also supports the following configuration variables:

| Variable | Default | Description |
| --- | --- | --- |
| `CHUNKS_PATH` | `data/izu/parent_child_chunks.jsonl` | Parent-child JSONL chunk file |
| `FAISS_INDEX_PATH` | `data/izu/faiss_parent_child.index` | FAISS index path |
| `EMBEDDINGS_PATH` | `data/izu/parent_child_embeddings.npy` | Stored embedding matrix |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat completion model |
| `TOP_K` | `10` | Number of vector candidates to retrieve |
| `DISTANCE_METRIC` | `cosine` | `cosine`, `l2`, or `ip` |
| `RERANKER` | `crossencoder` | `crossencoder`, `llm`, or `none` |
| `TEMPERATURE` | `0.3` | Generation temperature |
| `MAX_TOKENS` | `2000` | Maximum generated tokens |
| `MAX_HISTORY_LENGTH` | `6` | Number of conversation messages retained |

The language-specific entry points load `.env.english` and `.env.turkish`, respectively. Configure each file to point to the matching chunk and index artifacts.

## Running the servers

### General parent-child server

```bash
python rag_all_parent_child.py
```

Available at `http://localhost:8000` with API documentation at `http://localhost:8000/docs`.

### English server

```bash
python rag_english.py
```

Available at `http://localhost:5002`.

### Turkish server

```bash
python rag_turkish.py
```

Available at `http://localhost:5001`.

## API endpoints

All servers expose the following core endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Return service status and basic metadata |
| `GET` | `/health` | Report loaded chunks, parents, and vector count |
| `GET` | `/stats` | Return index and model configuration statistics |
| `POST` | `/chat` | Retrieve context and generate an answer |
| `GET` | `/conversation/{conversation_id}` | Retrieve in-memory conversation history |
| `DELETE` | `/conversation/{conversation_id}` | Clear a conversation |

### Chat request example

```bash
curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "What faculties does IZU have?",
    "conversation_id": "demo-conversation",
    "use_parent_retrieval": true
  }'
```

The response includes the generated answer, retrieved chunk IDs, a limited set of source contexts, the conversation ID, and response time.

## Testing

The repository includes a smoke test that initializes the configured service, performs a query, and verifies conversation history:

```bash
python test_rag_service.py
```

The test requires a valid `OPENAI_API_KEY` and working chunk/index configuration. It may create the FAISS index if the configured index files do not already exist.

## Data artifacts

The `data/izu/` directory contains the indexed IZU knowledge base and supporting artifacts, including:

- English and Turkish chunk JSONL files.
- Parent-child chunk JSONL files.
- FAISS indexes.
- NumPy embedding data.
- Test and semantic-validation datasets.
- A SQLite RAG data artifact.

Large generated artifacts are committed to the repository, so confirm that the expected data files are available before starting a server.

## Notes

- Conversation history is stored in memory and is lost when the process stops.
- CORS is currently configured to allow all origins; restrict `allow_origins` before deploying publicly.
- The answer generator instructs the model to use the supplied context and state when information is not explicitly available.
- The default reranker uses `cross-encoder/ms-marco-MiniLM-L-6-v2`; use `RERANKER=none` to avoid loading it.
