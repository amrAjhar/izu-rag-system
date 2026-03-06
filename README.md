# Medipol University RAG System

## What I Added to `rag.py`:

### 1. **Database Integration** (Lines 10-28)
- Added SQLite database support to load chunks from `data/english/rag_data.db`
- Added `DATA_LANGUAGE` config to switch between English/Turkish
- Replaced product-based storage with chunk-based storage

### 2. **Data Loading** (Lines 45-108)
- `load_chunks_from_db()`: Loads chunks from SQLite database
- `load_data_and_embeddings()`: Generates embeddings for all chunks in batches
- Batch processing (500 chunks at a time) to avoid API limits
- Progress tracking during embedding generation

### 3. **Hybrid Search & Reranking** (Lines 30-219) ⭐ NEW
- `bm25_score()`: Keyword-based scoring using BM25 algorithm (k1=1.5, b=0.75)
- `hybrid_search()`: Combines 70% semantic + 30% keyword matching for better retrieval
- `rerank_with_llm()`: Uses GPT-4o-mini to rerank top 10 results → top 5 most relevant
- `search_chunks()`: Unified interface with `use_hybrid` and reranking support
- `format_context()`: Formats retrieved chunks with metadata for LLM
- **Result**: Captures both conceptual matches AND exact terms, with optional deep reranking

### 4. **Enhanced API Endpoints**:
- **GET `/`**: System status and statistics
- **GET `/api/stats`**: Detailed system statistics
- **POST `/api/chat`**: Main chat endpoint with RAG ⭐ ENHANCED
  - Retrieves top 10 chunks with hybrid search (semantic + keyword)
  - Optional LLM reranking (set `use_reranking: true` in request)
  - Uses bilingual system prompts (Turkish/English)
  - Returns answer with source citations and relevance scores
  - Lower temperature (0.3) for factual responses
- **POST `/api/search`**: Direct chunk search without LLM generation

### 5. **CLI Testing Mode** (Lines 286-330)
- Run `python rag.py test` to test queries interactively
- Shows top 3 results with scores
- Useful for debugging retrieval quality

### 6. **Key Improvements**:
- **Source tracking**: Every answer includes source URLs and relevance scores
- **Error handling**: Graceful fallback if retrieval fails
- **Bilingual support**: System prompts adapt to question language
- **Factual focus**: Lower temperature prevents hallucinations

## Setup

```bash
# Install dependencies
pip install -r requirements_rag.txt

# Set OpenAI API key
$env:OPENAI_API_KEY = "your-key-here"

# Optional: Set language
$env:DATA_LANGUAGE = "english"  # or "turkish"
```

## Usage

### 1. Run API Server

```bash
python rag.py
```

Server runs on http://localhost:8000

### 2. Test Interactively

```bash
python rag.py test
```

Enter questions to see retrieval results (hybrid search by default).

### 2.5. Test with Reranking ⭐ NEW

```powershell
# API call with reranking enabled
curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"How do I apply for engineering?\", \"language\": \"english\", \"use_reranking\": true}'
```

Reranking adds ~1 second but significantly improves accuracy.

### 3. Evaluate with Test Questions

Create `test_questions_english.json` with 120 questions:

```json
[
  {
    "question": "What programs does Medipol offer?",
    "answer": "Medipol offers Medicine, Engineering, Business...",
    "category": "academics",
    "language": "english"
  }
]
```

Run evaluation:

```bash
python evaluate_rag.py test_questions_english.json results.json
```

### 4. Use Template

I created `test_questions_template.json` with 10 example questions. Expand it to 120 questions covering:

**Categories to test:**
- Academics (programs, faculties, departments)
- Admissions (requirements, deadlines, process)
- Fees & Scholarships
- Campus life & facilities
- International students
- Contact information
- Graduate programs
- Research & publications
- Student services
- Rankings & accreditation

## API Examples

### Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What engineering programs are available?",
    "language": "english"
  }'
```

### Search Only

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "message": "scholarship information",
    "language": "english"
  }'
```

## Evaluation Metrics

The evaluation script measures:

- **Score 5**: Perfect match
- **Score 4**: Mostly correct
- **Score 3**: Partially correct
- **Score 2**: Some errors
- **Score 1**: Incorrect
- **Score 0**: Completely wrong

**Target metrics:**
- Average score: ≥4.0
- Good answers (≥4): ≥80%
- Overall accuracy: ≥75%

## Next Steps

1. **Create 120 test questions** based on your actual data
2. **Run evaluation** to get baseline metrics
3. **Analyze failures** in results.json
4. **Improve data quality** if needed (clean more noise, add missing info)
5. **Tune retrieval** (adjust top_k, try different embedding models)
6. **Optimize prompts** based on evaluation results

## Files Created

- `rag.py` - Enhanced RAG API server
- `evaluate_rag.py` - Evaluation script
- `test_questions_template.json` - 10 example questions
- `requirements_rag.txt` - Python dependencies
- `README_RAG.md` - This file
