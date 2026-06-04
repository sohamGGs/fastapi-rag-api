# FastAPI RAG API

Ask natural language questions. Get answers grounded in your documents.

## Quick Start

```bash
pip install -r requirements.txt
python ingest.py          # populate ChromaDB from data/*.txt
uvicorn main:app --reload # start the API
```

Open http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ping` | Health check |
| POST | `/ask/` | RAG question answering |
| POST | `/search/` | Raw semantic search |
| GET | `/ask/info` | Pipeline status |
| GET | `/search/info` | Collection stats |

## Progressive Fallback

The app runs at every stage of installation:

| Packages installed | Mode |
|-------------------|------|
| fastapi only | Mock retrieval + Python mock LLM |
| + langchain | Mock retrieval + FakeListLLM |
| + chromadb + ingest run | Real retrieval + FakeListLLM |
| + sentence-transformers | Real retrieval + real embeddings |
| + OPENAI_API_KEY | Fully operational |

Check `GET /ask/info` to see which mode is active.

## Stack

- **FastAPI** — async web framework
- **LangChain LCEL** — prompt chaining
- **ChromaDB** — vector storage
- **all-MiniLM-L6-v2** — sentence embeddings