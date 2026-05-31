# FastAPI RAG API 🚀

A production-grade REST API and RAG (Retrieval-Augmented Generation) system 
built with FastAPI, LangChain, and ChromaDB.

Built as a 7-day structured learning project.

## Tech Stack
- **FastAPI** – Modern, high-performance web framework
- **Pydantic** – Data validation and settings management
- **Uvicorn** – ASGI server
- *(Coming soon)* LangChain, ChromaDB, OpenAI

## Getting Started

### Prerequisites
- Python 3.11+

### Installation
```bash
git clone https://github.com/sohamGGs/fastapi-rag-api.git
cd fastapi-rag-api
python -m venv venv
Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the API
```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ping` | Health check |
| GET | `/items/` | List all items (with filters) |
| GET | `/items/{id}` | Get item by ID |
| POST | `/items/` | Create new item |
| PUT | `/items/{id}` | Update item |
| DELETE | `/items/{id}` | Delete item |

## Day-by-Day Progress
- [x] Day 1 – FastAPI Foundations, CRUD, Pydantic Models
- [ ] Day 2 – Project Structure, Dependencies, Database Layer
- [ ] Day 3 – Authentication & Middleware
- [ ] Day 4 – LangChain & Embeddings
- [ ] Day 5 – ChromaDB & Vector Search
- [ ] Day 6 – RAG Pipeline
- [ ] Day 7 – Deployment