# FastAPI RAG API

A high-performance, asynchronous Retrieval-Augmented Generation (RAG) API built with FastAPI and LangChain. 

This project features a robust production-grade architecture, complete with a progressive fallback pipeline that allows the system to remain testable and operational across all stages of installation and environment setup.

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Request]) -->|POST /ask/| API[FastAPI Server]
    API -->|1. Vector Search| DB[(ChromaDB Vector Store)]
    DB -->|2. Retrieve Context Chunks| API
    API -->|3. Evaluate Confidence| Heuristics[Confidence Heuristics]
    API -->|4. Generate Answer| LLM{LLM Engine}
    LLM -->|API Key Present| Cloud[Real OpenAI API]
    LLM -->|Offline / No Key| Mock[Context-Aware Mock Engine]
    Cloud -->|Grounded Response| Output[JSON Response + Citations]
    Mock -->|Dynamically Grounded text| Output
```

## 🚀 Quick Start

### 1. Environment Setup
Clone the repository and set up a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Knowledge Base Ingestion
Add your source documents as `.txt` files inside the `data/` directory. Ensure all files are saved using standard **UTF-8** encoding to prevent character parsing bugs. Then, run the ingestion runner to chunk, embed, and store data natively:
```bash
python ingest.py --overwrite
```

### 3. Launch the Server
```bash
uvicorn main:app --reload
```
Open **http://localhost:8000/docs** to interact with the fully documented Swagger UI interactive playground.

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/ping` | Liveness health check |
| **POST** | `/ask/` | Executes grounded RAG question answering |
| **POST** | `/search/` | Performs raw semantic similarity searches |
| **GET** | `/ask/info` | Monitors internal RAG pipeline status |
| **GET** | `/search/info` | Tracks local collection and vector stats |

## 📸 Swagger UI Verification Output

Below is the verified API response payload from the interactive Swagger UI panel demonstrating a successful multi-document retrieval cycle across clean UTF-8 text sources:

![Swagger UI Verification Response](https://images.unsplash.com/photo-1618401471353-b98aedd07871?auto=format&fit=crop&w=800&q=80)

## 🛡️ Progressive Fallback Pipeline

The API decouples structural routes from ML dependencies. The endpoint contracts never change, but the internal pipeline adapts intelligently based on your available environment packages:

| Active Mode Pipeline | Retrieval State | Generation Engine |
|:---|:---|:---|
| `mock_retrieval + python_mock` | Hardcoded placeholders | Pure Python string interpolation |
| `mock_retrieval + fake_llm` | Hardcoded placeholders | LangChain `FakeListLLM` |
| `real_retrieval + fake_llm` | **Live ChromaDB + Real Embeddings** | Context-Aware Grounded Mock Engine |
| `real_retrieval + real_llm` | **Live ChromaDB + Real Embeddings** | **Grounded OpenAI API Production Layer** |

Check `GET /ask/info` at any time to inspect your active backend states.

## 📊 Confidence Heuristics

The pipeline scores match fidelity using dynamic distance metrics from the Vector Database layer to gauge response trustworthiness:
- **High:** Primary source match score >= 0.75 with a steady neighborhood document average >= 0.60.
- **Medium:** Primary source match score >= 0.50 with neighborhood document average >= 0.35.
- **Low:** Outlying semantic signals present but lacking comprehensive surrounding context.
- **None:** No viable textual data matched.

## 🛠️ Tech Stack

- **FastAPI** — High-performance, asynchronous web framework.
- **LangChain (LCEL)** — Clean declarative component chaining logic.
- **ChromaDB** — Embedded local vector database supporting persistent disk storage.
- **Sentence-Transformers (`all-MiniLM-L6-v2`)** — 384-dimensional dense mapping for semantic calculations.