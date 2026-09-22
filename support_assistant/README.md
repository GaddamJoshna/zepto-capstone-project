
•
# Module 3 — Zepto Generative AI Support Assistant

## Overview

This module implements a small, locally runnable Zepto support assistant using:

- Sentence Transformers: `all-MiniLM-L6-v2` for local embeddings
- ChromaDB for persistent vector storage and cosine-similarity retrieval
- LangGraph `StateGraph` for intent routing
- Pydantic for validated structured responses
- FastAPI for the local HTTP API
- A deterministic offline mock mode controlled by `MOCK_LLM`

The graded baseline uses `MOCK_LLM=1` by default. It does not call an LLM provider.

## Folder structure

```text
support_assistant/
├── docs/
│   ├── doc_01_delivery.txt
│   ├── doc_02_returns_refunds.txt
│   ├── doc_03_membership.txt
│   ├── doc_04_order_tracking.txt
│   ├── doc_05_cancellation.txt
│   ├── doc_06_damaged_missing.txt
│   ├── doc_07_gift_cards.txt
│   └── doc_08_support_hours.txt
├── chroma_db/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Architecture

```text
User query
   |
   v
FastAPI POST /ask
   |
   v
LangGraph: classify_intent
   |
   +--> policy_question
   |       |
   |       v
   |   retrieve_and_answer
   |       |
   |       +--> query embedding using all-MiniLM-L6-v2
   |       +--> top-3 cosine retrieval from ChromaDB
   |       +--> mock answer or optional real-LLM answer
   |
   +--> general_question
           |
           v
       direct_answer
           |
           +--> fixed mock response or optional real-LLM response
   |
   v
Pydantic AskResponse
{answer, sources, confidence}
```

### Pipeline walkthrough

1. **Ingestion:** `load_documents()` in `main.py` reads all eight text files from `docs/`. Each short document is treated as one chunk.
2. **Embedding:** `build_collection()` uses the local `all-MiniLM-L6-v2` Sentence Transformer model to create normalized embeddings.
3. **Indexing:** The embeddings and document text are stored in the persistent ChromaDB collection named `zepto_policy_chunks` inside `chroma_db/`.
4. **Intent classification:** `classify_intent_node()` uses the required keyword heuristic in mock mode. Keywords include delivery, return, refund, membership, tracking, cancel, gift card, and support hours.
5. **Retrieval:** `retrieve_and_answer_node()` embeds the incoming policy question and retrieves the top three chunks from ChromaDB using cosine similarity.
6. **Generation:** In mock mode, the answer is generated deterministically from the first retrieved chunk using the format `Based on the retrieved context: ...`. For general questions, `direct_answer_node()` returns a fixed response.
7. **Validation:** `AskResponse` guarantees that the response contains `answer`, `sources`, and `confidence`, with confidence restricted to 0–1.
8. **API:** FastAPI exposes the graph through `POST /ask`.

### MOCK_LLM behavior

- `MOCK_LLM` unset or set to `1`: required offline deterministic mock mode. No LLM provider is called.
- `MOCK_LLM=0`: optional real-LLM extension. The code includes a Groq-based path using `GROQ_API_KEY`; this is not required for grading.

The retrieval and embedding steps run in both modes for policy questions. The final answer-generation step changes between the deterministic mock response and the optional real-LLM response.

## Structured prompt template

The actual prompt template is defined in `STRUCTURED_PROMPT_TEMPLATE` in `main.py`. It includes:

- Role
- Context
- Task
- Format
- Length
- Negative constraint against unsupported information
- Few-shot example

It is used by the optional real-LLM extension.

## Installation

From the `support_assistant` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, the first execution may download the open-source embedding model once. No LLM API key is needed for the required mock mode.

## Run the FastAPI service

Keep `MOCK_LLM` at its default:

```bash
uvicorn main:app --reload --port 7860
```

The service is available at:

```text
http://127.0.0.1:7860
```

Interactive API documentation:

```text
http://127.0.0.1:7860/docs
```

## Example calls

### 1. Policy question: retrieval route

```bash
curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery fee for orders below INR 149?"}'
```

Example response shape:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
  "sources": ["doc_01_delivery"],
  "confidence": 1.0
}
```

The exact answer snippet may contain the first 200 characters of the top retrieved document. The `sources` list contains the retrieved chunk IDs.

### 2. General question: direct route

```bash
curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'
```

Expected response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker

Build the image:

```bash
docker build -t zepto-support .
```

Run the container:

```bash
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```

Test the endpoint:

```bash
curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"How long do I have to report a damaged item?"}'
```

The Dockerfile uses the required offline mock baseline. The first container build/run may need internet access to install Python packages and obtain the embedding model. No LLM provider call is made by the application in mock mode.
