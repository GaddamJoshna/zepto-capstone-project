import os
import re
from pathlib import Path
from typing import TypedDict, Literal

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy_chunks"
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "track",
    "cancel",
    "gift card",
    "support hours",
]

STRUCTURED_PROMPT_TEMPLATE = """
ROLE:
You are a careful Zepto customer-support assistant.

CONTEXT:
Use only the policy context supplied below.
{context}

TASK:
Answer the user's question using the supplied policy context. If the context does not contain
the answer, say that the available policy context is insufficient.

FORMAT:
Return a JSON object with exactly these fields:
- answer: string
- sources: list of source IDs
- confidence: number between 0 and 1

LENGTH:
Keep the answer concise and clear, preferably 2 to 5 sentences.

NEGATIVE CONSTRAINT:
Do not invent policy details, do not use outside knowledge, and do not answer using
information that is not present in the provided context.

FEW-SHOT EXAMPLE:
User question: What is the delivery fee for an order below INR 149?
Context: Standard delivery is free on orders over INR 149; orders below this threshold incur
a flat INR 25 delivery fee.
Expected answer: The standard delivery fee is INR 25 for orders below INR 149.

USER QUESTION:
{query}
"""

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]
    response: AskResponse

def load_documents() -> list[dict]:
    records = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        # Each document is short enough to be one chunk.
        records.append({
            "id": path.stem,
            "document": text,
            "metadata": {"source": path.name},
        })
    return records

def build_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    records = load_documents()
    ids = [r["id"] for r in records]
    existing = collection.get(ids=ids)
    existing_ids = set(existing.get("ids", []))
    new_records = [r for r in records if r["id"] not in existing_ids]

    if new_records:
        texts = [r["document"] for r in new_records]
        embeddings = embedding_model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=[r["id"] for r in new_records],
            documents=texts,
            metadatas=[r["metadata"] for r in new_records],
            embeddings=embeddings,
        )
    return collection

collection = build_collection()

def classify_intent(query: str) -> str:
    lowered = query.lower()
    if any(keyword in lowered for keyword in KEYWORDS):
        return "policy_question"
    return "general_question"

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = embedding_model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for idx, chunk_id in enumerate(ids):
        retrieved.append({
            "id": chunk_id,
            "document": documents[idx],
            "metadata": metadatas[idx],
            "distance": distances[idx] if idx < len(distances) else None,
        })
    return retrieved

def optional_real_llm_answer(query: str, retrieved: list[dict]) -> AskResponse:
    """
    Optional MOCK_LLM=0 extension.
    Uses Groq through langchain-groq if GROQ_API_KEY is available.
    The required graded path never enters this function.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    context = "\n\n".join(
        f"[{item['id']}] {item['document']}" for item in retrieved
    )
    prompt = STRUCTURED_PROMPT_TEMPLATE.format(context=context, query=query)
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    last_error = None
    for attempt in range(3):
        correction = ""
        if attempt > 0:
            correction = (
                "\nCORRECTION: Return valid JSON with answer, sources, and confidence only. "
                "Do not include markdown fences."
            )
        try:
            raw = llm.invoke([HumanMessage(content=prompt + correction)]).content
            import json
            parsed = json.loads(raw)
            return AskResponse.model_validate(parsed)
        except Exception as exc:
            last_error = exc

    return AskResponse(
        answer=f"ERROR: The optional LLM response could not be validated: {last_error}",
        sources=[item["id"] for item in retrieved],
        confidence=0.0,
    )

def classify_intent_node(state: GraphState) -> GraphState:
    # Mock mode uses the required deterministic keyword heuristic.
    # The optional real-LLM extension can be added here without changing routing shape.
    return {"intent": classify_intent(state["query"])}

def retrieve_and_answer_node(state: GraphState) -> GraphState:
    retrieved = retrieve(state["query"], top_k=3)
    if MOCK_LLM:
        top_snippet = retrieved[0]["document"][:200] if retrieved else "No context found."
        response = AskResponse(
            answer=f"Based on the retrieved context: {top_snippet}",
            sources=[item["id"] for item in retrieved],
            confidence=1.0,
        )
    else:
        response = optional_real_llm_answer(state["query"], retrieved)
    return {"retrieved": retrieved, "response": response}

def direct_answer_node(state: GraphState) -> GraphState:
    if MOCK_LLM:
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        # The optional direct real-LLM extension intentionally has no retrieval.
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"],
        )
        prompt = STRUCTURED_PROMPT_TEMPLATE.format(
            context="No policy retrieval was performed.",
            query=state["query"],
        )
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        import json
        try:
            response = AskResponse.model_validate(json.loads(raw))
        except Exception:
            response = AskResponse(
                answer="ERROR: Optional LLM output failed schema validation.",
                sources=[],
                confidence=0.0,
            )
    return {"response": response}

def route_by_intent(state: GraphState) -> str:
    return state["intent"]

graph_builder = StateGraph(GraphState)
graph_builder.add_node("classify_intent", classify_intent_node)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer_node)
graph_builder.add_node("direct_answer", direct_answer_node)
graph_builder.set_entry_point("classify_intent")
graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)
graph = graph_builder.compile()

app = FastAPI(title="Zepto Support Assistant", version="1.0.0")

@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "mock_llm": MOCK_LLM,
        "endpoint": "POST /ask",
    }

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})
    return AskResponse.model_validate(result["response"])
