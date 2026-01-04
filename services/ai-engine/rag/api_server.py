"""
FastAPI server for RAG vector store queries
Exposes HTTP API for threat intelligence retrieval
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from vector_store import VectorStore, ThreatIntelligenceRAG

app = FastAPI(title="YUGMASTRA RAG API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
vector_store = VectorStore(store_type="chromadb")
rag = ThreatIntelligenceRAG(vector_store)


class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "evolution"
    k: int = 3
    alpha: float = 0.5


class QueryResponse(BaseModel):
    retrieved_documents: List[Dict[str, Any]]
    context: str
    query: str


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG system for relevant threat intelligence"""
    try:
        result = rag.query(
            question=request.query,
            k=request.k,
            alpha=request.alpha
        )

        return QueryResponse(
            retrieved_documents=result["retrieved_documents"],
            context=result["context"],
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG API",
        "vector_store": vector_store.store_type,
        "documents_count": len(rag.threat_intel_docs)
    }


@app.post("/add_document")
async def add_document(content: str, metadata: Dict[str, Any]):
    """Add new document to vector store"""
    try:
        vector_store.add_documents([content], [metadata])
        return {"status": "success", "message": "Document added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
