from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import AI modules
from agents.red_team import RedTeamAgent
from agents.blue_team import BlueTeamAgent
from agents.evolution import EvolutionAgent
from models.llm_manager import LLMManager
from services.rag_service import RAGService
from services.vector_store import VectorStore
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YUGMĀSTRA AI Engine",
    description="Advanced AI/ML Engine for Cybersecurity Intelligence",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI services
llm_manager = LLMManager()
rag_service = RAGService()
vector_store = VectorStore()

# Initialize agents
red_team_agent = RedTeamAgent(llm_manager, rag_service)
blue_team_agent = BlueTeamAgent(llm_manager, rag_service)
evolution_agent = EvolutionAgent(llm_manager, rag_service, vector_store)

# Request/Response Models
class Message(BaseModel):
    role: str
    content: str

class AIRequest(BaseModel):
    message: str
    mode: str  # 'red-team', 'blue-team', 'evolution'
    history: Optional[List[Message]] = []
    context: Optional[Dict[str, Any]] = {}

class AIResponse(BaseModel):
    response: str
    mode: str
    confidence: float
    sources: List[str] = []
    timestamp: str

class TrainingRequest(BaseModel):
    dataset_path: str
    model_type: str
    epochs: int = 3
    batch_size: int = 8

class EmbeddingRequest(BaseModel):
    texts: List[str]
    store: bool = True

# Health check
@app.get("/")
async def root():
    return {
        "service": "YUGMĀSTRA AI Engine",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "llm": llm_manager.is_ready(),
            "rag": rag_service.is_ready(),
            "vector_store": vector_store.is_ready(),
        }
    }

# Main AI endpoint
@app.post("/api/ai/chat", response_model=AIResponse)
async def ai_chat(request: AIRequest):
    """
    Main AI chat endpoint supporting multiple agent modes
    """
    try:
        logger.info(f"Processing AI request - Mode: {request.mode}")

        # Route to appropriate agent
        if request.mode == "red-team":
            response = await red_team_agent.generate_response(
                message=request.message,
                history=request.history,
                context=request.context
            )
        elif request.mode == "blue-team":
            response = await blue_team_agent.generate_response(
                message=request.message,
                history=request.history,
                context=request.context
            )
        elif request.mode == "evolution":
            response = await evolution_agent.generate_response(
                message=request.message,
                history=request.history,
                context=request.context
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid AI mode")

        return AIResponse(
            response=response["text"],
            mode=request.mode,
            confidence=response.get("confidence", 0.95),
            sources=response.get("sources", []),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"AI chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# LLM Training endpoint
@app.post("/api/ai/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train custom LLM model on cybersecurity data
    """
    try:
        logger.info(f"Starting model training - Type: {request.model_type}")

        # Add training to background tasks
        background_tasks.add_task(
            llm_manager.train_model,
            dataset_path=request.dataset_path,
            model_type=request.model_type,
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        return {
            "status": "training_started",
            "model_type": request.model_type,
            "message": "Model training initiated in background"
        }

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Embedding endpoint
@app.post("/api/ai/embed")
async def create_embeddings(request: EmbeddingRequest):
    """
    Create vector embeddings for text data
    """
    try:
        embeddings = await vector_store.embed_texts(request.texts)

        if request.store:
            await vector_store.store_embeddings(request.texts, embeddings)

        return {
            "status": "success",
            "count": len(embeddings),
            "stored": request.store
        }

    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG query endpoint
@app.post("/api/ai/rag/query")
async def rag_query(query: str, top_k: int = 5):
    """
    Query RAG system for relevant context
    """
    try:
        results = await rag_service.query(query, top_k=top_k)

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge ingestion endpoint
@app.post("/api/ai/ingest")
async def ingest_knowledge(
    documents: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Ingest cybersecurity knowledge into RAG system
    """
    try:
        logger.info(f"Ingesting {len(documents)} documents")

        if background_tasks:
            background_tasks.add_task(
                rag_service.ingest_documents,
                documents=documents,
                metadata=metadata
            )

            return {
                "status": "ingestion_started",
                "count": len(documents)
            }
        else:
            await rag_service.ingest_documents(documents, metadata)

            return {
                "status": "success",
                "count": len(documents)
            }

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model info endpoint
@app.get("/api/ai/models")
async def get_models():
    """
    Get information about loaded AI models
    """
    try:
        models_info = llm_manager.get_models_info()

        return {
            "models": models_info,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Models info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
