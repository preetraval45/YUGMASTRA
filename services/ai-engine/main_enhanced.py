from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Import AI modules
from agents.red_team import RedTeamAgent
from agents.blue_team import BlueTeamAgent
from agents.evolution import EvolutionAgent
from models.llm_manager import LLMManager
from models.knowledge_graph import Neo4jKnowledgeGraph, GraphNode, GraphEdge, KnowledgeGraphBuilder
from models.zero_day_discovery import ZeroDayDiscoveryEngine, BehaviorAnomaly
from models.siem_rule_generator import SIEMRuleGeneratorEngine, RuleFormat, Severity
from services.rag_service import RAGService
from services.vector_store import VectorStore
from utils.logger import setup_logger
from utils.config import get_settings
from utils.retry import async_retry_on_failure, CircuitBreaker
from utils.redis_queue import RedisQueue
from utils.cache import RedisCache
from utils.streaming import stream_llm_response, create_sse_response
from middleware.rate_limiter import RateLimitMiddleware
from middleware.request_id import RequestIDMiddleware
from middleware.logging_middleware import LoggingMiddleware

logger = setup_logger(__name__)
settings = None
redis_client = None
cache = None
request_queue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, redis_client, cache, request_queue

    logger.info("Starting YUGMASTRA AI Engine...")
    settings = get_settings()
    redis_client = await aioredis.from_url(
        settings.redis.url,
        encoding="utf-8",
        decode_responses=True
    )
    logger.info("Redis connected")
    cache = RedisCache(settings.redis.url, settings.monitoring.enable_metrics)
    request_queue = RedisQueue(settings.redis.url)
    await request_queue.connect()
    logger.info("Request queue initialized")

    logger.info("AI Engine startup complete")

    yield

    logger.info("Shutting down AI Engine...")

    if redis_client:
        await redis_client.close()

    if cache:
        await cache.disconnect()

    if request_queue:
        await request_queue.disconnect()

    logger.info("AI Engine shutdown complete")



app = FastAPI(
    title="YUGMĀSTRA AI Engine",
    description="Production-Ready AI/ML Engine for Cybersecurity Intelligence",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_llm_manager = None
_rag_service = None
_vector_store = None
_knowledge_graph = None
_kg_builder = None
_zero_day_engine = None
_siem_generator = None
_red_team_agent = None
_blue_team_agent = None
_evolution_agent = None


def get_llm_manager() -> LLMManager:
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# Circuit breakers for external services
ollama_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)


# Request/Response Models
class Message(BaseModel):
    role: str
    content: str


class AIRequest(BaseModel):
    message: str
    mode: str  # 'red-team', 'blue-team', 'evolution'
    history: Optional[List[Message]] = []
    context: Optional[Dict[str, Any]] = {}
    stream: Optional[bool] = False


class AIResponse(BaseModel):
    response: str
    mode: str
    confidence: float
    sources: List[str] = []
    timestamp: str
    cached: bool = False


class TrainingRequest(BaseModel):
    dataset_path: str
    model_type: str
    epochs: int = 3
    batch_size: int = 8


class EmbeddingRequest(BaseModel):
    texts: List[str]
    store: bool = True


@app.get("/")
async def root():
    return {
        "service": "YUGMĀSTRA AI Engine",
        "status": "operational",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    llm_manager = get_llm_manager()
    rag_service = get_rag_service()
    vector_store = get_vector_store()

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": False,
            "llm": False,
            "rag": False,
            "vector_store": False,
            "queue": False,
            "cache": False
        }
    }

    try:
        if redis_client:
            await redis_client.ping()
            health_status["services"]["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")

    try:
        health_status["services"]["llm"] = llm_manager.is_ready()
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")

    try:
        health_status["services"]["rag"] = rag_service.is_ready()
    except Exception as e:
        logger.error(f"RAG health check failed: {str(e)}")

    try:
        health_status["services"]["vector_store"] = vector_store.is_ready()
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")

    try:
        if request_queue:
            health_status["services"]["queue"] = True
    except Exception as e:
        logger.error(f"Queue health check failed: {str(e)}")

    try:
        if cache:
            health_status["services"]["cache"] = True
    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}")
    all_healthy = all(health_status["services"].values())
    health_status["status"] = "healthy" if all_healthy else "degraded"

    return health_status


@app.get("/ready")
async def readiness_check():
    llm_manager = get_llm_manager()

    if not llm_manager.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"status": "ready", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
async def metrics():
    metrics_data = {
        "queue_length": 0,
        "processing_count": 0,
        "cache_size": 0
    }

    if request_queue:
        try:
            metrics_data["queue_length"] = await request_queue.get_queue_length()
            metrics_data["processing_count"] = await request_queue.get_processing_count()
        except Exception as e:
            logger.error(f"Error getting queue metrics: {str(e)}")

    return metrics_data


@app.post("/api/ai/chat", response_model=AIResponse)
@async_retry_on_failure(max_attempts=3, exceptions=(HTTPException,))
async def ai_chat(request: AIRequest, req: Request):
    try:
        logger.info(f"Processing AI request - Mode: {request.mode}, Stream: {request.stream}")
        cache_key = f"{request.mode}:{hash(request.message)}"
        if not request.stream and cache:
            cached_response = await cache.get(cache_key, namespace="ai_chat")
            if cached_response:
                logger.info(f"Cache hit for {cache_key}")
                return AIResponse(**cached_response, cached=True)
        llm_manager = get_llm_manager()
        rag_service = get_rag_service()

        if request.mode == "red-team":
            global _red_team_agent
            if _red_team_agent is None:
                _red_team_agent = RedTeamAgent(llm_manager, rag_service)
            agent = _red_team_agent
        elif request.mode == "blue-team":
            global _blue_team_agent
            if _blue_team_agent is None:
                _blue_team_agent = BlueTeamAgent(llm_manager, rag_service)
            agent = _blue_team_agent
        elif request.mode == "evolution":
            global _evolution_agent
            if _evolution_agent is None:
                vector_store = get_vector_store()
                _evolution_agent = EvolutionAgent(llm_manager, rag_service, vector_store)
            agent = _evolution_agent
        else:
            raise HTTPException(status_code=400, detail="Invalid AI mode")
        response = await agent.generate_response(
            message=request.message,
            history=request.history,
            context=request.context
        )

        result = AIResponse(
            response=response["text"],
            mode=request.mode,
            confidence=response.get("confidence", 0.95),
            sources=response.get("sources", []),
            timestamp=datetime.now().isoformat(),
            cached=False
        )
        if not request.stream and cache:
            await cache.set(
                cache_key,
                result.dict(exclude={"cached"}),
                ttl=3600,
                namespace="ai_chat"
            )

        return result

    except Exception as e:
        logger.error(f"AI chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/chat/stream")
async def ai_chat_stream(request: AIRequest):
    try:
        logger.info(f"Processing streaming AI request - Mode: {request.mode}")
        llm_manager = get_llm_manager()
        rag_service = get_rag_service()

        if request.mode == "red-team":
            global _red_team_agent
            if _red_team_agent is None:
                _red_team_agent = RedTeamAgent(llm_manager, rag_service)
            agent = _red_team_agent
        elif request.mode == "blue-team":
            global _blue_team_agent
            if _blue_team_agent is None:
                _blue_team_agent = BlueTeamAgent(llm_manager, rag_service)
            agent = _blue_team_agent
        elif request.mode == "evolution":
            global _evolution_agent
            if _evolution_agent is None:
                vector_store = get_vector_store()
                _evolution_agent = EvolutionAgent(llm_manager, rag_service, vector_store)
            agent = _evolution_agent
        else:
            raise HTTPException(status_code=400, detail="Invalid AI mode")
        async def response_generator():
            response = await agent.generate_response(
                message=request.message,
                history=request.history,
                context=request.context
            )

            async for chunk in stream_llm_response(
                lambda: response,
            ):
                yield chunk

        return create_sse_response(response_generator())

    except Exception as e:
        logger.error(f"Streaming chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/queue/submit")
async def submit_to_queue(request: AIRequest):
    try:
        task_id = await request_queue.enqueue(
            task_data=request.dict(),
            priority=1 if request.mode == "red-team" else 0
        )

        return {
            "task_id": task_id,
            "status": "queued",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Queue submit error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/queue/status/{task_id}")
async def get_queue_status(task_id: str):
    try:
        status = await request_queue.get_task_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail="Task not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Queue status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=4,
        log_level="info"
    )
