"""
YUGMĀSTRA API Gateway

FastAPI-based unified API for all services
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, Any, List
import logging

from routers import (
    evolution,
    red_team,
    blue_team,
    knowledge_graph,
    cyber_range,
    analytics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting YUGMĀSTRA API Gateway...")
    # Initialize connections to services
    # await init_database()
    # await init_redis()
    # await init_neo4j()
    yield
    logger.info("Shutting down YUGMĀSTRA API Gateway...")
    # Cleanup
    # await close_database()
    # await close_redis()
    # await close_neo4j()


app = FastAPI(
    title="YUGMĀSTRA API",
    description="Autonomous Adversary-Defender Co-Evolution Platform API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "yugmastra-api"
    }


# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "YUGMĀSTRA API Gateway",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "evolution": "/api/v1/evolution",
            "red_team": "/api/v1/red-team",
            "blue_team": "/api/v1/blue-team",
            "knowledge_graph": "/api/v1/knowledge-graph",
            "cyber_range": "/api/v1/cyber-range",
            "analytics": "/api/v1/analytics"
        }
    }


# Include routers
app.include_router(evolution.router, prefix="/api/v1/evolution", tags=["evolution"])
app.include_router(red_team.router, prefix="/api/v1/red-team", tags=["red-team"])
app.include_router(blue_team.router, prefix="/api/v1/blue-team", tags=["blue-team"])
app.include_router(knowledge_graph.router, prefix="/api/v1/knowledge-graph", tags=["knowledge-graph"])
app.include_router(cyber_range.router, prefix="/api/v1/cyber-range", tags=["cyber-range"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])


# WebSocket for real-time updates
class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            # Handle different message types
            if data.get("type") == "subscribe":
                # Subscribe to specific events
                pass

            # Broadcast to all clients
            await manager.broadcast({
                "type": "update",
                "data": data
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
