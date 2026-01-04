"""
Real-Time WebSocket Server for YUGMASTRA
Handles live battle updates, collaborative features, real-time notifications
"""

import asyncio
import json
import logging
from typing import Dict, Set
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YUGMASTRA WebSocket Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""

    def __init__(self):
        # Active connections per battle
        self.battles: Dict[str, Set[WebSocket]] = {}
        # User metadata
        self.users: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, battle_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()

        if battle_id not in self.battles:
            self.battles[battle_id] = set()

        self.battles[battle_id].add(websocket)
        self.users[websocket] = {
            "battle_id": battle_id,
            "connected_at": datetime.now().isoformat()
        }

        logger.info(f"Client connected to battle {battle_id}. Total: {len(self.battles[battle_id])}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.users:
            battle_id = self.users[websocket]["battle_id"]
            self.battles[battle_id].discard(websocket)
            del self.users[websocket]

            logger.info(f"Client disconnected from battle {battle_id}. Remaining: {len(self.battles[battle_id])}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_json(message)

    async def broadcast(self, message: dict, battle_id: str):
        """Broadcast message to all clients in battle"""
        if battle_id in self.battles:
            disconnected = set()

            for connection in self.battles[battle_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn)

    async def broadcast_to_all(self, message: dict):
        """Broadcast to all battles"""
        for battle_id in self.battles:
            await self.broadcast(message, battle_id)


manager = ConnectionManager()


@app.websocket("/battle/{battle_id}")
async def battle_websocket(websocket: WebSocket, battle_id: str):
    """WebSocket endpoint for battle arena"""
    await manager.connect(websocket, battle_id)

    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "payload": {
                "battle_id": battle_id,
                "message": "Connected to battle arena",
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }, websocket)

        # Handle incoming messages
        while True:
            data = await websocket.receive_json()

            message_type = data.get("type")
            payload = data.get("payload", {})

            logger.info(f"Received {message_type} from battle {battle_id}")

            # Handle different message types
            if message_type == "ping":
                # Respond to heartbeat
                await manager.send_personal_message({
                    "type": "pong",
                    "payload": {"timestamp": datetime.now().isoformat()},
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }, websocket)

            elif message_type == "join_spectator":
                # User joined as spectator
                username = payload.get("username", "Anonymous")
                await manager.broadcast({
                    "type": "spectator_joined",
                    "payload": {"username": username},
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }, battle_id)

            elif message_type == "manual_defense":
                # Manual defense command
                command = payload.get("command")
                target = payload.get("target")

                # Broadcast defense action to all clients
                await manager.broadcast({
                    "type": "defense",
                    "payload": {
                        "action": command,
                        "target": target,
                        "source": "manual",
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }, battle_id)

            elif message_type == "attack":
                # Broadcast attack to all spectators
                await manager.broadcast({
                    "type": "attack",
                    "payload": payload,
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }, battle_id)

            elif message_type == "score_update":
                # Broadcast score update
                await manager.broadcast({
                    "type": "score_update",
                    "payload": payload,
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }, battle_id)

            elif message_type == "health_update":
                # Broadcast system health
                await manager.broadcast({
                    "type": "health_update",
                    "payload": payload,
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }, battle_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({
            "type": "spectator_left",
            "payload": {},
            "timestamp": int(datetime.now().timestamp() * 1000)
        }, battle_id)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    total_connections = sum(len(connections) for connections in manager.battles.values())

    return {
        "status": "healthy",
        "battles": len(manager.battles),
        "total_connections": total_connections,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/battles")
async def list_battles():
    """List active battles"""
    return {
        "battles": [
            {
                "battle_id": battle_id,
                "connections": len(connections)
            }
            for battle_id, connections in manager.battles.items()
        ]
    }


# Background task to simulate battle events (for testing)
async def simulate_battle_events():
    """Simulate attack/defense events for testing"""
    import random

    attack_types = [
        "SQL Injection", "XSS Attack", "RCE", "Privilege Escalation",
        "Lateral Movement", "Data Exfiltration", "DDoS", "Phishing"
    ]

    targets = ["web_server", "database", "api_gateway", "auth_service"]

    while True:
        await asyncio.sleep(random.randint(2, 5))

        # Generate random attack
        attack = {
            "type": "attack",
            "payload": {
                "id": f"attack-{int(datetime.now().timestamp())}",
                "type": random.choice(attack_types),
                "target": random.choice(targets),
                "severity": random.choice(["low", "medium", "high", "critical"]),
                "technique": f"T{random.randint(1000, 1999)}.{random.randint(1, 9):03d}"
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }

        # Broadcast to all battles
        await manager.broadcast_to_all(attack)

        # Simulate defense response (70% success rate)
        await asyncio.sleep(1)

        if random.random() > 0.3:
            defense = {
                "type": "defense",
                "payload": {
                    "action": random.choice([
                        "Blocked by firewall",
                        "Detected by IDS",
                        "ML model detected",
                        "Connection terminated"
                    ]),
                    "attack_id": attack["payload"]["id"],
                    "effectiveness": random.uniform(0.6, 0.99)
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            }

            await manager.broadcast_to_all(defense)


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Uncomment to enable battle simulation
    # asyncio.create_task(simulate_battle_events())
    logger.info("WebSocket server started")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
