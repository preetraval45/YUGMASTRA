"""
YUGMĀSTRA Live Attack Simulation API
====================================
Endpoints for real-time attack/defense simulation
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio
import json

# Import simulation engines
from models.live_simulation_engine import (
    LiveAttackSimulationEngine,
    DefenseRole,
    AttackStatus,
    HackerPersona,
    AssetType
)
from models.threat_intelligence import (
    ThreatIntelligenceEngine,
    ThreatSeverity,
    ThreatType
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YUGMĀSTRA Live Simulation Engine",
    description="Real-time Cyber Warfare Simulation Platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
simulation_engine = LiveAttackSimulationEngine()
threat_intel_engine = ThreatIntelligenceEngine()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_event(self, session_id: str, event: Dict[str, Any]):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(event)
            except Exception as e:
                logger.error(f"Error sending event: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()


# ==================== Pydantic Models ====================

class CreateSimulationRequest(BaseModel):
    scenario_id: str
    user_role: str = "soc_analyst"

class DefenseActionRequest(BaseModel):
    action_type: str
    target_asset_id: Optional[str] = None
    attack_id: Optional[str] = None

class ThreatIntelRequest(BaseModel):
    sources: Optional[List[str]] = None

class SearchIndicatorsRequest(BaseModel):
    query: str
    indicator_type: Optional[str] = None
    threat_type: Optional[str] = None
    min_confidence: int = 0


# ==================== Health & Status ====================

@app.get("/")
async def root():
    return {
        "service": "YUGMĀSTRA Live Simulation Engine",
        "status": "operational",
        "version": "2.0.0",
        "features": [
            "Multi-agent hacker personas",
            "Real-time attack simulation",
            "Cyber range with user roles",
            "Threat intelligence integration",
            "Live visualization",
            "Attack analytics & reporting"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "simulation_engine": len(simulation_engine.scenarios) > 0,
            "threat_intelligence": len(threat_intel_engine.threat_feeds) > 0,
            "active_simulations": len(simulation_engine.active_simulations)
        },
        "timestamp": datetime.now().isoformat()
    }


# ==================== Live Simulation Endpoints ====================

@app.get("/api/simulation/scenarios")
async def list_scenarios():
    """List available simulation scenarios"""
    scenarios = []
    for scenario_id, scenario in simulation_engine.scenarios.items():
        scenarios.append({
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "duration_minutes": scenario.duration_minutes,
            "difficulty": scenario.difficulty,
            "attacker_count": len(scenario.attacker_agents),
            "asset_count": len(scenario.network_assets),
            "objectives": scenario.objectives
        })

    return {
        "scenarios": scenarios,
        "count": len(scenarios)
    }

@app.post("/api/simulation/create")
async def create_simulation(request: CreateSimulationRequest):
    """Create a new live simulation session"""
    try:
        # Validate user role
        try:
            user_role = DefenseRole(request.user_role)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid user role: {request.user_role}")

        # Create simulation
        session_id = await simulation_engine.create_simulation(
            scenario_id=request.scenario_id,
            user_role=user_role
        )

        return {
            "status": "created",
            "session_id": session_id,
            "message": "Live simulation started",
            "websocket_url": f"/api/simulation/{session_id}/ws"
        }

    except Exception as e:
        logger.error(f"Error creating simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulation/{session_id}/status")
async def get_simulation_status(session_id: str):
    """Get current simulation status"""
    try:
        status = await simulation_engine.get_simulation_status(session_id)
        return status

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/simulation/{session_id}/action")
async def take_defense_action(session_id: str, request: DefenseActionRequest):
    """Take a defense action in the simulation"""
    try:
        action = await simulation_engine.take_defense_action(
            session_id=session_id,
            action_type=request.action_type,
            target_asset_id=request.target_asset_id,
            attack_id=request.attack_id
        )

        return {
            "status": "success",
            "action_id": action.id,
            "effectiveness": action.effectiveness,
            "timestamp": action.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Error taking defense action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulation/{session_id}/events")
async def get_simulation_events(
    session_id: str,
    since: Optional[str] = None
):
    """Get simulation events"""
    try:
        since_dt = datetime.fromisoformat(since) if since else None
        events = await simulation_engine.get_live_events(session_id, since_dt)

        return {
            "events": events,
            "count": len(events)
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.websocket("/api/simulation/{session_id}/ws")
async def simulation_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time simulation events"""
    await manager.connect(websocket, session_id)

    try:
        # Send initial status
        status = await simulation_engine.get_simulation_status(session_id)
        await manager.send_event(session_id, {
            "type": "status",
            "data": status
        })

        # Stream events in real-time
        last_event_time = datetime.now()

        while True:
            # Get new events
            events = await simulation_engine.get_live_events(session_id, last_event_time)

            for event in events:
                await manager.send_event(session_id, event)

            if events:
                last_event_time = datetime.fromisoformat(events[-1]["timestamp"])

            # Check if simulation is still running
            status = await simulation_engine.get_simulation_status(session_id)
            if not status["is_running"]:
                await manager.send_event(session_id, {
                    "type": "simulation_complete",
                    "data": status
                })
                break

            await asyncio.sleep(2)  # Poll every 2 seconds

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)


# ==================== Threat Intelligence Endpoints ====================

@app.post("/api/threat-intel/fetch")
async def fetch_threat_intelligence(request: ThreatIntelRequest):
    """Fetch latest threat intelligence from configured sources"""
    try:
        results = await threat_intel_engine.fetch_threat_intelligence(request.sources)

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching threat intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threat-intel/search")
async def search_indicators(request: SearchIndicatorsRequest):
    """Search threat indicators"""
    try:
        threat_type = ThreatType(request.threat_type) if request.threat_type else None

        indicators = await threat_intel_engine.search_indicators(
            query=request.query,
            indicator_type=request.indicator_type,
            threat_type=threat_type,
            min_confidence=request.min_confidence
        )

        return {
            "indicators": [
                {
                    "id": ind.id,
                    "type": ind.type,
                    "value": ind.value,
                    "threat_type": ind.threat_type,
                    "severity": ind.severity,
                    "confidence": ind.confidence,
                    "sources": ind.sources,
                    "tags": ind.tags,
                    "first_seen": ind.first_seen.isoformat(),
                    "last_seen": ind.last_seen.isoformat()
                }
                for ind in indicators
            ],
            "count": len(indicators)
        }

    except Exception as e:
        logger.error(f"Error searching indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threat-intel/enrich/{indicator_value}")
async def enrich_indicator(indicator_value: str, indicator_type: str):
    """Enrich an indicator with threat intelligence"""
    try:
        enrichment = await threat_intel_engine.enrich_indicator(
            indicator_value=indicator_value,
            indicator_type=indicator_type
        )

        return enrichment

    except Exception as e:
        logger.error(f"Error enriching indicator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threat-intel/campaigns/active")
async def get_active_campaigns():
    """Get currently active threat campaigns"""
    try:
        campaigns = await threat_intel_engine.get_active_campaigns()

        return {
            "campaigns": [
                {
                    "id": c.id,
                    "name": c.name,
                    "threat_actors": c.threat_actors,
                    "start_date": c.start_date.isoformat(),
                    "targets": c.targets,
                    "objectives": c.objectives,
                    "techniques": c.techniques,
                    "indicator_count": len(c.indicators)
                }
                for c in campaigns
            ],
            "count": len(campaigns)
        }

    except Exception as e:
        logger.error(f"Error getting campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threat-intel/actor/{actor_id}")
async def get_threat_actor_profile(actor_id: str):
    """Get detailed threat actor profile"""
    try:
        actor = await threat_intel_engine.get_threat_actor_profile(actor_id)

        if not actor:
            raise HTTPException(status_code=404, detail="Threat actor not found")

        return {
            "actor": {
                "id": actor.id,
                "name": actor.name,
                "aliases": actor.aliases,
                "country": actor.country,
                "motivation": actor.motivation,
                "sophistication": actor.sophistication,
                "targets": actor.targets,
                "ttps": actor.ttps,
                "tools": actor.tools,
                "campaigns": actor.campaigns,
                "first_seen": actor.first_seen.isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting threat actor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threat-intel/correlate")
async def correlate_indicators(indicators: List[str]):
    """Correlate multiple indicators"""
    try:
        correlation = await threat_intel_engine.correlate_indicators(indicators)
        return correlation

    except Exception as e:
        logger.error(f"Error correlating indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threat-intel/vulnerabilities/exploited")
async def get_exploited_vulnerabilities():
    """Get vulnerabilities known to be exploited in the wild"""
    try:
        vulns = await threat_intel_engine.get_exploited_vulnerabilities()

        return {
            "vulnerabilities": [
                {
                    "cve_id": v.cve_id,
                    "description": v.description,
                    "severity": v.severity,
                    "cvss_score": v.cvss_score,
                    "affected_products": v.affected_products,
                    "published_date": v.published_date.isoformat()
                }
                for v in vulns
            ],
            "count": len(vulns)
        }

    except Exception as e:
        logger.error(f"Error getting exploited vulnerabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threat-intel/statistics")
async def get_threat_intel_statistics():
    """Get threat intelligence statistics"""
    try:
        stats = threat_intel_engine.get_statistics()
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Analytics & Reporting ====================

@app.get("/api/analytics/simulation/{session_id}")
async def get_simulation_analytics(session_id: str):
    """Get detailed analytics for a simulation session"""
    try:
        status = await simulation_engine.get_simulation_status(session_id)

        # Calculate additional analytics
        if session_id not in simulation_engine.active_simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")

        session = simulation_engine.active_simulations[session_id]

        # Attack pattern analysis
        attack_phases = {}
        for attack in session.attack_actions:
            phase = attack.technique.phase
            attack_phases[phase] = attack_phases.get(phase, 0) + 1

        # Attacker effectiveness
        attacker_stats = {}
        for attack in session.attack_actions:
            persona = attack.attacker_persona
            if persona not in attacker_stats:
                attacker_stats[persona] = {
                    "total_attempts": 0,
                    "successful": 0,
                    "detected": 0,
                    "blocked": 0
                }

            attacker_stats[persona]["total_attempts"] += 1
            if attack.status == AttackStatus.SUCCESS:
                attacker_stats[persona]["successful"] += 1
            elif attack.status == AttackStatus.DETECTED:
                attacker_stats[persona]["detected"] += 1
            elif attack.status == AttackStatus.BLOCKED:
                attacker_stats[persona]["blocked"] += 1

        # Timeline of events
        timeline = []
        for event in session.events[-50:]:  # Last 50 events
            timeline.append({
                "timestamp": event["timestamp"],
                "type": event["type"],
                "description": str(event.get("data", ""))[:100]
            })

        return {
            "session_id": session_id,
            "analytics": {
                "overall_score": status["score"],
                "detection_rate": status["stats"]["attacks_detected"] / max(status["stats"]["total_attacks"], 1),
                "block_rate": status["stats"]["attacks_blocked"] / max(status["stats"]["attacks_detected"], 1),
                "compromise_rate": len(status["compromised_assets"]) / max(len(session.scenario.network_assets), 1),
                "attack_phases": attack_phases,
                "attacker_effectiveness": attacker_stats,
                "timeline": timeline
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
