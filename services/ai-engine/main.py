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
from agents.threat_intelligence import ThreatIntelligenceAgent
from agents.vulnerability_scanner import VulnerabilityScanner
from agents.incident_response import IncidentResponseAgent
from agents.security_advisor import SecurityAdvisorAgent
from agents.orchestrator import AIOrchestrator
from agents.predictive_intel import PredictiveThreatIntelligence
from agents.code_generator import SecurityCodeGenerator
from models.llm_manager import LLMManager
from services.rag_service import RAGService
from services.vector_store import VectorStore
from utils.logger import setup_logger
import os

# Setup logging first
logger = setup_logger(__name__)

# Conditional imports for advanced features
ENABLE_KNOWLEDGE_GRAPH = os.getenv('ENABLE_KNOWLEDGE_GRAPH', 'false').lower() == 'true'
ENABLE_ZERO_DAY = os.getenv('ENABLE_ZERO_DAY_DISCOVERY', 'false').lower() == 'true'
ENABLE_SIEM = os.getenv('ENABLE_SIEM_GENERATION', 'false').lower() == 'true'

knowledge_graph = None
kg_builder = None
zero_day_engine = None
siem_generator = None
GraphNode = None
GraphEdge = None
RuleFormat = None
Severity = None

if ENABLE_KNOWLEDGE_GRAPH:
    try:
        from models.knowledge_graph import Neo4jKnowledgeGraph, GraphNode, GraphEdge, KnowledgeGraphBuilder
        knowledge_graph = Neo4jKnowledgeGraph()
        kg_builder = KnowledgeGraphBuilder(knowledge_graph)
        logger.info("Knowledge Graph enabled")
    except Exception as e:
        logger.warning(f"Knowledge Graph disabled: {str(e)}")
        ENABLE_KNOWLEDGE_GRAPH = False

if ENABLE_ZERO_DAY:
    try:
        from models.zero_day_discovery import ZeroDayDiscoveryEngine, BehaviorAnomaly
        zero_day_engine = ZeroDayDiscoveryEngine()
        logger.info("Zero-Day Discovery enabled")
    except Exception as e:
        logger.warning(f"Zero-Day Discovery disabled: {str(e)}")
        ENABLE_ZERO_DAY = False

if ENABLE_SIEM:
    try:
        from models.siem_rule_generator import SIEMRuleGeneratorEngine, RuleFormat, Severity
        siem_generator = SIEMRuleGeneratorEngine(None)  # Will initialize after llm_manager
        logger.info("SIEM Rule Generator enabled")
    except Exception as e:
        logger.warning(f"SIEM Rule Generator disabled: {str(e)}")
        ENABLE_SIEM = False

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

# Initialize core AI services
llm_manager = LLMManager()
rag_service = RAGService()
vector_store = VectorStore()

# Initialize SIEM generator if enabled (needs llm_manager)
if ENABLE_SIEM and siem_generator is not None:
    siem_generator.llm_manager = llm_manager

# Initialize agents
red_team_agent = RedTeamAgent(llm_manager, rag_service)
blue_team_agent = BlueTeamAgent(llm_manager, rag_service)
evolution_agent = EvolutionAgent(llm_manager, rag_service, vector_store)
threat_intel_agent = ThreatIntelligenceAgent(llm_manager, rag_service)
vuln_scanner_agent = VulnerabilityScanner(llm_manager, rag_service)
incident_response_agent = IncidentResponseAgent(llm_manager, rag_service)
security_advisor_agent = SecurityAdvisorAgent(llm_manager, rag_service)
predictive_intel_agent = PredictiveThreatIntelligence(llm_manager, rag_service)
code_generator_agent = SecurityCodeGenerator(llm_manager, rag_service)

# Initialize orchestrator with all agents
all_agents = {
    "red_team": red_team_agent,
    "blue_team": blue_team_agent,
    "evolution": evolution_agent,
    "threat_intelligence": threat_intel_agent,
    "vulnerability_scanner": vuln_scanner_agent,
    "incident_response": incident_response_agent,
    "security_advisor": security_advisor_agent,
    "predictive_intel": predictive_intel_agent,
    "code_generator": code_generator_agent,
}
orchestrator = AIOrchestrator(all_agents)

# Helper functions for feature checks
def check_knowledge_graph_enabled():
    if not ENABLE_KNOWLEDGE_GRAPH or knowledge_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge Graph feature is disabled. Set ENABLE_KNOWLEDGE_GRAPH=true and ensure Neo4j is running."
        )

def check_zero_day_enabled():
    if not ENABLE_ZERO_DAY or zero_day_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Zero-Day Discovery feature is disabled. Set ENABLE_ZERO_DAY_DISCOVERY=true to enable."
        )

def check_siem_enabled():
    if not ENABLE_SIEM or siem_generator is None:
        raise HTTPException(
            status_code=503,
            detail="SIEM Rule Generator feature is disabled. Set ENABLE_SIEM_GENERATION=true to enable."
        )

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
    services = {
        "llm": llm_manager.is_ready(),
        "rag": rag_service.is_ready(),
        "vector_store": vector_store.is_ready(),
    }

    # Core AI Agents
    agents = {
        "red_team": red_team_agent is not None,
        "blue_team": blue_team_agent is not None,
        "evolution": evolution_agent is not None,
        "threat_intelligence": threat_intel_agent is not None,
        "vulnerability_scanner": vuln_scanner_agent is not None,
        "incident_response": incident_response_agent is not None,
        "security_advisor": security_advisor_agent is not None,
        "predictive_intel": predictive_intel_agent is not None,
        "code_generator": code_generator_agent is not None,
        "orchestrator": orchestrator is not None,
    }

    # Only check advanced features if enabled
    if ENABLE_KNOWLEDGE_GRAPH and knowledge_graph is not None:
        services["knowledge_graph"] = knowledge_graph.driver is not None or knowledge_graph.fallback_graph is not None

    if ENABLE_ZERO_DAY and zero_day_engine is not None:
        services["zero_day_engine"] = zero_day_engine.isolation_forest.is_trained

    if ENABLE_SIEM and siem_generator is not None:
        services["siem_generator"] = True

    return {
        "status": "healthy",
        "services": services,
        "agents": agents,
        "agent_count": len([a for a in agents.values() if a]),
        "features": {
            "knowledge_graph": ENABLE_KNOWLEDGE_GRAPH,
            "zero_day_discovery": ENABLE_ZERO_DAY,
            "siem_generation": ENABLE_SIEM,
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

# ==================== Knowledge Graph Endpoints ====================

@app.get("/api/knowledge-graph")
async def get_knowledge_graph(
    node_type: Optional[str] = None,
    limit: int = 100
):
    """
    Get knowledge graph data for visualization
    """
    if not ENABLE_KNOWLEDGE_GRAPH or knowledge_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge Graph feature is disabled. Set ENABLE_KNOWLEDGE_GRAPH=true to enable."
        )

    try:
        stats = knowledge_graph.get_graph_statistics()

        # For NetworkX fallback, convert to node/edge format
        if knowledge_graph.fallback_graph:
            import networkx as nx

            nodes = []
            for node_id, data in knowledge_graph.fallback_graph.nodes(data=True):
                if node_type and data.get('type') != node_type:
                    continue

                nodes.append({
                    'id': node_id,
                    'type': data.get('type', 'unknown'),
                    'label': data.get('properties', {}).get('name', node_id),
                    'properties': data.get('properties', {})
                })

                if len(nodes) >= limit:
                    break

            edges = []
            for source, target, data in knowledge_graph.fallback_graph.edges(data=True):
                edges.append({
                    'source': source,
                    'target': target,
                    'relationship': data.get('relationship', 'related_to'),
                    'weight': data.get('weight', 1.0)
                })

            return {
                "nodes": nodes[:limit],
                "edges": edges[:limit * 2],
                "statistics": stats
            }

        # For Neo4j, query the database
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT $limit
        """

        knowledge_graph.query_cypher(query, {"limit": limit})

        return {
            "nodes": [],
            "edges": [],
            "statistics": stats,
            "message": "Neo4j implementation - data processing needed"
        }

    except Exception as e:
        logger.error(f"Knowledge graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-graph/query")
async def query_knowledge_graph(query: str):
    """
    Natural language query to knowledge graph
    """
    if not ENABLE_KNOWLEDGE_GRAPH or knowledge_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge Graph feature is disabled. Set ENABLE_KNOWLEDGE_GRAPH=true and ensure Neo4j is running."
        )

    try:
        results = knowledge_graph.natural_language_query(query)

        return {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Knowledge graph query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-graph/nodes")
async def add_graph_node(
    id: str,
    type: str,
    properties: Dict[str, Any]
):
    """
    Add a node to the knowledge graph
    """
    check_knowledge_graph_enabled()

    try:
        node = GraphNode(
            id=id,
            type=type,
            properties=properties
        )

        success = knowledge_graph.add_node(node)

        if success:
            return {
                "status": "success",
                "node_id": id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add node")

    except Exception as e:
        logger.error(f"Add node error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-graph/edges")
async def add_graph_edge(
    source: str,
    target: str,
    relationship: str,
    weight: float = 1.0,
    properties: Optional[Dict[str, Any]] = None
):
    """
    Add an edge to the knowledge graph
    """
    check_knowledge_graph_enabled()

    try:
        edge = GraphEdge(
            source=source,
            target=target,
            relationship=relationship,
            weight=weight,
            properties=properties or {}
        )

        success = knowledge_graph.add_edge(edge)

        if success:
            return {
                "status": "success",
                "edge": f"{source} -> {target}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add edge")

    except Exception as e:
        logger.error(f"Add edge error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-graph/attack-chains/{technique_id}")
async def get_attack_chains(technique_id: str, max_depth: int = 5):
    """
    Find attack chains starting from a technique
    """
    try:
        chains = knowledge_graph.find_attack_chains(technique_id, max_depth)

        return {
            "technique": technique_id,
            "chains": chains,
            "count": len(chains),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Attack chains error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-graph/mitigations/{attack_id}")
async def get_mitigations(attack_id: str):
    """
    Find mitigations for an attack
    """
    try:
        mitigations = knowledge_graph.find_mitigations(attack_id)

        return {
            "attack": attack_id,
            "mitigations": mitigations,
            "count": len(mitigations),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Mitigations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-graph/similar/{node_id}")
async def get_similar_nodes(node_id: str, top_k: int = 5):
    """
    Find similar nodes using embeddings or structural similarity
    """
    try:
        similar = knowledge_graph.find_similar_nodes(node_id, top_k)

        return {
            "node": node_id,
            "similar_nodes": [{"id": s[0], "similarity": s[1]} for s in similar],
            "count": len(similar),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Similar nodes error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-graph/ingest/mitre")
async def ingest_mitre_data(mitre_data: Dict[str, Any]):
    """
    Ingest MITRE ATT&CK data into knowledge graph
    """
    try:
        count = kg_builder.add_mitre_attack_data(mitre_data)

        return {
            "status": "success",
            "nodes_added": count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"MITRE ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-graph/ingest/battle")
async def ingest_battle_results(battle_data: Dict[str, Any]):
    """
    Ingest battle results into knowledge graph
    """
    try:
        count = kg_builder.add_battle_results(battle_data)

        return {
            "status": "success",
            "nodes_added": count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Battle ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-graph/ingest/cve")
async def ingest_cve_data(cve_data: List[Dict[str, Any]]):
    """
    Ingest CVE vulnerability data into knowledge graph
    """
    try:
        count = kg_builder.add_vulnerability_data(cve_data)

        return {
            "status": "success",
            "nodes_added": count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"CVE ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-graph/statistics")
async def get_graph_statistics():
    """
    Get knowledge graph statistics
    """
    try:
        stats = knowledge_graph.get_graph_statistics()

        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Zero-Day Discovery Endpoints ====================

@app.post("/api/zero-day/train")
async def train_zero_day_models(
    normal_behavior_data: List[List[float]],
    normal_sequences: Optional[List[List[List[float]]]] = None
):
    """
    Train zero-day detection models on normal system behavior
    """
    try:
        import numpy as np

        behavior_array = np.array(normal_behavior_data)
        sequence_array = np.array(normal_sequences) if normal_sequences else None

        results = await zero_day_engine.train_models(behavior_array, sequence_array)

        return {
            "status": "training_complete",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Zero-day training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/zero-day/analyze/behavior")
async def analyze_behavior(
    behavior_data: List[List[float]],
    metadata: List[Dict[str, Any]]
):
    """
    Analyze system behavior for anomalies
    """
    try:
        import numpy as np

        behavior_array = np.array(behavior_data)
        anomalies = await zero_day_engine.analyze_system_behavior(behavior_array, metadata)

        return {
            "status": "analysis_complete",
            "anomalies": [
                {
                    "id": a.id,
                    "component": a.component,
                    "type": a.anomaly_type,
                    "score": a.anomaly_score,
                    "deviation": a.deviation,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in anomalies
            ],
            "count": len(anomalies),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Behavior analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/zero-day/analyze/sequences")
async def analyze_sequences(
    sequence_data: List[List[List[float]]],
    metadata: List[Dict[str, Any]]
):
    """
    Analyze execution sequences for anomalies
    """
    try:
        import numpy as np

        sequence_array = np.array(sequence_data)
        anomalies = await zero_day_engine.analyze_execution_sequences(sequence_array, metadata)

        return {
            "status": "analysis_complete",
            "anomalies": [
                {
                    "id": a.id,
                    "component": a.component,
                    "type": a.anomaly_type,
                    "score": a.anomaly_score,
                    "deviation": a.deviation,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in anomalies
            ],
            "count": len(anomalies),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Sequence analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/zero-day/discover")
async def discover_vulnerabilities(
    attack_patterns: List[Dict[str, Any]]
):
    """
    Discover potential zero-day vulnerabilities from anomalies and patterns
    """
    try:
        # Get recent anomalies
        recent_anomalies = zero_day_engine.behavior_anomalies[-100:]  # Last 100 anomalies

        vulnerabilities = await zero_day_engine.discover_vulnerabilities(
            recent_anomalies,
            attack_patterns
        )

        return {
            "status": "discovery_complete",
            "vulnerabilities": [
                {
                    "id": v.id,
                    "type": v.type,
                    "severity": v.severity,
                    "confidence": v.confidence,
                    "description": v.description,
                    "affected_component": v.affected_component,
                    "detection_method": v.detection_method,
                    "discovered_at": v.discovered_at.isoformat(),
                    "cvss_score": v.cvss_score
                }
                for v in vulnerabilities
            ],
            "count": len(vulnerabilities),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Vulnerability discovery error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/zero-day/vulnerabilities")
async def get_vulnerabilities(
    min_severity: float = 0.0,
    min_confidence: float = 0.0
):
    """
    Get discovered zero-day vulnerabilities with filtering
    """
    try:
        vulnerabilities = zero_day_engine.get_discovered_vulnerabilities(
            min_severity=min_severity,
            min_confidence=min_confidence
        )

        return {
            "vulnerabilities": [
                {
                    "id": v.id,
                    "type": v.type,
                    "severity": v.severity,
                    "confidence": v.confidence,
                    "description": v.description,
                    "affected_component": v.affected_component,
                    "detection_method": v.detection_method,
                    "discovered_at": v.discovered_at.isoformat(),
                    "cvss_score": v.cvss_score,
                    "exploit_available": v.exploit_available
                }
                for v in vulnerabilities
            ],
            "count": len(vulnerabilities),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get vulnerabilities error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/zero-day/statistics")
async def get_zero_day_statistics():
    """
    Get zero-day discovery statistics
    """
    try:
        stats = zero_day_engine.get_statistics()

        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Zero-day statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SIEM Rule Generator Endpoints ====================

@app.post("/api/siem/generate-rule")
async def generate_siem_rule(
    attack_pattern: Dict[str, Any],
    format: str = "sigma",
    enhance_with_llm: bool = False
):
    """
    Generate SIEM detection rule from attack pattern

    Supported formats: sigma, splunk, elastic, suricata, snort, yara
    """
    try:
        rule_format = RuleFormat(format.lower())

        rule = await siem_generator.generate_rule(
            attack_pattern=attack_pattern,
            rule_format=rule_format,
            enhance_with_llm=enhance_with_llm
        )

        return {
            "status": "success",
            "rule": {
                "id": rule.id,
                "title": rule.title,
                "description": rule.description,
                "severity": rule.severity.value,
                "format": rule.format.value,
                "content": rule.rule_content,
                "tags": rule.tags,
                "mitre_attack": rule.mitre_attack,
                "created_at": rule.created_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"SIEM rule generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/siem/batch-generate")
async def batch_generate_rules(
    attack_patterns: List[Dict[str, Any]],
    formats: Optional[List[str]] = None
):
    """
    Generate SIEM rules in multiple formats for multiple attack patterns
    """
    try:
        # Convert string formats to RuleFormat enums
        rule_formats = None
        if formats:
            rule_formats = [RuleFormat(fmt.lower()) for fmt in formats]

        results = await siem_generator.batch_generate_rules(
            attack_patterns=attack_patterns,
            formats=rule_formats
        )

        # Convert results to serializable format
        output = {}
        for fmt, rules in results.items():
            output[fmt.value] = [
                {
                    "id": r.id,
                    "title": r.title,
                    "severity": r.severity.value,
                    "format": r.format.value,
                    "content": r.rule_content
                }
                for r in rules
            ]

        return {
            "status": "success",
            "results": output,
            "total_rules": sum(len(r) for r in results.values()),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch rule generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/siem/rules")
async def get_siem_rules(
    format: Optional[str] = None,
    severity: Optional[str] = None
):
    """
    Get generated SIEM rules with optional filtering
    """
    try:
        rule_format = RuleFormat(format.lower()) if format else None
        rule_severity = Severity(severity.lower()) if severity else None

        rules = siem_generator.get_rules(
            rule_format=rule_format,
            severity=rule_severity
        )

        return {
            "rules": [
                {
                    "id": r.id,
                    "title": r.title,
                    "description": r.description,
                    "severity": r.severity.value,
                    "format": r.format.value,
                    "content": r.rule_content,
                    "tags": r.tags,
                    "mitre_attack": r.mitre_attack,
                    "created_at": r.created_at.isoformat(),
                    "author": r.author
                }
                for r in rules
            ],
            "count": len(rules),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get SIEM rules error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/siem/statistics")
async def get_siem_statistics():
    """
    Get SIEM rule generation statistics
    """
    try:
        stats = siem_generator.get_statistics()

        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"SIEM statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/siem/formats")
async def get_supported_formats():
    """
    Get list of supported SIEM rule formats
    """
    return {
        "formats": [fmt.value for fmt in RuleFormat],
        "severities": [sev.value for sev in Severity],
        "timestamp": datetime.now().isoformat()
    }

# ==================== Threat Intelligence Endpoints ====================

@app.post("/api/threat-intel/analyze-ioc")
async def analyze_ioc(ioc: str, ioc_type: str):
    """Analyze Indicator of Compromise"""
    try:
        result = await threat_intel_agent.analyze_ioc(ioc, ioc_type)
        return result
    except Exception as e:
        logger.error(f"IOC analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threat-intel/map-mitre")
async def map_to_mitre(attack_description: str):
    """Map attack to MITRE ATT&CK framework"""
    try:
        result = await threat_intel_agent.map_to_mitre(attack_description)
        return result
    except Exception as e:
        logger.error(f"MITRE mapping error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threat-intel/profile-actor")
async def profile_threat_actor(actor_info: str):
    """Profile a threat actor"""
    try:
        result = await threat_intel_agent.profile_threat_actor(actor_info)
        return result
    except Exception as e:
        logger.error(f"Actor profiling error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threat-intel/identify-malware")
async def identify_malware(malware_data: str):
    """Identify malware family"""
    try:
        result = await threat_intel_agent.identify_malware(malware_data)
        return result
    except Exception as e:
        logger.error(f"Malware identification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Vulnerability Scanner Endpoints ====================

@app.post("/api/vuln-scan/code")
async def scan_code(code: str, language: str):
    """Scan code for vulnerabilities"""
    try:
        result = await vuln_scanner_agent.scan_code(code, language)
        return result
    except Exception as e:
        logger.error(f"Code scan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vuln-scan/config")
async def assess_configuration(config: Dict[str, Any], service_type: str):
    """Assess security configuration"""
    try:
        result = await vuln_scanner_agent.assess_configuration(config, service_type)
        return result
    except Exception as e:
        logger.error(f"Config assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vuln-scan/dependencies")
async def check_dependencies(dependencies: List[Dict[str, str]]):
    """Check dependencies for vulnerabilities"""
    try:
        result = await vuln_scanner_agent.check_dependencies(dependencies)
        return result
    except Exception as e:
        logger.error(f"Dependency check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vuln-scan/cvss")
async def calculate_cvss(vulnerability_details: str):
    """Calculate CVSS score"""
    try:
        result = await vuln_scanner_agent.calculate_cvss(vulnerability_details)
        return result
    except Exception as e:
        logger.error(f"CVSS calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Incident Response Endpoints ====================

@app.post("/api/incident/detect")
async def detect_incident(events: List[Dict[str, Any]]):
    """Detect and classify incidents"""
    try:
        result = await incident_response_agent.detect_incident(events)
        return result
    except Exception as e:
        logger.error(f"Incident detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/incident/root-cause")
async def perform_root_cause_analysis(incident_data: Dict[str, Any]):
    """Perform root cause analysis"""
    try:
        result = await incident_response_agent.perform_root_cause_analysis(incident_data)
        return result
    except Exception as e:
        logger.error(f"Root cause analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/incident/playbook")
async def generate_response_playbook(incident_type: str, severity: str):
    """Generate incident response playbook"""
    try:
        result = await incident_response_agent.generate_response_playbook(incident_type, severity)
        return result
    except Exception as e:
        logger.error(f"Playbook generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/incident/timeline")
async def reconstruct_timeline(events: List[Dict[str, Any]]):
    """Reconstruct incident timeline"""
    try:
        result = await incident_response_agent.reconstruct_timeline(events)
        return result
    except Exception as e:
        logger.error(f"Timeline reconstruction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Security Advisor Endpoints ====================

@app.post("/api/advisor/review-architecture")
async def review_architecture(architecture: str):
    """Review security architecture"""
    try:
        result = await security_advisor_agent.review_architecture(architecture)
        return result
    except Exception as e:
        logger.error(f"Architecture review error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/advisor/assess-compliance")
async def assess_compliance(framework: str, current_state: str):
    """Assess compliance with security framework"""
    try:
        result = await security_advisor_agent.assess_compliance(framework, current_state)
        return result
    except Exception as e:
        logger.error(f"Compliance assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/advisor/risk-assessment")
async def perform_risk_assessment(assets: List[Dict[str, Any]]):
    """Perform risk assessment"""
    try:
        result = await security_advisor_agent.perform_risk_assessment(assets)
        return result
    except Exception as e:
        logger.error(f"Risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/advisor/security-roadmap")
async def create_security_roadmap(current_state: str, target_state: str, timeline: str):
    """Create security roadmap"""
    try:
        result = await security_advisor_agent.create_security_roadmap(current_state, target_state, timeline)
        return result
    except Exception as e:
        logger.error(f"Roadmap creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/advisor/maturity-assessment")
async def assess_security_maturity(organization_info: str):
    """Assess security maturity"""
    try:
        result = await security_advisor_agent.assess_security_maturity(organization_info)
        return result
    except Exception as e:
        logger.error(f"Maturity assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AI Orchestrator Endpoints ====================

@app.post("/api/orchestrator/workflow")
async def execute_workflow(workflow_type: str, context: Dict[str, Any]):
    """Execute multi-agent workflow"""
    try:
        result = await orchestrator.execute_workflow(workflow_type, context)
        return result
    except Exception as e:
        logger.error(f"Workflow execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orchestrator/status")
async def get_orchestrator_status():
    """Get orchestrator status"""
    try:
        status = orchestrator.get_status()
        return status
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Predictive Intelligence Endpoints ====================

@app.post("/api/predictive/future-attacks")
async def predict_future_attacks(timeframe: str, context: str):
    """Predict future cyber attacks"""
    try:
        result = await predictive_intel_agent.predict_future_attacks(timeframe, context)
        return result
    except Exception as e:
        logger.error(f"Attack prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictive/emerging-threats")
async def forecast_emerging_threats(technology: str):
    """Forecast emerging threats for new technology"""
    try:
        result = await predictive_intel_agent.forecast_emerging_threats(technology)
        return result
    except Exception as e:
        logger.error(f"Threat forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictive/vulnerability-trends")
async def analyze_vulnerability_trends(historical_data: List[Dict[str, Any]]):
    """Analyze vulnerability trends"""
    try:
        result = await predictive_intel_agent.analyze_vulnerability_trends(historical_data)
        return result
    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictive/industry-threats")
async def forecast_industry_threats(industry: str, timeframe: str):
    """Forecast industry-specific threats"""
    try:
        result = await predictive_intel_agent.forecast_industry_threats(industry, timeframe)
        return result
    except Exception as e:
        logger.error(f"Industry forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictive/ransomware-trends")
async def predict_ransomware_trends():
    """Predict ransomware trends"""
    try:
        result = await predictive_intel_agent.predict_ransomware_trends()
        return result
    except Exception as e:
        logger.error(f"Ransomware prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictive/threat-calendar")
async def generate_threat_calendar(year: int):
    """Generate predictive threat calendar"""
    try:
        result = await predictive_intel_agent.generate_threat_calendar(year)
        return result
    except Exception as e:
        logger.error(f"Calendar generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Code Generator Endpoints ====================

@app.post("/api/codegen/secure-api")
async def generate_secure_api(spec: Dict[str, Any]):
    """Generate secure API endpoint code"""
    try:
        result = await code_generator_agent.generate_secure_api(spec)
        return result
    except Exception as e:
        logger.error(f"API generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/auth-system")
async def generate_auth_system(requirements: str):
    """Generate authentication system"""
    try:
        result = await code_generator_agent.generate_auth_system(requirements)
        return result
    except Exception as e:
        logger.error(f"Auth system generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/encryption")
async def generate_encryption_code(use_case: str, language: str = "python"):
    """Generate encryption code"""
    try:
        result = await code_generator_agent.generate_encryption_code(use_case, language)
        return result
    except Exception as e:
        logger.error(f"Encryption code generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/input-validation")
async def generate_input_validation(fields: List[Dict[str, str]], language: str = "python"):
    """Generate input validation functions"""
    try:
        result = await code_generator_agent.generate_input_validation(fields, language)
        return result
    except Exception as e:
        logger.error(f"Validation generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/security-tests")
async def generate_security_tests(code: str, language: str):
    """Generate security tests"""
    try:
        result = await code_generator_agent.generate_security_tests(code, language)
        return result
    except Exception as e:
        logger.error(f"Test generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/waf-rules")
async def generate_waf_rules(attack_patterns: List[str]):
    """Generate WAF rules"""
    try:
        result = await code_generator_agent.generate_waf_rules(attack_patterns)
        return result
    except Exception as e:
        logger.error(f"WAF rules generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/dockerfile")
async def generate_secure_dockerfile(base_requirements: str):
    """Generate secure Dockerfile"""
    try:
        result = await code_generator_agent.generate_secure_dockerfile(base_requirements)
        return result
    except Exception as e:
        logger.error(f"Dockerfile generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen/kubernetes")
async def generate_kubernetes_security(app_spec: Dict[str, Any]):
    """Generate secure Kubernetes manifests"""
    try:
        result = await code_generator_agent.generate_kubernetes_security(app_spec)
        return result
    except Exception as e:
        logger.error(f"Kubernetes manifest generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
