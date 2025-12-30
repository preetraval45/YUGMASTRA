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
from models.knowledge_graph import Neo4jKnowledgeGraph, GraphNode, GraphEdge, KnowledgeGraphBuilder
from models.zero_day_discovery import ZeroDayDiscoveryEngine, BehaviorAnomaly
from models.siem_rule_generator import SIEMRuleGeneratorEngine, RuleFormat, Severity
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
knowledge_graph = Neo4jKnowledgeGraph()
kg_builder = KnowledgeGraphBuilder(knowledge_graph)
zero_day_engine = ZeroDayDiscoveryEngine()
siem_generator = SIEMRuleGeneratorEngine(llm_manager)

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
            "knowledge_graph": knowledge_graph.driver is not None or knowledge_graph.fallback_graph is not None,
            "zero_day_engine": zero_day_engine.isolation_forest.is_trained,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
