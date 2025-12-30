"""
Advanced Knowledge Graph with Neo4j
Multi-dimensional graph with temporal and probabilistic edges
Uses: Neo4j (FREE Community Edition), NetworkX (FREE), Node2Vec (FREE)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    id: str
    type: str  # attack, defense, vulnerability, asset, technique, actor
    properties: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class GraphEdge:
    source: str
    target: str
    relationship: str  # mitigates, exploits, detects, leads_to, similar_to
    weight: float = 1.0
    properties: Dict[str, Any] = None
    temporal: bool = False
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.temporal and self.timestamp is None:
            self.timestamp = datetime.utcnow()

class Neo4jKnowledgeGraph:
    """
    Advanced Knowledge Graph using Neo4j
    Supports multi-dimensional relationships and temporal tracking
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "yugmastra"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.fallback_graph = None  # NetworkX fallback if Neo4j unavailable

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("✅ Neo4j connection established")
            self._initialize_schema()
        except ImportError:
            logger.warning("⚠️ Neo4j driver not installed. Using NetworkX fallback. Install with: pip install neo4j")
            self._initialize_fallback()
        except Exception as e:
            logger.warning(f"⚠️ Neo4j connection failed: {e}. Using NetworkX fallback.")
            self._initialize_fallback()

    def _initialize_fallback(self):
        """Initialize NetworkX as fallback graph database"""
        try:
            import networkx as nx
            self.fallback_graph = nx.MultiDiGraph()
            logger.info("✅ NetworkX fallback graph initialized")
        except ImportError:
            logger.error("❌ NetworkX not available. Install with: pip install networkx")
            raise

    def _initialize_schema(self):
        """Create indexes and constraints in Neo4j"""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Attack) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Defense) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Asset) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Technique) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Actor) REQUIRE n.id IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")

            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (n:Attack) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Defense) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Vulnerability) ON (n.cve_id)",
            ]

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")

        logger.info("✅ Neo4j schema initialized")

    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the knowledge graph"""
        if self.driver:
            return self._add_node_neo4j(node)
        else:
            return self._add_node_fallback(node)

    def _add_node_neo4j(self, node: GraphNode) -> bool:
        """Add node to Neo4j"""
        try:
            with self.driver.session() as session:
                query = f"""
                MERGE (n:{node.type.capitalize()} {{id: $id}})
                SET n += $properties
                SET n.created_at = datetime($created_at)
                RETURN n
                """

                result = session.run(
                    query,
                    id=node.id,
                    properties=node.properties,
                    created_at=node.created_at.isoformat()
                )

                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to add node to Neo4j: {e}")
            return False

    def _add_node_fallback(self, node: GraphNode) -> bool:
        """Add node to NetworkX fallback"""
        try:
            self.fallback_graph.add_node(
                node.id,
                type=node.type,
                properties=node.properties,
                embeddings=node.embeddings,
                created_at=node.created_at
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add node to fallback graph: {e}")
            return False

    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the knowledge graph"""
        if self.driver:
            return self._add_edge_neo4j(edge)
        else:
            return self._add_edge_fallback(edge)

    def _add_edge_neo4j(self, edge: GraphEdge) -> bool:
        """Add edge to Neo4j"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                CREATE (source)-[r:{edge.relationship.upper()} {{
                    weight: $weight,
                    properties: $properties,
                    temporal: $temporal,
                    timestamp: datetime($timestamp)
                }}]->(target)
                RETURN r
                """

                result = session.run(
                    query,
                    source_id=edge.source,
                    target_id=edge.target,
                    weight=edge.weight,
                    properties=json.dumps(edge.properties),
                    temporal=edge.temporal,
                    timestamp=edge.timestamp.isoformat() if edge.timestamp else None
                )

                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to add edge to Neo4j: {e}")
            return False

    def _add_edge_fallback(self, edge: GraphEdge) -> bool:
        """Add edge to NetworkX fallback"""
        try:
            self.fallback_graph.add_edge(
                edge.source,
                edge.target,
                relationship=edge.relationship,
                weight=edge.weight,
                properties=edge.properties,
                temporal=edge.temporal,
                timestamp=edge.timestamp
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add edge to fallback graph: {e}")
            return False

    def query_cypher(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query"""
        if not self.driver:
            logger.warning("Cypher queries only available with Neo4j")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []

    def find_attack_chains(self, start_technique: str, max_depth: int = 5) -> List[List[str]]:
        """Find all attack chains starting from a technique"""
        if self.driver:
            query = """
            MATCH path = (start:Technique {id: $technique_id})-[:LEADS_TO*1..%d]->(end:Technique)
            RETURN [node in nodes(path) | node.id] as chain,
                   reduce(w = 1.0, rel in relationships(path) | w * rel.weight) as probability
            ORDER BY probability DESC
            LIMIT 10
            """ % max_depth

            results = self.query_cypher(query, {"technique_id": start_technique})
            return [r["chain"] for r in results]
        else:
            # NetworkX fallback - simple path finding
            import networkx as nx
            chains = []
            try:
                for target in self.fallback_graph.nodes():
                    if target == start_technique:
                        continue
                    try:
                        paths = nx.all_simple_paths(
                            self.fallback_graph,
                            start_technique,
                            target,
                            cutoff=max_depth
                        )
                        chains.extend(list(paths)[:10])
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
            except Exception as e:
                logger.error(f"Failed to find attack chains: {e}")

            return chains[:10]

    def find_mitigations(self, attack_id: str) -> List[Dict[str, Any]]:
        """Find all mitigations for an attack"""
        if self.driver:
            query = """
            MATCH (attack:Attack {id: $attack_id})<-[:MITIGATES]-(defense:Defense)
            RETURN defense.id as id, defense.name as name, defense.properties as properties
            ORDER BY defense.effectiveness DESC
            """

            return self.query_cypher(query, {"attack_id": attack_id})
        else:
            # NetworkX fallback
            mitigations = []
            try:
                for predecessor in self.fallback_graph.predecessors(attack_id):
                    for edge_data in self.fallback_graph.get_edge_data(predecessor, attack_id).values():
                        if edge_data.get('relationship') == 'mitigates':
                            node_data = self.fallback_graph.nodes[predecessor]
                            mitigations.append({
                                'id': predecessor,
                                'name': node_data.get('properties', {}).get('name', predecessor),
                                'properties': node_data.get('properties', {})
                            })
            except Exception as e:
                logger.error(f"Failed to find mitigations: {e}")

            return mitigations

    def find_similar_nodes(self, node_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar nodes using embeddings"""
        if self.driver:
            # For Neo4j, we'd use graph embeddings (Node2Vec)
            # This requires additional processing
            logger.warning("Similarity search requires Node2Vec embeddings - computing on demand")
            return self._compute_node2vec_similarity(node_id, top_k)
        else:
            # NetworkX fallback - use structural similarity
            return self._compute_structural_similarity(node_id, top_k)

    def _compute_structural_similarity(self, node_id: str, top_k: int) -> List[Tuple[str, float]]:
        """Compute structural similarity using NetworkX"""
        import networkx as nx

        try:
            similarities = []
            node_neighbors = set(self.fallback_graph.neighbors(node_id))

            for other_node in self.fallback_graph.nodes():
                if other_node == node_id:
                    continue

                other_neighbors = set(self.fallback_graph.neighbors(other_node))

                # Jaccard similarity
                if len(node_neighbors) == 0 and len(other_neighbors) == 0:
                    similarity = 0.0
                else:
                    intersection = len(node_neighbors & other_neighbors)
                    union = len(node_neighbors | other_neighbors)
                    similarity = intersection / union if union > 0 else 0.0

                similarities.append((other_node, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Failed to compute structural similarity: {e}")
            return []

    def _compute_node2vec_similarity(self, node_id: str, top_k: int) -> List[Tuple[str, float]]:
        """Compute Node2Vec embeddings and find similar nodes"""
        logger.info("Node2Vec similarity computation not yet implemented")
        return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if self.driver:
            with self.driver.session() as session:
                # Count nodes by type
                node_counts = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as type, count(n) as count
                """)

                # Count relationships by type
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relationship, count(r) as count
                """)

                return {
                    "nodes": {record["type"]: record["count"] for record in node_counts},
                    "relationships": {record["relationship"]: record["count"] for record in rel_counts}
                }
        else:
            # NetworkX fallback
            node_types = {}
            for node, data in self.fallback_graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1

            rel_types = {}
            for u, v, data in self.fallback_graph.edges(data=True):
                rel_type = data.get('relationship', 'unknown')
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

            return {
                "nodes": node_types,
                "relationships": rel_types,
                "total_nodes": self.fallback_graph.number_of_nodes(),
                "total_edges": self.fallback_graph.number_of_edges()
            }

    def natural_language_query(self, question: str) -> Dict[str, Any]:
        """
        Convert natural language question to Cypher query
        This is a simplified implementation - production would use LLM
        """
        question_lower = question.lower()

        # Simple pattern matching for common questions
        if "mitigate" in question_lower or "defense" in question_lower:
            # Extract attack name from question
            words = question_lower.split()
            if len(words) > 2:
                attack_name = " ".join(words[-2:])
                return {
                    "query_type": "mitigation",
                    "results": self.find_mitigations(attack_name)
                }

        elif "attack chain" in question_lower or "path" in question_lower:
            words = question_lower.split()
            if len(words) > 2:
                technique = words[-1]
                return {
                    "query_type": "attack_chain",
                    "results": self.find_attack_chains(technique)
                }

        elif "similar" in question_lower:
            words = question_lower.split()
            if len(words) > 2:
                node_id = words[-1]
                return {
                    "query_type": "similarity",
                    "results": self.find_similar_nodes(node_id)
                }

        else:
            return {
                "query_type": "unknown",
                "results": [],
                "message": "Could not understand query. Try asking about mitigations, attack chains, or similar nodes."
            }

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

class KnowledgeGraphBuilder:
    """
    Builder class to populate knowledge graph from various sources
    """

    def __init__(self, graph: Neo4jKnowledgeGraph):
        self.graph = graph

    def add_mitre_attack_data(self, mitre_data: Dict[str, Any]) -> int:
        """Add MITRE ATT&CK data to knowledge graph"""
        count = 0

        try:
            # Add techniques
            for technique in mitre_data.get('techniques', []):
                node = GraphNode(
                    id=technique['id'],
                    type='technique',
                    properties={
                        'name': technique['name'],
                        'description': technique.get('description', ''),
                        'tactic': technique.get('tactic', ''),
                        'platform': technique.get('platform', [])
                    }
                )
                if self.graph.add_node(node):
                    count += 1

            # Add relationships
            for technique in mitre_data.get('techniques', []):
                # Add subtechnique relationships
                for subtechnique_id in technique.get('subtechniques', []):
                    edge = GraphEdge(
                        source=technique['id'],
                        target=subtechnique_id,
                        relationship='has_subtechnique',
                        weight=1.0
                    )
                    self.graph.add_edge(edge)

            logger.info(f"✅ Added {count} MITRE ATT&CK nodes to knowledge graph")
        except Exception as e:
            logger.error(f"Failed to add MITRE data: {e}")

        return count

    def add_battle_results(self, battle_data: Dict[str, Any]) -> int:
        """Add battle results to knowledge graph with temporal tracking"""
        count = 0

        try:
            # Add attack nodes from battle
            for attack in battle_data.get('attacks', []):
                node = GraphNode(
                    id=f"attack_{attack['id']}",
                    type='attack',
                    properties={
                        'name': attack['name'],
                        'success': attack.get('success', False),
                        'technique': attack.get('technique', ''),
                        'battle_id': battle_data['id']
                    }
                )
                if self.graph.add_node(node):
                    count += 1

                # Add temporal relationship to technique
                if attack.get('technique'):
                    edge = GraphEdge(
                        source=f"attack_{attack['id']}",
                        target=attack['technique'],
                        relationship='uses_technique',
                        temporal=True,
                        timestamp=datetime.fromisoformat(battle_data['timestamp'])
                    )
                    self.graph.add_edge(edge)

            # Add defense responses
            for defense in battle_data.get('defenses', []):
                node = GraphNode(
                    id=f"defense_{defense['id']}",
                    type='defense',
                    properties={
                        'name': defense['name'],
                        'effectiveness': defense.get('effectiveness', 0.0),
                        'battle_id': battle_data['id']
                    }
                )
                if self.graph.add_node(node):
                    count += 1

                # Link defense to attack it countered
                if defense.get('countered_attack'):
                    edge = GraphEdge(
                        source=f"defense_{defense['id']}",
                        target=f"attack_{defense['countered_attack']}",
                        relationship='mitigates',
                        weight=defense.get('effectiveness', 0.5),
                        temporal=True
                    )
                    self.graph.add_edge(edge)

            logger.info(f"✅ Added {count} battle result nodes to knowledge graph")
        except Exception as e:
            logger.error(f"Failed to add battle results: {e}")

        return count

    def add_vulnerability_data(self, cve_data: List[Dict[str, Any]]) -> int:
        """Add CVE vulnerability data to knowledge graph"""
        count = 0

        try:
            for cve in cve_data:
                node = GraphNode(
                    id=cve['id'],
                    type='vulnerability',
                    properties={
                        'cve_id': cve['id'],
                        'description': cve.get('description', ''),
                        'cvss_score': cve.get('cvss_score', 0.0),
                        'published_date': cve.get('published_date', ''),
                        'affected_products': cve.get('affected_products', [])
                    }
                )
                if self.graph.add_node(node):
                    count += 1

            logger.info(f"✅ Added {count} vulnerability nodes to knowledge graph")
        except Exception as e:
            logger.error(f"Failed to add vulnerability data: {e}")

        return count
