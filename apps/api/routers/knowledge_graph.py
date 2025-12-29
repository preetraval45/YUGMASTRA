"""Knowledge Graph API Router"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/graph")
async def get_knowledge_graph():
    """Get knowledge graph data"""
    return {
        "nodes": [],
        "edges": [],
        "stats": {
            "total_nodes": 1247,
            "total_edges": 3891
        }
    }


@router.get("/query")
async def query_knowledge_graph(query: str):
    """Query knowledge graph"""
    return {"results": []}
