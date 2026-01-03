# AI Engine Status & Configuration

## Current Status

### ✅ Core Features (Working)
- **FastAPI Server**: Running on port 8001
- **LLM Manager**: Ollama integration ready
- **Red Team Agent**: Attack pattern generation
- **Blue Team Agent**: Defense strategy generation
- **Evolution Agent**: Co-evolutionary AI training
- **RAG Service**: Retrieval-augmented generation
- **Vector Store**: Embedding storage and search

### ⚠️ Advanced Features (Disabled - Require Dependencies)
- **Knowledge Graph**: Requires Neo4j server
- **Zero-Day Discovery**: Requires full ML stack
- **SIEM Rule Generation**: Requires knowledge graph

## Why Advanced Features Are Disabled

The AI Engine has been configured to run with core features only because:

1. **Knowledge Graph** requires Neo4j database server (not started)
2. **Zero-Day Discovery** requires large ML models (resource intensive)
3. **SIEM Rule Generator** depends on knowledge graph

This allows the AI Engine to:
- ✅ Start successfully without errors
- ✅ Provide core AI/ML capabilities
- ✅ Support red team / blue team simulations
- ✅ Run in development without heavy dependencies

## Enabling Advanced Features

### Option 1: Enable Knowledge Graph (Recommended)

**Step 1**: Add Neo4j to docker-compose.yml
```yaml
services:
  neo4j:
    image: neo4j:5.16-community
    container_name: yugmastra-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/yugmastra
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache__size: 1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - yugmastra-network

volumes:
  neo4j_data:
  neo4j_logs:
```

**Step 2**: Update .env
```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yugmastra
ENABLE_KNOWLEDGE_GRAPH=true
```

**Step 3**: Uncomment imports in main.py
```python
from models.knowledge_graph import Neo4jKnowledgeGraph, GraphNode, GraphEdge, KnowledgeGraphBuilder
```

**Step 4**: Rebuild and restart
```bash
docker-compose up -d neo4j
docker-compose build ai-engine
docker-compose restart ai-engine
```

### Option 2: Enable All Advanced Features

**Requirements**:
- 8GB+ RAM available
- Neo4j server running
- GPU (optional, for faster ML)

**Steps**:
1. Follow Option 1 for Neo4j
2. Update .env:
   ```env
   ENABLE_KNOWLEDGE_GRAPH=true
   ENABLE_ZERO_DAY_DISCOVERY=true
   ENABLE_SIEM_GENERATION=true
   ```
3. Uncomment all imports in main.py
4. Rebuild AI Engine:
   ```bash
   docker-compose build ai-engine
   docker-compose restart ai-engine
   ```

**Expected Resource Usage**:
- CPU: 2-4 cores
- RAM: 4-8 GB
- Disk: 10+ GB for models
- Startup Time: 2-5 minutes

## Feature Flags

You can control which features are enabled via environment variables:

```env
# .env file
ENABLE_KNOWLEDGE_GRAPH=false      # Neo4j-based attack graph
ENABLE_ZERO_DAY_DISCOVERY=false   # ML-based vulnerability detection
ENABLE_SIEM_GENERATION=false      # Automated SIEM rule creation
ENABLE_THREAT_INTEL=true          # Threat intelligence aggregation
```

## Current Configuration

### Core Services (Always Enabled)
| Service | Status | Port | Dependencies |
|---------|--------|------|--------------|
| FastAPI | ✅ Running | 8001 | Python, FastAPI |
| Ollama LLM | ✅ Ready | 11434 | Ollama server |
| RAG Service | ✅ Ready | - | ChromaDB |
| Vector Store | ✅ Ready | - | FAISS |

### Advanced Services (Optional)
| Service | Status | Dependencies | Resource Impact |
|---------|--------|--------------|-----------------|
| Knowledge Graph | ⚠️ Disabled | Neo4j server | Medium (2GB RAM) |
| Zero-Day Discovery | ⚠️ Disabled | PyTorch, Large models | High (4GB RAM) |
| SIEM Generator | ⚠️ Disabled | Knowledge Graph | Low (500MB RAM) |

## Performance Comparison

### Current (Core Features Only)
- **Startup Time**: 10-30 seconds
- **Memory Usage**: 1-2 GB
- **CPU Usage**: Low (10-20%)
- **Container Health**: ✅ Healthy

### With All Features Enabled
- **Startup Time**: 2-5 minutes
- **Memory Usage**: 6-10 GB
- **CPU Usage**: Medium (30-60%)
- **Container Health**: ✅ Healthy (slower)

## Troubleshooting

### AI Engine Shows "Unhealthy"
**Cause**: Feature imports failing or heavy ML model loading
**Solutions**:
1. Check logs: `docker logs yugmastra-ai-engine --tail 100`
2. Verify all dependencies installed: `docker exec yugmastra-ai-engine pip list`
3. Disable heavy features in .env
4. Increase health check timeout in docker-compose.yml

### "ModuleNotFoundError: No module named 'models.knowledge_graph'"
**Cause**: Trying to import advanced modules without dependencies
**Solution**: Keep imports commented out in main.py (current state)

### Slow Startup
**Cause**: Loading large ML models at startup
**Solution**:
1. Use lazy loading for ML models
2. Disable unused features
3. Increase Docker container memory limit

## API Endpoints Status

### ✅ Working (Core Features)
```
GET  /                          # Service info
GET  /health                   # Health check
POST /api/ai/chat             # AI chat (basic LLM)
POST /api/ai/train            # Model training
POST /api/ai/embed            # Text embeddings
POST /api/ai/rag/query        # RAG queries
POST /api/ai/ingest           # Document ingestion
GET  /api/ai/models           # Available models
```

### ⚠️ Disabled (Requires Advanced Features)
```
GET  /api/knowledge-graph                          # Requires Neo4j
POST /api/knowledge-graph/query                    # Requires Neo4j
POST /api/knowledge-graph/nodes                    # Requires Neo4j
POST /api/knowledge-graph/edges                    # Requires Neo4j
GET  /api/knowledge-graph/attack-chains/{id}       # Requires Neo4j
POST /api/zero-day/train                           # Requires ML models
POST /api/zero-day/predict                         # Requires ML models
POST /api/siem/generate                            # Requires Knowledge Graph
```

## Recommendations

### For Development (Current Setup)
**Keep core features only**:
- ✅ Fast startup
- ✅ Low resource usage
- ✅ Sufficient for testing
- ✅ Reliable operation

### For Production (Enable Advanced Features)
**Enable all features**:
- Provides full AI capabilities
- Better attack/defense intelligence
- Automated SIEM rule generation
- Advanced threat correlation

**Requirements**:
- Add Neo4j service
- Allocate 8GB+ RAM
- Consider GPU for ML acceleration
- Set up proper monitoring

## Migration Path

### Phase 1: Current (Core Only) ✅
- Basic AI chat working
- RAG service functional
- Red/Blue team agents ready
- No external dependencies

### Phase 2: Add Knowledge Graph
- Install Neo4j
- Enable graph features
- Populate initial graph data
- Test attack chain queries

### Phase 3: Full Feature Set
- Enable ML models
- Add zero-day discovery
- Enable SIEM generation
- Full production deployment

## Next Steps

1. **For Development**: Continue with current core setup
2. **To Enable Knowledge Graph**: Follow Option 1 above
3. **For Production**: Plan Phase 2-3 migration
4. **Optimization**: Consider selective feature enabling based on usage

---

**Last Updated**: 2026-01-02
**Current Mode**: Core Features Only
**Recommended Mode**: Core (Development), Full (Production)
**Resource Impact**: Low (Current), Medium-High (Full Features)
