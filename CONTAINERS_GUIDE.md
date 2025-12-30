# YUGMĀSTRA Docker Containers Guide

## 📦 Container Breakdown

### **ESSENTIAL (5 containers)** - Minimum for app to work
These are the bare minimum needed for YUGMĀSTRA to function:

1. **postgres** - Database for storing all application data
   - User accounts, settings, attack/defense data
   - **CANNOT REMOVE** - Core data storage

2. **redis** - Fast cache and session storage
   - Session management, caching
   - **CANNOT REMOVE** - Performance critical

3. **ai-engine** - Your custom AI/ML service
   - Red Team, Blue Team, Evolution AI agents
   - LLM models, RAG, vector search
   - **ESSENTIAL FOR AI FEATURES**

4. **web** - Next.js frontend application
   - Main user interface
   - **CANNOT REMOVE** - This is your app!

5. **nginx** - Reverse proxy/load balancer
   - Routes traffic to web and AI engine
   - **CANNOT REMOVE** - Entry point

**Total: 5 containers** ✅

---

### **OPTIONAL BUT USEFUL (6 containers)** - Add these for specific features

6. **neo4j** - Graph database
   - Knowledge graph visualization
   - Relationship mapping between threats/attacks
   - **REMOVE IF**: You don't need graph visualization
   - **KEEP IF**: You want knowledge graph features

7. **elasticsearch** - Search and analytics
   - Log aggregation and search
   - Full-text search capabilities
   - **REMOVE IF**: You don't need advanced logging
   - **KEEP IF**: You need SIEM-like features

8. **kibana** - Elasticsearch UI
   - Log visualization dashboard
   - **REMOVE IF**: No elasticsearch or don't need log UI
   - **KEEP IF**: You want to view logs graphically

9. **grafana** - Metrics and monitoring
   - System performance dashboards
   - **REMOVE IF**: You don't need monitoring dashboards
   - **KEEP IF**: You want pretty metrics graphs

10. **prometheus** - Metrics collection
    - Collects system metrics for Grafana
    - **REMOVE IF**: No grafana
    - **KEEP IF**: You need monitoring

11. **minio** - Object storage (S3-compatible)
    - File storage for uploads, exports
    - **REMOVE IF**: You can store files locally
    - **KEEP IF**: You need distributed file storage

---

### **PROBABLY DON'T NEED (6 containers)** - Remove these for simplicity

12. **api** (FastAPI backend)
    - **REDUNDANT** - AI Engine already provides API
    - **REMOVE** - Not needed if using AI Engine

13. **kafka** + **zookeeper** (2 containers)
    - Message streaming platform
    - **OVERKILL** for development
    - **REMOVE** - Only needed for massive scale

14. **red-team-ai**, **blue-team-ai**, **evolution-engine** (3 containers)
    - **REDUNDANT** - AI Engine already has all 3 agents
    - **REMOVE** - Consolidated into ai-engine

15. **cyber-range**
    - Separate training environment
    - **REMOVE** - Not essential for core functionality

---

## 🚀 Recommended Setups

### **Option 1: Minimal (5 containers)** - FASTEST SETUP ⚡
Just the essentials - perfect for development and testing the AI features.

```bash
docker-compose -f docker-compose.minimal.yml up -d
```

**Containers**: postgres, redis, ai-engine, web, nginx

**Use when**:
- Just want to test the AI assistant
- Development on your local machine
- Learning how the system works
- Don't need monitoring/logging features

---

### **Option 2: Standard (8 containers)** - RECOMMENDED FOR FULL FEATURES 🎯
Essential + useful features without bloat.

**Add**: neo4j, elasticsearch, kibana

```bash
# Start minimal setup first
docker-compose -f docker-compose.minimal.yml up -d

# Then add optional services
docker-compose up -d neo4j elasticsearch kibana
```

**Use when**:
- Want knowledge graph visualization
- Need logging and search capabilities
- Building the full application

---

### **Option 3: Full (17 containers)** - COMPLETE SYSTEM 🏢
Everything enabled - only for production or comprehensive testing.

```bash
docker-compose up -d
```

**Use when**:
- Production deployment
- Need all monitoring and analytics
- Demonstrating full capabilities
- High-traffic scenarios

---

## 💾 Disk Space Comparison

| Setup | Containers | Approx Size | RAM Usage |
|-------|-----------|-------------|-----------|
| Minimal | 5 | ~2GB | ~2GB |
| Standard | 8 | ~4GB | ~4GB |
| Full | 17 | ~10GB | ~8GB |

---

## 🔧 Commands

### Start Minimal Setup
```bash
docker-compose -f docker-compose.minimal.yml up -d
```

### Stop Everything
```bash
docker-compose down
```

### Remove Unused Containers
```bash
# Stop and remove specific containers
docker-compose rm -s -f kafka zookeeper red-team-ai blue-team-ai evolution-engine cyber-range api

# Clean up unused volumes
docker volume prune
```

### Check What's Running
```bash
docker-compose ps
```

### View Logs
```bash
# All containers
docker-compose logs -f

# Specific container
docker-compose logs -f ai-engine
```

---

## 📊 What Each Container Does

### Core Application Flow
```
User → nginx → web (Next.js) → ai-engine (Python) → postgres/redis
                                                    ↓
                                            Custom LLM Models
```

### With Knowledge Graph
```
User → nginx → web → ai-engine → neo4j (graph visualization)
```

### With Logging
```
User → nginx → web → ai-engine → elasticsearch → kibana (log viewer)
```

---

## ⚡ Quick Start (Minimal)

1. **Start minimal setup**:
   ```bash
   docker-compose -f docker-compose.minimal.yml up -d
   ```

2. **Wait for containers to be healthy**:
   ```bash
   docker-compose -f docker-compose.minimal.yml ps
   ```

3. **Access application**:
   - Web: http://localhost:200
   - AI Engine API: http://localhost:8001
   - Login: preetraval45@gmail.com / yugmastra2025

4. **Test AI Assistant**:
   - Click "AI Assistant" in sidebar
   - Try all 3 modes: Red Team, Blue Team, Evolution

---

## 🎯 My Recommendation

**For Development**: Use **docker-compose.minimal.yml** (5 containers)
- Fast startup
- Low resource usage
- All core features work
- AI Assistant fully functional

**For Demo/Production**: Use **Standard** (8 containers)
- Add neo4j for knowledge graph
- Add elasticsearch + kibana for logging
- Still manageable resource usage

**Avoid**: Running all 17 containers unless you specifically need every feature!

---

## 🗑️ Containers You Can Safely Remove

From the full docker-compose.yml, these are **not needed**:

1. ❌ **kafka** - Message queue (overkill)
2. ❌ **zookeeper** - Kafka dependency (not needed)
3. ❌ **api** - Redundant (ai-engine has API)
4. ❌ **red-team-ai** - Consolidated into ai-engine
5. ❌ **blue-team-ai** - Consolidated into ai-engine
6. ❌ **evolution-engine** - Consolidated into ai-engine
7. ❌ **cyber-range** - Optional training environment
8. ⚠️ **grafana** - Nice to have, but optional
9. ⚠️ **prometheus** - Only needed with Grafana
10. ⚠️ **minio** - Only if you need object storage

That's **10 containers** you can remove safely! 🎉

---

## 🚀 Next Steps

1. Stop current containers:
   ```bash
   docker-compose down
   ```

2. Start minimal setup:
   ```bash
   docker-compose -f docker-compose.minimal.yml up -d
   ```

3. Test the application at http://localhost:200

4. If you need additional features later, add them one by one:
   ```bash
   docker-compose up -d neo4j  # Add graph database
   docker-compose up -d elasticsearch kibana  # Add logging
   ```

**Simple, fast, and efficient!** 🚀
