# 🚀 YUGMĀSTRA - Quick Start Guide

## ✅ What You Have Now

**ONLY 5 ESSENTIAL CONTAINERS** (Optimized from 17!)

1. **postgres** - Database
2. **redis** - Cache
3. **ai-engine** - Your custom trained AI (Red/Blue/Evolution agents)
4. **web** - Next.js frontend + API
5. **nginx** - Reverse proxy

**Total RAM**: ~2GB (was 8GB!)
**Disk Space**: ~3GB (was 10GB!)

---

## 🎯 Quick Start (3 commands!)

```bash
# 1. Start everything
docker-compose up -d

# 2. Wait for containers (30 seconds)
docker-compose ps

# 3. Open browser
http://localhost:200
```

**Login**: preetraval45@gmail.com / yugmastra2025

---

## 🤖 Use Your Custom AI

1. **Login** at http://localhost:200
2. Click **"AI Assistant"** in sidebar (✨ sparkle icon)
3. **Select AI Mode**:
   - 🗡️ **Red Team** - Attack vectors, exploitation, pentesting
   - 🛡️ **Blue Team** - Defense, detection, incident response
   - ⚡ **Evolution** - Adaptive intelligence, emerging threats
4. **Ask anything**!

### Example Questions:

**Red Team**:
- "What are the latest SQL injection techniques?"
- "How do I perform privilege escalation on Windows?"
- "Explain the attack chain for ransomware"

**Blue Team**:
- "How do I detect PowerShell attacks?"
- "What SIEM rules should I implement?"
- "Best practices for incident response"

**Evolution**:
- "What are emerging AI-powered threats?"
- "Analyze the current threat landscape"
- "Predict future attack trends"

---

## 🎓 Train Your Own AI Models

### 1. Create Dataset

Create `training_data.json`:
```json
[
  {
    "text": "SQL injection allows attackers to manipulate database queries...",
    "label": "vulnerability"
  },
  {
    "text": "Ransomware encrypts files and demands payment...",
    "label": "malware"
  }
]
```

### 2. Copy to AI Engine
```bash
docker cp training_data.json yugmastra-ai-engine:/app/data/
```

### 3. Train Model
```bash
curl -X POST http://localhost:8001/api/ai/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/app/data/training_data.json",
    "model_type": "distilbert",
    "epochs": 5,
    "batch_size": 16
  }'
```

### 4. Check Training Progress
```bash
docker logs -f yugmastra-ai-engine
```

---

## 📊 Available APIs

### AI Chat
```bash
curl -X POST http://localhost:8001/api/ai/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is SQL injection?",
    "mode": "red-team",
    "history": []
  }'
```

### Get AI Models Info
```bash
curl http://localhost:8001/api/ai/models
```

### Create Embeddings
```bash
curl -X POST http://localhost:8001/api/ai/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Threat description here..."],
    "store": true
  }'
```

### RAG Query
```bash
curl "http://localhost:8001/api/ai/rag/query?query=ransomware&top_k=5"
```

---

## 🛠️ Useful Commands

### View All Containers
```bash
docker-compose ps
```

### View Logs
```bash
# All containers
docker-compose logs -f

# Specific container
docker logs -f yugmastra-ai-engine
docker logs -f yugmastra-web
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific
docker-compose restart ai-engine
docker-compose restart web
```

### Stop Everything
```bash
docker-compose down
```

### Clean Up (Remove data)
```bash
docker-compose down -v  # WARNING: Deletes all data!
```

### Rebuild After Code Changes
```bash
# Rebuild AI Engine
docker-compose up -d --build ai-engine

# Rebuild Web
docker-compose up -d --build web
```

---

## 🔧 Troubleshooting

### AI Engine Not Responding
```bash
# Check if it's running
docker logs yugmastra-ai-engine

# Restart it
docker-compose restart ai-engine

# Wait 2 minutes for models to load
```

### Web App Not Loading
```bash
# Check nginx
docker logs yugmastra-nginx

# Restart web and nginx
docker-compose restart web nginx
```

### Database Connection Error
```bash
# Check postgres
docker-compose ps postgres

# Should show "healthy"
# If not, restart
docker-compose restart postgres
```

### Port Already in Use
```bash
# Change port in docker-compose.yml
# Line: "200:80" -> "3000:80" (or any free port)
```

---

## 📚 Documentation

- **AI Engine Guide**: `services/ai-engine/README.md`
- **Container Guide**: `CONTAINERS_GUIDE.md`
- **Full Documentation**: Check README files in each service

---

## 🎯 What's Different from Before

### REMOVED (Merged into AI Engine):
- ❌ red-team-ai container
- ❌ blue-team-ai container
- ❌ evolution-engine container
- ❌ api container (FastAPI)
- ❌ kafka + zookeeper
- ❌ elasticsearch + kibana
- ❌ grafana + prometheus
- ❌ minio
- ❌ neo4j
- ❌ cyber-range

### NOW (All-in-One):
- ✅ **ai-engine** - Has all 3 agents (Red/Blue/Evolution) + LLM + RAG + Vector Store
- ✅ **web** - Has Next.js frontend + API routes
- ✅ Clean, simple, fast!

---

## ⚡ Performance

| Metric | Before | Now |
|--------|--------|-----|
| Containers | 17 | 5 |
| RAM Usage | ~8GB | ~2GB |
| Disk Space | ~10GB | ~3GB |
| Startup Time | ~5 min | ~1 min |
| Build Time | ~10 min | ~3 min |

---

## 🎉 You're Ready!

Your custom AI cybersecurity platform is running with:
- ✅ Custom trained LLM models
- ✅ RAG system with vector search
- ✅ Multi-agent AI (Red/Blue/Evolution)
- ✅ Beautiful web interface
- ✅ Only essential containers
- ✅ Fast and efficient

**Access now**: http://localhost:200

**Questions?** Check the documentation in `services/ai-engine/README.md`

---

**Built with**: PyTorch, Transformers, LangChain, ChromaDB, FAISS, FastAPI, Next.js 🚀
