# 🐳 YUGMĀSTRA Docker Containers

## System Owner: Preet Raval (preetraval45@gmail.com)

---

## Container Overview

| Container Name | Port | Image | Purpose |
|----------------|------|-------|---------|
| **yugmastra-nginx** | **200** | nginx:alpine | **Main entry point - Access here!** |
| yugmastra-web | 3000 (internal) | yugmastra-web | Next.js web application |
| yugmastra-api | 8000 (internal) | yugmastra-api | API Gateway |
| yugmastra-red-team-ai | 8000 (internal) | yugmastra-red-team-ai | Attack AI service |
| yugmastra-blue-team-ai | 8000 (internal) | yugmastra-blue-team-ai | Defense AI service |
| yugmastra-evolution-engine | 8000 (internal) | yugmastra-evolution-engine | Co-evolution orchestrator |
| yugmastra-cyber-range | 8000 (internal) | yugmastra-cyber-range | Training environment |
| yugmastra-postgres | 5432 (internal) | postgres:15-alpine | Database |
| yugmastra-redis | 6379 (internal) | redis:7-alpine | Cache & Queue |
| yugmastra-neo4j | 7687 (internal) | neo4j:5-community | Knowledge graph DB |
| yugmastra-elasticsearch | 9200 (internal) | elasticsearch:8.12.0 | Search engine |
| yugmastra-kibana | 5601 (internal) | kibana:8.12.0 | Log viewer |
| yugmastra-kafka | 29092 (internal) | cp-kafka:7.6.0 | Event streaming |
| yugmastra-zookeeper | 2181 (internal) | cp-zookeeper:7.6.0 | Kafka coordinator |
| yugmastra-prometheus | 9090 (internal) | prometheus:latest | Metrics collector |
| yugmastra-grafana | 3000 (internal) | grafana:latest | Monitoring dashboard |
| yugmastra-minio | 9000 (internal) | minio:latest | Object storage |

---

## 🌐 Access URL

**Main System:** http://localhost:200

All services accessible through nginx reverse proxy on port 200.

---

## 📊 Check Container Stats

```bash
docker stats
```

This shows real-time:
- CPU usage %
- Memory usage
- Network I/O
- Container names

---

## 🔍 Quick Commands

### List all containers
```bash
docker-compose ps
```

### Check resource usage
```bash
docker stats --no-stream
```

### View specific container
```bash
docker stats yugmastra-web
```

---

## 💾 Image Sizes

- **Web App:** ~1.19 GB
- **API Gateway:** ~714 MB
- **AI Services:** ~3.3-10 GB each (includes PyTorch)
- **Databases:** Varies by data
- **Infrastructure:** Minimal (~100-500 MB)

**Total System:** ~30-40 GB

---

## ⚡ Single Port Architecture

Only **port 200** is exposed externally.
All other services communicate internally through Docker network: `yugmastra-network`

This is secure and clean! 🔒
