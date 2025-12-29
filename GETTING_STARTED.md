# Getting Started with YUGMĀSTRA

## Welcome! 👋

You now have a fully scaffolded **enterprise-grade, cross-platform AI cybersecurity research platform**. Here's everything you need to know to get started.

---

## What You Have

### ✅ Complete Project Structure
- **Monorepo** with Turborepo for efficient builds
- **7 microservices** (Red Team AI, Blue Team AI, Evolution Engine, Cyber Range, Knowledge Graph, API Gateway, and more)
- **3 frontend apps** (Web, Mobile, Desktop)
- **Custom AI/ML models** built from scratch
- **Full infrastructure** (Docker, K8s, monitoring)

### ✅ Custom AI/ML Stack
- **Custom Transformer** - Built from scratch for attack planning
- **Custom NLP Engine** - LSTM-based for report generation
- **Custom RL Agents** - PPO algorithm implementation
- **Autoencoder** - For anomaly detection
- **Bidirectional LSTM** - For temporal analysis

### ✅ Cross-Platform Applications
- **Web** - Next.js 14 with TypeScript
- **Mobile** - React Native (iOS/Android)
- **Desktop** - Electron (Windows/macOS/Linux)

### ✅ Production Infrastructure
- **Docker Compose** - Local development
- **Kubernetes** - Production deployment
- **Monitoring** - Prometheus + Grafana
- **Logging** - ELK Stack
- **Databases** - PostgreSQL, Neo4j, Redis

---

## Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
# Node.js dependencies
npm install

# Python dependencies
pip install -r requirements.txt
```

### Step 2: Start Infrastructure

```bash
# Start databases and supporting services
docker-compose up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose ps
```

You should see:
- ✅ PostgreSQL (port 5432)
- ✅ Redis (port 6379)
- ✅ Neo4j (port 7474, 7687)
- ✅ Elasticsearch (port 9200)
- ✅ Kafka (port 9092)
- ✅ Prometheus (port 9090)
- ✅ Grafana (port 3001)

### Step 3: Start Applications

**Terminal 1 - Web App:**
```bash
cd apps/web
npm run dev
# Visit: http://localhost:3000
```

**Terminal 2 - API Gateway:**
```bash
cd apps/api
pip install fastapi uvicorn pydantic
uvicorn main:app --reload
# Visit: http://localhost:8000/docs
```

**Terminal 3 - Mobile App (Optional):**
```bash
cd apps/mobile
npm start
# Scan QR code with Expo Go app
```

---

## Project Structure Overview

```
yugmastra/
│
├── apps/                          # 🖥️ Applications
│   ├── web/                      # Next.js web app
│   ├── mobile/                   # React Native mobile
│   ├── desktop/                  # Electron desktop
│   └── api/                      # FastAPI gateway
│
├── services/                      # 🚀 Backend Services
│   ├── red-team-ai/              # 🔴 Attack agent
│   ├── blue-team-ai/             # 🔵 Defense agent
│   ├── evolution-engine/         # 🧬 Co-evolution core
│   ├── cyber-range/              # 🎯 Simulation environment
│   └── knowledge-graph/          # 🕸️ Graph database service
│
├── ml/                           # 🤖 Custom AI Models
│   ├── models/
│   │   ├── custom_transformer.py # Transformer from scratch
│   │   └── custom_nlp.py         # NLP engine
│   ├── training/                 # Training scripts
│   └── datasets/                 # Data generation
│
├── packages/                      # 📦 Shared Packages
│   ├── ui/                       # React components
│   ├── core/                     # Business logic
│   ├── types/                    # TypeScript types
│   └── ai-sdk/                   # AI client SDK
│
└── infrastructure/               # 🏗️ DevOps
    ├── docker/                   # Docker configs
    ├── kubernetes/               # K8s manifests
    └── terraform/                # Infrastructure as Code
```

---

## Next Steps

### Option 1: Explore the Platform

1. **Open the Web App** → http://localhost:3000
   - Beautiful landing page
   - Dashboard overview
   - Evolution metrics

2. **Check API Docs** → http://localhost:8000/docs
   - Interactive API documentation
   - Test endpoints
   - WebSocket connections

3. **View Monitoring** → http://localhost:3001
   - Grafana dashboards
   - System metrics
   - Custom AI metrics

### Option 2: Start Training AI Agents

```bash
# Train Red Team AI
cd services/red-team-ai
python -m pip install -e .
python training/train_red_agent.py

# Train Blue Team AI
cd services/blue-team-ai
python -m pip install -e .
python training/train_blue_agent.py

# Run Co-Evolution
cd services/evolution-engine
python -m pip install -e .
python -m src.core.evolution_loop --episodes 1000
```

### Option 3: Develop Custom Features

**Add a new AI model:**
```bash
# Create new model in ml/models/
touch ml/models/my_custom_model.py
```

**Add a new API endpoint:**
```bash
# Edit apps/api/routers/
# Add new router and endpoints
```

**Customize the UI:**
```bash
cd apps/web
# Edit pages in app/
# Add components in components/
```

---

## Key Technologies

### Backend
- **Python 3.11+** - Core language
- **PyTorch 2.0** - Deep learning
- **FastAPI** - Modern Python web framework
- **Gymnasium** - RL environments

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **TailwindCSS** - Styling
- **React Native** - Mobile development
- **Electron** - Desktop apps

### Infrastructure
- **Docker** - Containerization
- **Kubernetes** - Orchestration
- **PostgreSQL** - Relational DB
- **Neo4j** - Graph database
- **Redis** - Caching
- **Kafka** - Event streaming

---

## Development Tips

### Hot Reloading
- Web app: Automatic with Next.js
- API: Automatic with `--reload` flag
- Mobile: Shake device → Reload

### Debugging
```bash
# Backend logs
docker-compose logs -f

# Specific service
docker-compose logs -f postgres

# Python debugging
import pdb; pdb.set_trace()

# Node debugging
console.log()
```

### Common Issues

**Port already in use:**
```bash
# Find process
lsof -i :3000  # or :8000, :5432, etc.

# Kill process
kill -9 <PID>
```

**Docker issues:**
```bash
# Reset Docker
docker-compose down -v
docker system prune -af
docker-compose up -d
```

**Dependency issues:**
```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

# Python
pip install --upgrade --force-reinstall -r requirements.txt
```

---

## Testing

```bash
# Backend tests
pytest services/*/tests/

# Frontend tests
npm test

# E2E tests
npm run test:e2e
```

---

## Building for Production

### Web App
```bash
cd apps/web
npm run build
npm start
```

### Mobile App
```bash
cd apps/mobile
# iOS
npm run build:ios
# Android
npm run build:android
```

### Desktop App
```bash
cd apps/desktop
npm run make
# Check out/make/ for installers
```

### Docker Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Documentation

- **Architecture**: [README.md](README.md)
- **Implementation Guide**: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) (to be created)

---

## Research Goals

This platform enables research on:

1. **Zero-Day Discovery** - Can AI discover novel attacks?
2. **Autonomous Defense** - Can defenses emerge without human labeling?
3. **Co-Evolution** - How do red-blue agents reach equilibrium?
4. **Explainable Security** - Can AI explain its decisions?
5. **Transfer Learning** - Do strategies generalize?

### Potential Publications

- USENIX Security
- IEEE S&P
- ACM CCS
- NDSS

---

## Getting Help

- **Documentation**: Check [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: contact@yugmastra.ai

---

## What's Next?

### Immediate Tasks
1. ✅ Review the codebase structure
2. ⏳ Set up your development environment
3. ⏳ Start the applications
4. ⏳ Explore the platform
5. ⏳ Begin customization

### Week 1 Goals
- Get all services running
- Understand the architecture
- Train a basic RL agent
- Customize the UI

### Month 1 Goals
- Implement cyber range fully
- Complete co-evolution training
- Add knowledge graph integration
- Create first research results

---

## Community & Support

- **Star the repo** ⭐
- **Fork and contribute** 🍴
- **Report bugs** 🐛
- **Suggest features** 💡
- **Share your research** 📄

---

**Welcome to the future of autonomous cybersecurity research!** 🚀🛡️

Built with ❤️ using cutting-edge AI and modern software engineering practices.
