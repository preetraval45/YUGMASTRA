# YUGMĀSTRA Implementation Guide

## Project Overview

YUGMĀSTRA is a cutting-edge **Autonomous Adversary-Defender Co-Evolution Platform** that uses AI to discover novel cybersecurity attack and defense strategies through adversarial self-play.

### Core Innovation

Instead of relying on pre-programmed rules or labeled datasets, YUGMĀSTRA lets AI agents **learn and evolve** attack and defense strategies by competing against each other—similar to how AlphaGo mastered Go through self-play.

---

## Project Structure

```
yugmastra/
├── apps/                          # Frontend applications
│   ├── web/                      # Next.js web app
│   ├── mobile/                   # React Native (iOS/Android)
│   ├── desktop/                  # Electron (Windows/macOS/Linux)
│   └── api/                      # FastAPI gateway
│
├── services/                      # Backend microservices
│   ├── red-team-ai/              # Attack agent (custom RL)
│   ├── blue-team-ai/             # Defense agent (custom ML/DL)
│   ├── evolution-engine/         # Co-evolution core (MARL)
│   ├── knowledge-graph/          # Neo4j graph service
│   └── cyber-range/              # Simulation environment
│
├── ml/                           # Custom AI/ML models
│   ├── models/
│   │   ├── custom_transformer.py # Built-from-scratch Transformer
│   │   └── custom_nlp.py         # Custom NLP engine
│   ├── training/                 # Training scripts
│   └── datasets/                 # Data generators
│
├── packages/                      # Shared libraries
│   ├── ui/                       # React components
│   ├── core/                     # Business logic
│   ├── ai-sdk/                   # AI client SDK
│   └── types/                    # TypeScript types
│
└── infrastructure/               # DevOps
    ├── docker/                   # Docker configs
    ├── kubernetes/               # K8s manifests
    └── terraform/                # IaC
```

---

## Technology Stack

### Backend & AI (Custom-Built)
- **Python 3.11+** - Core language
- **PyTorch 2.0+** - Deep learning framework
- **Custom Transformer** - Built from scratch for attack planning
- **Custom NLP Engine** - LSTM-based report generation
- **Custom RL** - PPO, SAC algorithms from scratch
- **FastAPI** - API framework
- **Ray/RLlib** - Distributed training

### Databases
- **PostgreSQL 15** - Relational data
- **Neo4j 5** - Knowledge graph
- **Redis 7** - Caching
- **TimescaleDB** - Time-series metrics

### Frontend
- **Next.js 14** - Web (React 18, TypeScript)
- **React Native** - Mobile (iOS/Android via Expo)
- **Electron** - Desktop (Windows/macOS/Linux)
- **TailwindCSS** - Styling
- **Recharts + D3.js** - Visualizations
- **Socket.IO** - Real-time updates

### Infrastructure
- **Docker** - Containerization
- **Kubernetes** - Orchestration
- **Kafka** - Event streaming
- **Prometheus + Grafana** - Monitoring
- **ELK Stack** - Logging

---

## Quick Start

### Prerequisites
```bash
# Required
Node.js 20+
Python 3.11+
Docker 24+
Docker Compose 2.20+

# Optional (for development)
Kubernetes (minikube/kind)
Terraform
```

### Installation

1. **Clone and setup**
```bash
git clone <your-repo-url>
cd yugmastra
npm install
pip install -r requirements.txt
```

2. **Start infrastructure**
```bash
# Start databases and services
docker-compose up -d

# Verify services
docker-compose ps
```

3. **Initialize services**
```bash
# Backend services
cd services/red-team-ai && pip install -e .
cd services/blue-team-ai && pip install -e .
cd services/evolution-engine && pip install -e .

# Web app
cd apps/web && npm install && npm run dev

# API Gateway
cd apps/api && uvicorn main:app --reload
```

4. **Access applications**
- **Web**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3001
- **Kibana**: http://localhost:5601
- **Neo4j**: http://localhost:7474

---

## Architecture Deep Dive

### 1. Red Team AI (Autonomous Attack Agent)

**Location**: `services/red-team-ai/`

**Custom AI Components**:
- **Policy Network**: Custom neural net with multi-head attention
- **Value Network**: Advantage estimation
- **PPO Algorithm**: Implemented from scratch
- **Attack Planner**: Transformer-based strategy generation
- **Evasion Module**: Adversarial ML for detection avoidance

**Key Files**:
- `src/agents/attack_agent.py` - Main agent logic
- `src/models/` - Custom neural architectures
- `training/train_red_agent.py` - Training loop

**Training**:
```bash
cd services/red-team-ai
python training/train_red_agent.py --config configs/ppo_config.yaml
```

### 2. Blue Team AI (Autonomous Defense Agent)

**Location**: `services/blue-team-ai/`

**Custom ML Components**:
- **Anomaly Detector**: Custom autoencoder
- **Temporal Analyzer**: Bidirectional LSTM
- **Response Policy**: Policy gradient network
- **Alert Correlator**: Graph-based correlation
- **Adaptive Thresholds**: Dynamic threshold management

**Key Files**:
- `src/agents/defense_agent.py` - Main agent logic
- `src/detection/` - Anomaly detection modules
- `src/response/` - Automated response

### 3. Co-Evolution Engine

**Location**: `services/evolution-engine/`

**Algorithms**:
- **Multi-Agent RL (MARL)** - Red-blue simultaneous training
- **Self-Play** - Agents play against themselves
- **Population-Based Training (PBT)** - Multiple agent variants
- **Curriculum Learning** - Progressive difficulty
- **Nash Equilibrium Detection** - Convergence detection

**Key Files**:
- `src/core/evolution_loop.py` - Main training loop
- `src/marl/` - MARL algorithms
- `src/curriculum/` - Curriculum manager

**Run Evolution**:
```bash
cd services/evolution-engine
python -m src.core.evolution_loop --episodes 1000
```

### 4. Custom AI/ML Models

**Location**: `ml/models/`

#### Custom Transformer (`custom_transformer.py`)
- Built from scratch using PyTorch
- Multi-head attention mechanism
- Positional encoding
- Encoder-decoder architecture
- Used for attack strategy generation

#### Custom NLP Engine (`custom_nlp.py`)
- LSTM-based sequence-to-sequence
- Attention mechanism
- Custom tokenizer
- Report generation from structured data

**Usage**:
```python
from ml.models.custom_transformer import AttackStrategyTransformer

model = AttackStrategyTransformer(action_vocab_size=100)
attack_plan = model.plan_attack(environment_state, max_steps=50)
```

### 5. Cyber Range (Simulation Environment)

**Location**: `services/cyber-range/`

**Components**:
- **Virtual Network**: Docker-based enterprise simulation
- **Gymnasium Environment**: RL-compatible interface
- **Monitoring**: Traffic capture, log aggregation
- **Scenarios**: Pre-built attack scenarios

**Key Files**:
- `src/environment/cyber_env.py` - Gymnasium environment
- `docker/` - Docker network configuration

**Usage**:
```python
from cyber_range.src.environment.cyber_env import CyberRangeEnv

env = CyberRangeEnv(difficulty=0.5)
obs, info = env.reset()

for _ in range(1000):
    actions = {
        'red_action': red_agent.select_action(obs),
        'blue_action': blue_agent.select_action(obs)
    }
    obs, rewards, done, truncated, info = env.step(actions)
```

---

## Development Workflow

### 1. Training AI Agents

```bash
# Train red agent
cd services/red-team-ai
python training/train_red_agent.py

# Train blue agent
cd services/blue-team-ai
python training/train_blue_agent.py

# Co-evolution training
cd services/evolution-engine
python -m src.core.evolution_loop
```

### 2. Frontend Development

```bash
# Web app
cd apps/web
npm run dev

# Mobile app
cd apps/mobile
npm start

# Desktop app
cd apps/desktop
npm run dev
```

### 3. API Development

```bash
cd apps/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Testing

```bash
# Backend tests
pytest services/*/tests/

# Frontend tests
npm test

# Integration tests
pytest tests/integration/
```

---

## Deployment

### Docker Deployment

```bash
# Build all services
docker-compose build

# Deploy
docker-compose up -d

# Scale services
docker-compose up -d --scale red-team-ai=3
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f infrastructure/kubernetes/

# Check status
kubectl get pods -n yugmastra

# View logs
kubectl logs -f deployment/api-gateway -n yugmastra
```

### Cloud Deployment (AWS/GCP/Azure)

```bash
# Terraform
cd infrastructure/terraform/aws
terraform init
terraform plan
terraform apply
```

---

## Monitoring & Observability

### Grafana Dashboards
- **Evolution Metrics**: Red vs Blue performance
- **System Metrics**: CPU, memory, network
- **AI Metrics**: Training loss, reward curves

### Prometheus Metrics
- Custom metrics exported from all services
- Training progress
- API latency

### ELK Stack
- Centralized logging
- Attack/defense event logs
- Error tracking

### Access Monitoring
```bash
# Grafana
http://localhost:3001
# Default: admin/admin

# Prometheus
http://localhost:9090

# Kibana
http://localhost:5601
```

---

## Research & Publications

### Potential Research Papers

1. **"YUGMĀSTRA: Self-Play MARL for Autonomous Cyber Defense"**
   - Venue: USENIX Security / IEEE S&P
   - Contribution: Novel co-evolutionary framework

2. **"Zero-Day Discovery Through Adversarial AI"**
   - Venue: ACM CCS
   - Contribution: AI-discovered attack patterns

3. **"Explainable AI for Security Operations"**
   - Venue: NDSS
   - Contribution: Knowledge distillation for SOC

### Metrics to Track
- Zero-day-like pattern discovery rate
- Defense strategy adaptation speed
- Nash equilibrium convergence time
- False positive reduction over time
- Strategy diversity metrics

---

## Next Steps

### Phase 1: Core Implementation (Months 1-4)
- [x] Project structure
- [x] Custom AI models
- [x] Core services scaffolding
- [ ] Cyber range implementation
- [ ] Basic web UI
- [ ] Integration testing

### Phase 2: Advanced Features (Months 5-8)
- [ ] Full co-evolution training
- [ ] Knowledge graph integration
- [ ] Mobile apps
- [ ] Desktop apps
- [ ] Advanced visualizations

### Phase 3: Research & Publication (Months 9-12)
- [ ] Benchmark against baselines
- [ ] Collect experimental data
- [ ] Write research papers
- [ ] Submit to conferences

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: contact@yugmastra.ai

---

## License

MIT License - see [LICENSE](LICENSE)

---

**Built with cutting-edge AI for the future of autonomous cybersecurity research** 🛡️🤖
