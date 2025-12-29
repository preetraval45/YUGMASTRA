# YUGMĀSTRA

**Autonomous Adversary-Defender Co-Evolution Platform**

> Where cybersecurity defenses are not engineered—they emerge.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node 20+](https://img.shields.io/badge/node-20+-green.svg)](https://nodejs.org/)

---

## Vision

YUGMĀSTRA is a groundbreaking AI system where **offensive and defensive cyber agents continuously evolve against each other** in real time—discovering zero-day-like attack strategies and adaptive defenses without human-labeled data.

This is **AlphaGo-style self-play for cybersecurity**.

### Core Thesis

Attack strategies and defense strategies should:
- ❌ NOT be pre-defined
- ❌ NOT be labeled
- ✅ **EMERGE through adversarial self-play**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YUGMĀSTRA PLATFORM                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐           ┌──────────────┐                │
│  │   Red AI     │  ←──→     │   Blue AI    │                │
│  │  (Attacker)  │           │  (Defender)  │                │
│  └──────┬───────┘           └──────┬───────┘                │
│         │                          │                         │
│         └──────────┬───────────────┘                         │
│                    ▼                                          │
│         ┌──────────────────────┐                             │
│         │  Co-Evolution Engine │                             │
│         │   (MARL + Self-Play) │                             │
│         └──────────┬───────────┘                             │
│                    ▼                                          │
│         ┌──────────────────────┐                             │
│         │   Knowledge Graph    │                             │
│         │   (Neo4j + Vectors)  │                             │
│         └──────────┬───────────┘                             │
│                    ▼                                          │
│         ┌──────────────────────┐                             │
│         │ Knowledge Distillation│                            │
│         │    (LLM Explainer)    │                            │
│         └──────────────────────┘                             │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Cyber Range (Simulation)                   │
│  [Web Server] [Database] [Endpoints] [SIEM] [Network]       │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Autonomous Red Team AI
- Generates novel attack strategies using RL + LLMs
- Learns to evade detection systems
- Discovers vulnerability chains
- No pre-scripted attack playbooks

### 2. Autonomous Blue Team AI
- Develops adaptive defense strategies
- Learns from attack patterns
- Generates detection rules automatically
- Reduces false positives over time

### 3. Co-Evolution Engine
- Multi-Agent Reinforcement Learning (MARL)
- Self-play training mechanism
- Automatic difficulty scaling
- Nash equilibrium detection

### 4. Knowledge Distillation
- Converts AI behavior → human-readable insights
- Generates SIEM rules (Sigma format)
- Maps to MITRE ATT&CK framework
- Produces research reports

### 5. Cross-Platform Support
- **Web**: Next.js dashboard
- **Mobile**: iOS & Android (React Native)
- **Desktop**: Windows, macOS, Linux (Electron)

---

## Technology Stack

### Backend & AI
- **Python 3.11+** - Core AI/ML
- **FastAPI** - High-performance APIs
- **PyTorch 2.0+** - Deep learning
- **Ray/RLlib** - Distributed RL
- **Transformers** - LLM integration
- **LangChain** - LLM orchestration

### Frontend
- **Next.js 14** - Web application
- **React Native** - Mobile apps
- **Electron** - Desktop apps
- **TypeScript** - Type safety
- **TailwindCSS** - Styling

### Infrastructure
- **PostgreSQL** - Relational data
- **Neo4j** - Knowledge graph
- **Redis** - Caching
- **Docker** - Containerization
- **Kubernetes** - Orchestration

### Observability
- **ELK Stack** - Logging
- **Prometheus + Grafana** - Metrics
- **OpenTelemetry** - Tracing

---

## Project Structure

```
yugmastra/
├── apps/
│   ├── web/                 # Next.js web app
│   ├── mobile/              # React Native app
│   ├── desktop/             # Electron app
│   └── api/                 # FastAPI gateway
├── packages/
│   ├── ui/                  # Shared React components
│   ├── core/                # Business logic
│   ├── ai-sdk/              # AI client SDK
│   └── types/               # TypeScript types
├── services/
│   ├── red-team-ai/        # Attack agent
│   ├── blue-team-ai/       # Defense agent
│   ├── evolution-engine/   # Co-evolution core
│   ├── knowledge-graph/    # Graph service
│   └── cyber-range/        # Simulation
├── ml/
│   ├── models/             # PyTorch models
│   ├── training/           # Training scripts
│   └── datasets/           # Synthetic data
└── infrastructure/
    ├── docker/             # Docker configs
    ├── kubernetes/         # K8s manifests
    └── terraform/          # IaC
```

---

## Getting Started

### Prerequisites

- **Node.js 20+**
- **Python 3.11+**
- **Docker 24+**
- **Docker Compose 2.20+**
- **Git**

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/yugmastra.git
cd yugmastra

# Install dependencies
npm install
pip install -r requirements.txt

# Start development environment
docker-compose up -d

# Run the platform
npm run dev
```

---

## Research Contributions

This platform enables research in:

1. **Zero-Day Discovery** - Can AI discover novel attack vectors?
2. **Adaptive Defense** - Do evolved defenses outperform rule-based SOCs?
3. **Adversarial Co-Evolution** - How does red-blue equilibrium emerge?
4. **Explainable Security AI** - Can AI-generated policies be trusted?

### Potential Publications

- "Self-Play MARL for Autonomous Cyber Defense" (USENIX Security)
- "Co-Evolutionary AI for Zero-Day Threat Discovery" (IEEE S&P)
- "From Detection to Discovery: AI-Driven Adversarial Cybersecurity" (ACM CCS)

---

## Roadmap

### Phase 1: MVP (Months 1-4)
- ✅ Monorepo setup
- ⏳ Cyber Range simulation
- ⏳ Basic Red/Blue agents
- ⏳ Simple co-evolution loop

### Phase 2: Core Platform (Months 5-8)
- Advanced RL algorithms
- Knowledge graph integration
- Web + Mobile UIs
- Real-time dashboards

### Phase 3: Enterprise (Months 9-12)
- Multi-tenancy
- Federated learning
- Live enterprise integration
- Security hardening

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use YUGMĀSTRA in your research, please cite:

```bibtex
@software{yugmastra2025,
  title={YUGMĀSTRA: Autonomous Adversary-Defender Co-Evolution Platform},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/yugmastra}
}
```

---

## Contact

- **Website**: [yugmastra.ai](https://yugmastra.ai)
- **Email**: contact@yugmastra.ai
- **Twitter**: [@yugmastra](https://twitter.com/yugmastra)

---

**Built with ❤️ for the future of autonomous cybersecurity**
