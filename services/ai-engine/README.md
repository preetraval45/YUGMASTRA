# YUGMĀSTRA AI Engine

Advanced AI/ML Engine for Cybersecurity Intelligence with custom LLM training, RAG, and multi-agent systems.

## 🚀 Features

### 1. **Custom LLM Training Pipeline**
- Fine-tune GPT-2, DistilBERT, and other transformer models
- Train on cybersecurity datasets (MITRE ATT&CK, CVE databases, threat intelligence)
- Support for PyTorch and TensorFlow models
- Automated model training and evaluation

### 2. **RAG (Retrieval-Augmented Generation)**
- Vector database integration with ChromaDB
- Semantic search across cybersecurity knowledge bases
- Automatic document ingestion and embedding
- Context-aware response generation

### 3. **Multi-Agent AI System**

#### Red Team Agent
- Offensive security analysis
- Attack vector identification
- Penetration testing strategies
- Exploitation techniques
- MITRE ATT&CK mapping

#### Blue Team Agent
- Defensive security analysis
- Threat detection strategies
- Incident response procedures
- Security hardening recommendations
- SIEM/SOC operations

#### Evolution Agent
- Adaptive threat intelligence
- Combines red and blue team perspectives
- Emerging threat analysis
- Predictive security modeling
- Zero-day vulnerability tracking

### 4. **Vector Store (FAISS)**
- High-performance similarity search
- Multiple index types for different data
- Efficient embedding storage and retrieval
- Real-time threat pattern matching

### 5. **Deep Learning Models**
- Threat classification models
- Anomaly detection with autoencoders
- NLP for security text analysis
- Time-series analysis for attack prediction

## 📦 Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional, for faster training)
- Docker and Docker Compose

### Local Setup

```bash
cd services/ai-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run the service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Docker Setup

```bash
# From project root
docker-compose up -d ai-engine
```

## 🔧 Configuration

Edit `.env` file:

```env
DATABASE_URL=postgresql://yugmastra:password@postgres:5432/yugmastra
REDIS_URL=redis://redis:6379/0
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yugmastra

MODEL_DIR=./models
DATA_DIR=./data

HUGGINGFACE_TOKEN=your_token_here  # Optional

LOG_LEVEL=INFO
```

## 📡 API Endpoints

### Health Check
```http
GET /
GET /health
```

### AI Chat
```http
POST /api/ai/chat

{
  "message": "What are the latest ransomware attack techniques?",
  "mode": "evolution",  // red-team | blue-team | evolution
  "history": [],
  "context": {}
}
```

### Train Model
```http
POST /api/ai/train

{
  "dataset_path": "/path/to/dataset.json",
  "model_type": "gpt2",
  "epochs": 3,
  "batch_size": 8
}
```

### Create Embeddings
```http
POST /api/ai/embed

{
  "texts": ["Sample threat description..."],
  "store": true
}
```

### RAG Query
```http
POST /api/ai/rag/query?query=SQL%20injection&top_k=5
```

### Ingest Knowledge
```http
POST /api/ai/ingest

{
  "documents": ["Document 1...", "Document 2..."],
  "metadata": {"source": "MITRE ATT&CK"}
}
```

### Get Models Info
```http
GET /api/ai/models
```

## 🧠 AI Agents Usage

### Red Team Agent
```python
from agents import RedTeamAgent
from models import LLMManager
from services import RAGService

llm = LLMManager()
rag = RAGService()
red_team = RedTeamAgent(llm, rag)

response = await red_team.generate_response(
    message="How can I test for SQL injection?",
    history=[],
    context={}
)
```

### Blue Team Agent
```python
from agents import BlueTeamAgent

blue_team = BlueTeamAgent(llm, rag)

response = await blue_team.generate_response(
    message="How do I detect ransomware attacks?",
    history=[],
    context={}
)
```

### Evolution Agent
```python
from agents import EvolutionAgent
from services import VectorStore

vector_store = VectorStore()
evolution = EvolutionAgent(llm, rag, vector_store)

response = await evolution.generate_response(
    message="What are emerging AI-powered threats?",
    history=[],
    context={}
)
```

## 🎓 Training Custom Models

### Prepare Dataset

Create JSON dataset:
```json
[
  {
    "text": "SQL injection is a code injection technique...",
    "label": "vulnerability"
  },
  {
    "text": "Ransomware encrypts victim files...",
    "label": "malware"
  }
]
```

### Train Model

```bash
curl -X POST http://localhost:8001/api/ai/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/app/data/cybersecurity_dataset.json",
    "model_type": "distilbert",
    "epochs": 5,
    "batch_size": 16
  }'
```

## 📊 Vector Store Operations

### Store Threat Intelligence

```python
from services import VectorStore

vector_store = VectorStore()

threats = [
    "APT29 uses spear-phishing emails...",
    "Log4Shell vulnerability allows remote code execution..."
]

embeddings = await vector_store.embed_texts(threats)
await vector_store.store_embeddings(threats, embeddings, index_type="threats")
```

### Search Similar Threats

```python
results = await vector_store.search(
    query="ransomware attack techniques",
    top_k=5,
    index_type="threats"
)
```

## 🗄️ RAG Knowledge Base

### Ingest MITRE ATT&CK Data

```python
from services import RAGService

rag = RAGService()
await rag.load_mitre_attack_data()
```

### Query Knowledge Base

```python
results = await rag.query(
    query="privilege escalation techniques",
    top_k=5,
    collection_name="mitre_attack"
)
```

## 🔬 Model Architecture

### LLM Manager
- **GPT-2**: Text generation and contextual responses
- **DistilBERT**: Threat classification (5 classes)
- **Sentence Transformers**: Semantic embeddings

### Threat Classification Labels
1. Benign
2. Malware
3. Phishing
4. DoS/DDoS
5. Exploit

## 📈 Performance

- **Response Time**: < 500ms per query
- **Embedding Generation**: ~100 texts/second
- **Vector Search**: < 50ms for 1M vectors
- **Model Inference**: GPU-accelerated

## 🛠️ Development

### Project Structure

```
ai-engine/
├── agents/
│   ├── red_team.py       # Red Team AI agent
│   ├── blue_team.py      # Blue Team AI agent
│   └── evolution.py      # Evolution AI agent
├── models/
│   └── llm_manager.py    # LLM model management
├── services/
│   ├── rag_service.py    # RAG implementation
│   └── vector_store.py   # FAISS vector store
├── utils/
│   └── logger.py         # Logging utilities
├── data/                 # Data storage
├── models/               # Trained models
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
└── Dockerfile           # Docker configuration
```

### Add Custom Agent

1. Create agent file in `agents/`
2. Implement `generate_response()` method
3. Register in `main.py`

```python
from agents.custom import CustomAgent

custom_agent = CustomAgent(llm_manager, rag_service)
```

## 🚀 Next Steps

### Planned Features
- [ ] GPT-3/GPT-4 integration
- [ ] Fine-tuning on larger cybersecurity datasets
- [ ] Real-time threat intelligence feeds
- [ ] Automated vulnerability scanner
- [ ] Multi-modal AI (text + network traffic)
- [ ] Federated learning for privacy
- [ ] Quantum-safe cryptography analysis

### Training Datasets
- MITRE ATT&CK framework
- CVE database (NVD)
- Security research papers
- Dark web threat intelligence
- Malware analysis reports
- Security blog posts and articles

## 📝 License

Part of YUGMĀSTRA - Advanced Cyber Defense Platform

## 🤝 Contributing

This is a custom AI engine built specifically for YUGMĀSTRA. For questions or contributions, contact the development team.

## ⚠️ Important Notes

- **GPU Recommended**: For training and inference, NVIDIA GPU with CUDA support is highly recommended
- **Memory Requirements**: Minimum 8GB RAM, 16GB recommended for training
- **Disk Space**: Models and embeddings require significant storage (10GB+)
- **API Rate Limiting**: Implement rate limiting in production
- **Security**: Never expose API without authentication in production

## 🎯 Use Cases

1. **Threat Analysis**: Analyze security incidents and provide insights
2. **Vulnerability Research**: Identify and explain vulnerabilities
3. **Security Training**: Interactive cybersecurity education
4. **Incident Response**: Guide response procedures
5. **Penetration Testing**: Ethical hacking assistance
6. **Security Auditing**: Automated security assessments

---

**Powered by**: PyTorch, Transformers, LangChain, ChromaDB, FAISS, FastAPI
