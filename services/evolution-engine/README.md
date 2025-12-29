# Co-Evolution Engine

The heart of YUGMĀSTRA - manages adversarial co-evolution between Red and Blue AI agents.

## Architecture

```
evolution-engine/
├── src/
│   ├── core/                # Main evolution loop
│   ├── marl/                # Multi-Agent RL
│   ├── curriculum/          # Difficulty scaling
│   ├── equilibrium/         # Nash equilibrium detection
│   └── population/          # Population-based training
├── configs/                 # Evolution configurations
└── tests/                   # Unit tests
```

## Key Algorithms

### 1. Self-Play Training
- Simultaneous red-blue training
- Curriculum learning
- Automatic difficulty adjustment

### 2. Population-Based Training
- Multiple agent variants
- Tournament selection
- Genetic algorithm for hyperparameters

### 3. Equilibrium Detection
- Nash equilibrium metrics
- Convergence detection
- Strategy diversity measurement

## Usage

```bash
python -m src.core.evolution_loop --config configs/default.yaml
```
