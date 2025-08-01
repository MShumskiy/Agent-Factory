# Agent Factory

An AI agent development platform with integrated experimentation capabilities.

## 🏗️ Repository Structure

### 📦 Production Code (`src/agent_factory/`)
Production-ready, tested, and deployable code:
- **agents/**: Production AI agents
- **pipelines/**: Data processing pipelines  
- **core/**: Core utilities and shared functionality
- **api/**: REST API endpoints
- **config/**: Configuration management

### 🧪 Research & Experimentation (`research/`)
Space for experimentation and research:
- **experiments/**: Organized experiments by domain
- **notebooks/**: Research notebooks for analysis and prototyping

### 📊 Data Layer (`data/`)
Centralized data management:
- **datasets/**: Raw, processed, and synthetic datasets
- **knowledge_bases/**: Domain-specific knowledge bases
- **test_data/**: Data for testing

### 📤 Outputs (`outputs/`)
Generated artifacts and results:
- **models/**: Trained models and checkpoints
- **reports/**: Research reports and documentation
- **visualizations/**: Charts, graphs, and plots
- **experiments_results/**: Experiment outputs and metrics

## 🚀 Quick Start

### Production Usage
```python
from agent_factory.agents import KnowledgeAgent
from agent_factory.pipelines import IngestionPipeline

# Use production components
agent = KnowledgeAgent()
pipeline = IngestionPipeline()
```

### Research & Experimentation
```bash
# Navigate to research area
cd research/experiments/ingestion/

# Run experiments
jupyter notebook ingestion_experiments.ipynb
```

## 🔬 Experiment Integration Workflow

1. **Prototype** in `research/notebooks/prototypes/`
2. **Experiment** in `research/experiments/{domain}/`
3. **Evaluate** results in `outputs/experiments_results/`
4. **Promote** successful experiments to `src/agent_factory/`

## 📋 Development Guidelines

### Production Code Standards
- Full test coverage in `tests/`
- Documentation and type hints
- Configuration via `src/agent_factory/config/`
- Follow established patterns

### Research Code Guidelines
- Document experiments with clear objectives
- Use consistent naming conventions
- Store results in appropriate `outputs/` subdirectories
- Include environment and dependency information

## 🔧 Configuration

- **Production configs**: `configs/`
- **Research configs**: `research/research_configs/`
- **Environment variables**: `.env`

## 📚 Documentation

- **API Documentation**: `docs/api/`
- **Architecture**: `docs/architecture/`
- **User Guides**: `docs/user-guides/`