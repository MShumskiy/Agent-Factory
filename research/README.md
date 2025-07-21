# Research Guidelines

## 🧪 Experimentation Workflow

### 1. Starting a New Experiment

1. **Create experiment directory**:
   ```bash
   mkdir research/experiments/{domain}/{experiment_name}
   cd research/experiments/{domain}/{experiment_name}
   ```

2. **Document your experiment**:
   Create `experiment_config.json`:
   ```json
   {
     "name": "chunk_size_optimization",
     "objective": "Optimize chunk size for document ingestion",
     "hypothesis": "Larger chunks improve retrieval accuracy",
     "metrics": ["accuracy", "recall", "processing_time"],
     "baseline": "current_production_pipeline",
     "data": "test_dataset_v1",
     "duration_estimate": "2 days"
   }
   ```

### 2. Running Experiments

- Use `research/notebooks/prototypes/` for initial exploration
- Move to `research/experiments/{domain}/` for structured experiments
- Store configurations in `research/research_configs/`

### 3. Results Management

- **Metrics**: Store in `outputs/experiments_results/`
- **Models**: Save to `outputs/models/` with clear naming
- **Visualizations**: Save to `outputs/visualizations/`
- **Reports**: Document findings in `outputs/reports/`

### 4. Integration to Production

#### Criteria for Promotion:
- [ ] Significant improvement over baseline
- [ ] Proper testing and validation
- [ ] Documentation and code quality
- [ ] No breaking changes to existing APIs

#### Promotion Process:
1. Create feature branch: `feature/experiment-{name}`
2. Move code to `src/agent_factory/`
3. Add tests in `tests/`
4. Update documentation
5. Create pull request

## 📊 Experiment Templates

### Ingestion Pipeline Experiment
```python
# research/experiments/ingestion/experiment_template.py
import sys
sys.path.append('../../../src')

from agent_factory.pipelines import IngPipeline
from research.research_configs import load_experiment_config

# Load experiment configuration
config = load_experiment_config("ingestion_experiment.json")

# Run experiment
results = run_experiment(config)

# Save results
save_results(results, "outputs/experiments_results/")
```

### Agent Performance Experiment
```python
# research/experiments/agents/performance_test.py
from agent_factory.agents import KnowledgeAgent
from research.utils import evaluate_agent

# Test different configurations
configs = load_test_configurations()
results = {}

for config_name, config in configs.items():
    agent = KnowledgeAgent(config)
    performance = evaluate_agent(agent, test_dataset)
    results[config_name] = performance

# Save comparative results
save_comparison_report(results)
```

## 🔧 Research Utilities

Create shared utilities in `research/utils/`:
- `experiment_runner.py`: Common experiment patterns
- `metrics.py`: Evaluation metrics
- `visualization.py`: Plotting utilities
- `data_loader.py`: Dataset loading helpers
