# AccessibleDeepAgent

**A fairness-focused AI agent framework with neuroadaptive accessibility capabilities**

AccessibleDeepAgent is a comprehensive research platform that combines a multi-tool reasoning agent (DeepAgent) with advanced bias-mitigation workflows designed for the Humane Intelligence Accessibility Bias Bounty. The codebase (~9,600 lines of code) provides both a powerful general-purpose agent framework and specialized neuroadaptive accessibility tools to address fairness gaps in emotion AI systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [DeepAgent Framework](#deepagent-framework)
- [ADK: Neuroadaptive Accessibility Agent](#adk-neuroadaptive-accessibility-agent)
- [Fairness & Bias Mitigation](#fairness--bias-mitigation)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Advanced Usage](#advanced-usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

AccessibleDeepAgent serves two primary purposes:

1. **DeepAgent Framework**: A research-grade multi-tool reasoning agent that coordinates LLM orchestration, tool search, action planning, and evaluation across multiple benchmark suites (ToolBench, GAIA, API-Bank, RestBench, ALFWorld, WebShop, ToolHop).

2. **ADK (Accessibility Development Kit)**: A neuroadaptive accessibility system that addresses emotion AI bias against neurodivergent users (particularly alexithymia) through bidirectional reasoning, automated fairness testing (BeTaL), and real-time cognitive state adaptation.

The platform enables researchers to:
- Reproduce competition submissions for accessibility bias bounties
- Run agents against any supported benchmark suite
- Experiment with neuroinclusive accessibility policies
- Develop and test bias mitigation strategies
- Create custom fairness evaluation harnesses

---

## Key Features

### DeepAgent Framework

- **LLM Orchestration**: Coordinates reasoning models, auxiliary models, and thought folding for multi-step planning and self-reflection
- **Tool Layer**: Executable adapters for ToolBench/ToolHop APIs, RapidAPI, Python execution, web search, audio/vision utilities
- **Prompt Engineering**: System prompts for open/closed-set evaluation, tool intent classification, and dataset-specific templates
- **Evaluation Harness**: Dataset-aligned scripts computing metrics across 7+ benchmark suites
- **Async Processing**: Concurrent LLM completions with rate limiting and timeout handling

### ADK: Neuroadaptive Accessibility

- **Real-time Signal Processing**: Monitors eye tracking, interaction patterns, mouse movement, and device sensors
- **Cognitive State Estimation**: Estimates cognitive load, attention, fatigue, stress, and reading comprehension
- **Bidirectional Reasoning**: Prevents emotion AI bias through forward/reverse verification (40% FNR reduction for alexithymic users)
- **BeTaL Automated Testing**: LLM-guided benchmark generation achieving 5.8% fairness gap (vs 12.5% baseline)
- **Memory System**: Persistent user profiles and adaptation history using mem0.ai
- **UI Adaptation**: Real-time accessibility adjustments based on cognitive state
- **Contrastive Learning**: Ensures semantic consistency across reasoning paths

---

## Architecture

### System Overview

```
AccessibleDeepAgent
├── DeepAgent Core (Multi-tool Reasoning)
│   ├── LLM Orchestration Layer
│   ├── Tool Search & Execution
│   ├── Action Planning & Reflection
│   └── Evaluation Harness
│
└── ADK (Neuroadaptive Accessibility)
    ├── Loop A: Signal Normalization
    ├── Loop B: State Estimation (+ XGC-AVis)
    ├── CMS: Continuum Memory System (mem0.ai)
    ├── Loop C: Content Refinement (Factuality, Personalization, Coherence)
    ├── UI Adaptation Engine
    ├── Bidirectional Reasoning Network (Bias Mitigation)
    ├── BeTaL: Automated Fairness Testing
    └── Loop E: Logging & Evaluation
```

### Component Interaction

1. **DeepAgent** handles general-purpose task solving and benchmark evaluation
2. **ADK** provides accessibility-aware enhancements and fairness guarantees
3. **Shared Infrastructure**: Both systems use common utilities for LLM calls, logging, and configuration management

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for local model serving)
- 8GB+ RAM recommended

### Core Installation

```bash
# Clone the repository
git clone https://github.com/Tuesdaythe13th/AccessibleDeepAgent.git
cd AccessibleDeepAgent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### ADK Installation (Optional)

For neuroadaptive accessibility features:

```bash
# Install ADK-specific dependencies
pip install -r requirements-adk.txt

# Verify installation
python -c "from adk.agents.core import AccessibilityCoordinator; print('ADK installed successfully')"
```

### Configuration

Edit `config/base_config.yaml` to set up API keys and dataset paths:

```yaml
# API Keys
toolbench_url: "YOUR_TOOLBENCH_SERVICE_URL"
google_serper_api_key: "YOUR_SERPER_KEY"
jina_api_key: "YOUR_JINA_KEY"

# Model Servers
reasoning_model:
  url: "http://localhost:8000/v1"
  api_key: "YOUR_API_KEY"

# Dataset Paths
gaia_data_path: "./data/GAIA/dataset.json"
toolbench_data_path: "./data/ToolBench/dataset.json"
```

For ADK configuration, see `src/adk/config/adk_config.yaml`.

---

## Quick Start

### DeepAgent: Run an Agent Task

```bash
# Run agent on GAIA benchmark
python src/run_deep_agent.py \
    --config_path ./config/base_config.yaml \
    --dataset_name gaia \
    --split test \
    --subset_num 32 \
    --enable_tool_search \
    --enable_thought_folding \
    --max_action_limit 30 \
    --eval

# Run single question
python src/run_deep_agent.py \
    --config_path ./config/base_config.yaml \
    --single_question "What is the capital of France?" \
    --enable_tool_search
```

### ADK: Basic Accessibility Agent

```python
import asyncio
from adk.agents.core import AccessibilityCoordinator
from adk.utils import SignalType

async def main():
    # Initialize coordinator
    coordinator = AccessibilityCoordinator()
    await coordinator.initialize()

    # Start session
    session_id = await coordinator.start_session(user_id="user123")

    # Process user interaction with signals
    raw_signals = [
        (SignalType.EYE_TRACKING, 0.7, {"device": "webcam"}),
        (SignalType.INTERACTION_TIMING, 0.65, {"avg_response_time_ms": 850}),
        (SignalType.MOUSE_MOVEMENT, 0.55, {"movement_pattern": "erratic"}),
    ]

    content = "Your content to make accessible..."

    result = await coordinator.process_user_interaction(
        raw_signals=raw_signals,
        user_id="user123",
        content_to_refine=content,
        context={"page": "documentation"}
    )

    print(f"Cognitive Load: {result['cognitive_state']['cognitive_load']:.2f}")
    print(f"UI Adaptations: {len(result['ui_adaptations'])}")
    print(f"Refined Content: {result['refined_content']}")

    # End session
    await coordinator.end_session()
    await coordinator.close()

asyncio.run(main())
```

### Run Accessibility Bias Analysis

```bash
# Run the Jupyter notebook for Valence API bias analysis
jupyter notebook bounty_valence_analysis.ipynb

# Or run the Python script version
python src/adk/examples/bounty_valence_analysis_corrected.py
```

---

## Repository Structure

```
AccessibleDeepAgent/
├── config/                      # Configuration files
│   ├── base_config.yaml        # Main DeepAgent configuration
│   └── alfworld_config.yaml    # ALFWorld-specific settings
│
├── data/                        # Benchmark datasets
│   ├── API-Bank/               # API-Bank benchmark data
│   ├── GAIA/                   # GAIA benchmark data
│   ├── ToolBench/              # ToolBench data
│   ├── RestBench/              # RestBench data
│   ├── ALFWorld/               # ALFWorld environments
│   └── WebShop/                # WebShop data
│
├── docs/                        # Documentation
│   └── ADVANCED_DATA_NOTEBOOK.md  # Bias bounty submission guide
│
├── src/                         # Source code
│   ├── adk/                    # Neuroadaptive Accessibility Agent
│   │   ├── agents/             # Agent implementations
│   │   │   ├── core/          # Core orchestration (Coordinator, Pipeline, Policy)
│   │   │   ├── loop_a/        # Signal normalization
│   │   │   ├── loop_b/        # State estimation
│   │   │   ├── loop_c/        # Content refinement
│   │   │   └── loop_e/        # Logging & evaluation
│   │   ├── betal/             # Automated fairness testing
│   │   ├── config/            # ADK configuration
│   │   ├── docs/              # ADK documentation
│   │   ├── evaluation/        # Bias metrics
│   │   ├── examples/          # Usage examples
│   │   ├── tools/             # Memory system (mem0.ai)
│   │   ├── training/          # Model training utilities
│   │   ├── utils/             # ADK utilities
│   │   ├── bidirectional_reasoning.py  # Bias mitigation network
│   │   ├── neuroadaptive_wrapper.py    # High-level wrapper
│   │   └── run_accessibility_agent.py  # ADK entry point
│   │
│   ├── envs/                   # Environment wrappers (GAIA, etc.)
│   ├── evaluate/               # Evaluation scripts per benchmark
│   │   ├── evaluate_base.py   # Shared evaluation logic
│   │   ├── evaluate_toolbench.py
│   │   ├── evaluate_gaia.py
│   │   └── ...
│   ├── prompts/                # Prompt templates
│   │   ├── prompts_deepagent.py
│   │   ├── prompts_react.py
│   │   ├── prompts_webthinker.py
│   │   └── task_specific_prompts.py
│   ├── tools/                  # Tool implementations
│   │   └── tool_manager.py    # Tool orchestration
│   ├── utils/                  # Shared utilities
│   └── run_deep_agent.py      # Main DeepAgent entry point
│
├── bounty_valence_analysis.ipynb  # Jupyter notebook for bias analysis
├── verify_results.py           # Results verification script
├── requirements.txt            # Core dependencies
├── requirements-adk.txt        # ADK-specific dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## DeepAgent Framework

### Supported Benchmarks

| Benchmark | Type | Metrics | Status |
|-----------|------|---------|--------|
| **ToolBench** | Tool use | Success rate, efficiency | ✅ Supported |
| **GAIA** | QA with tools | Accuracy | ✅ Supported |
| **API-Bank** | API calling | Success rate | ✅ Supported |
| **RestBench** | REST API use | API correctness | ✅ Supported |
| **ToolHop** | Multi-hop tool use | Path accuracy | ✅ Supported |
| **ALFWorld** | Embodied tasks | Goal completion | ✅ Supported |
| **WebShop** | Web navigation | Reward score | ✅ Supported |

### Core Components

#### 1. LLM Orchestration (`src/run_deep_agent.py`)

Coordinates:
- Reasoning model (primary task solver)
- Auxiliary model (tool selection, reflection)
- Tool search retriever
- Thought folding (self-reflection)
- Episode/working/tool memory

#### 2. Tool Layer (`src/tools/`)

Provides executable interfaces for:
- ToolBench/ToolHop APIs
- RapidAPI integration
- Python code execution
- Web search (Serper, Jina)
- Audio processing (Whisper)
- Vision utilities
- Database queries

#### 3. Prompt Engineering (`src/prompts/`)

Templates for:
- Open-set QA (GAIA, general questions)
- Closed-set tasks (ToolBench, API-Bank)
- Embodied tasks (ALFWorld)
- Web navigation (WebShop)
- Tool intent classification
- Thought folding instructions

#### 4. Evaluation Harness (`src/evaluate/`)

Dataset-specific evaluators:
- `evaluate_toolbench.py`: ToolBench success metrics
- `evaluate_gaia.py`: GAIA accuracy
- `evaluate_api_bank.py`: API-Bank correctness
- `evaluate_base.py`: Shared evaluation infrastructure

### Usage Examples

#### Run ToolBench Evaluation

```bash
python src/run_deep_agent.py \
    --config_path ./config/base_config.yaml \
    --dataset_name toolbench \
    --split test \
    --enable_tool_search \
    --top_k 5 \
    --concurrent_limit 10 \
    --eval
```

#### Run ALFWorld Tasks

```bash
python src/run_deep_agent.py \
    --config_path ./config/alfworld_config.yaml \
    --dataset_name alfworld \
    --max_action_limit 50 \
    --eval
```

#### Evaluate Existing Predictions

```bash
python src/evaluate/evaluate_toolbench.py \
    --prediction_path ./results/predictions.json \
    --split test
```

---

## ADK: Neuroadaptive Accessibility Agent

The ADK implements a multi-loop architecture for real-time accessibility adaptation:

### Loop Architecture

#### Loop A: Signal Normalization
- **Agent**: `SignalNormalizer`
- **Function**: Normalizes heterogeneous user signals (eye tracking, interaction timing, mouse movement)
- **Strategies**: Z-score, min-max, robust normalization
- **Output**: Standardized signal vectors

#### Loop B: State Estimation
- **Agent**: `StateEstimator`
- **Function**: Estimates cognitive state from normalized signals
- **Optional**: XGC-AVis integration for advanced ML-based estimation
- **Output**: Cognitive load, attention, fatigue, stress, comprehension scores

#### Continuum Memory System (CMS)
- **Components**: `MemoryManager`, `MemoryStore` (mem0.ai)
- **Function**: Persistent storage of user preferences, accessibility profiles, interaction history
- **Features**: Semantic search, temporal decay, cross-session learning

#### Loop C: Content Refinement
- **Agents**: `FactualityAgent`, `PersonalizationAgent`, `CoherenceAgent`
- **Coordinator**: `RefinementCoordinator` (meta-agent)
- **Function**: Iteratively refines content for factuality, personalization, and coherence
- **Output**: Adapted content matching user cognitive state

#### UI Adaptation Engine
- **Agent**: `UIAdaptationAgent`
- **Function**: Generates real-time UI modifications
- **Categories**: Text size, contrast, color scheme, layout density, animation speed, audio, language
- **Priority**: Ranked recommendations based on cognitive state severity

#### Loop E: Logging & Evaluation
- **Agents**: `LoggingAndEvalAgent`, `LoopStopChecker`
- **Function**: Dual logging (system + evaluation), loop termination decisions
- **Metrics**: Latency, accessibility score, refinement iterations, success rate, fairness metrics

### Running the ADK

```bash
# Demo mode (single interaction)
python src/adk/run_accessibility_agent.py --mode demo --user-id user123

# Interactive mode (manual signal input)
python src/adk/run_accessibility_agent.py --mode interactive --user-id user123

# Streaming mode (continuous processing)
python src/adk/run_accessibility_agent.py --mode stream --user-id user123

# Custom configuration
python src/adk/run_accessibility_agent.py --config src/adk/config/custom_config.yaml
```

### ADK Examples

```bash
# Basic usage
python src/adk/examples/basic_usage.py

# Advanced usage with custom profiles
python src/adk/examples/advanced_usage.py

# Bias mitigation demonstration
python src/adk/examples/bias_mitigation_demo.py

# BeTaL automated fairness testing
python src/adk/examples/betal_demo.py

# Bounty submission analysis
python src/adk/examples/bounty_valence_analysis_corrected.py
```

---

## Fairness & Bias Mitigation

AccessibleDeepAgent implements two complementary bias mitigation strategies:

### 1. Bidirectional Reasoning Network

**Problem**: Traditional emotion AI systems exhibit high false negative rates for neurodivergent users with alexithymia (difficulty expressing emotions), leading to unfair treatment.

**Solution**: Bidirectional verification instead of unidirectional classification.

```
Traditional:  Audio → [Encoder] → Emotion (one-way)
Bidirectional: Audio → [Encoder] → Emotion → [Decoder] → Reconstructed Audio
              ↓                                              ↓
           Embedding 1                                  Embedding 2
              └──────────── Contrastive Loss ────────────────┘
```

**Key Features**:
- Forward path: Audio → Emotion prediction
- Reverse path: Emotion → Audio reconstruction
- Contrastive learning ensures semantic consistency
- Mismatch detection triggers alexithymia-aware handling
- **Result**: 40% reduction in false negative rate for alexithymic users

**Implementation**: `src/adk/bidirectional_reasoning.py`

### 2. BeTaL: Automated Fairness Testing

**Problem**: Manual benchmark design misses edge cases and requires extensive domain expertise.

**Solution**: LLM-guided automated benchmark generation.

```
Designer LLM (Claude Opus)
    ↓ proposes benchmark parameters
Student LLM (o4-mini)
    ↓ evaluated on benchmark
Feedback Loop
    ↓ optimizes for fairness gaps
Converged Benchmark (5 iterations)
```

**Key Features**:
- Designer model proposes test scenarios targeting fairness
- Student model is evaluated on bias metrics
- Feedback loop systematically finds challenging cases
- **Result**: 5.8% fairness gap (vs 12.5% for baseline methods)

**Implementation**: `src/adk/betal/`

### Fairness Metrics

The system tracks:
- **False Negative Rate (FNR)** per demographic group
- **Verification Parity**: Equal verification rates across groups
- **Accuracy Parity**: Equal accuracy across groups
- **Alexithymia Adaptation Success**: Correct handling of flat affect

See `src/adk/evaluation/bias_metrics.py` for implementation.

---

## Evaluation & Benchmarks

### Running Evaluations

```bash
# DeepAgent evaluation on multiple benchmarks
python src/run_deep_agent.py \
    --dataset_name toolbench \
    --eval \
    --output_path ./results/

# Manual evaluation
python src/evaluate/evaluate_toolbench.py \
    --prediction_path ./results/predictions.json \
    --split test

# Verify results
python verify_results.py --results_dir ./results/
```

### Metrics by Benchmark

- **ToolBench**: Success rate, efficiency (tool calls per task)
- **GAIA**: Exact match accuracy
- **API-Bank**: API correctness, parameter accuracy
- **ALFWorld**: Goal completion rate, steps to completion
- **WebShop**: Reward score, purchase accuracy
- **ADK Fairness**: FNR parity, verification parity, accuracy parity

---

## Advanced Usage

### Custom Tool Integration

```python
from tools.tool_manager import ToolManager

# Register custom tool
tool_manager = ToolManager(config)
tool_manager.register_tool(
    name="custom_api",
    description="My custom API",
    execute_fn=my_custom_function,
    parameters={"param1": "string", "param2": "int"}
)
```

### Custom Accessibility Profiles

```python
from adk.utils import AccessibilityProfile

profile = AccessibilityProfile(
    profile_id="profile_dyslexia",
    profile_name="Dyslexia Friendly",
    user_id="user123",
    settings={
        "font_family": "OpenDyslexic",
        "text_size": 1.2,
        "letter_spacing": 1.3,
        "line_height": 1.8,
        "simplified_language": True,
        "reduce_cognitive_load": True
    }
)

await memory_manager.save_accessibility_profile(profile)
```

### Thought Folding & Self-Reflection

```bash
python src/run_deep_agent.py \
    --enable_thought_folding \
    --max_fold_limit 3 \
    --fold_threshold 0.7 \
    --dataset_name gaia
```

### Concurrent Evaluation

```bash
python src/run_deep_agent.py \
    --dataset_name toolbench \
    --concurrent_limit 20 \
    --timeout 120 \
    --eval
```

---

## Documentation

### Primary Documentation

- **Main README**: [README.md](README.md) (this file)
- **ADK Documentation**: [src/adk/docs/README.md](src/adk/docs/README.md)
- **Bidirectional Reasoning**: [src/adk/docs/BIDIRECTIONAL_REASONING.md](src/adk/docs/BIDIRECTIONAL_REASONING.md)
- **BeTaL Framework**: [src/adk/docs/BETAL.md](src/adk/docs/BETAL.md)
- **Bias Bounty Submission**: [docs/ADVANCED_DATA_NOTEBOOK.md](docs/ADVANCED_DATA_NOTEBOOK.md)
- **Detailed Results**: [src/adk/docs/DETAILED_RESULTS.md](src/adk/docs/DETAILED_RESULTS.md)

### Additional Resources

- **Configuration Guide**: See comments in `config/base_config.yaml` and `src/adk/config/adk_config.yaml`
- **API Documentation**: See docstrings in source files
- **Examples**: Review `src/adk/examples/` for usage patterns

---

## Contributing

We welcome contributions! Areas of particular interest:

- Additional benchmark integrations
- Improved fairness evaluation metrics
- New accessibility adaptation strategies
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/AccessibleDeepAgent.git
cd AccessibleDeepAgent

# Create branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-adk.txt

# Run tests (when available)
pytest tests/

# Make changes and commit
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

---

## Citation

If you use AccessibleDeepAgent in your research, please cite:

```bibtex
@software{accessibledeepagent2025,
  title={AccessibleDeepAgent: A Fairness-Focused Multi-Tool Reasoning Agent with Neuroadaptive Accessibility},
  author={Tuesday, ARTIFEX Labs},
  year={2025},
  url={https://github.com/Tuesdaythe13th/AccessibleDeepAgent},
  note={Framework for bias-aware AI agents with neuroinclusive accessibility}
}
```

```

---

## License

This project is licensed under the MIT License

---

## Acknowledgments

- **Humane Intelligence Accessibility Bias Bounty**: For motivating the fairness-focused development
- **DeepAgent Research**: Foundation for the multi-tool reasoning framework
- **Google ADK**: Architecture inspiration for the neuroadaptive system
- **mem0.ai**: Memory system infrastructure
- **OpenAI, Anthropic**: LLM providers for reasoning and content refinement
- **Valence AI**: Emotion API for bias analysis baseline

---

## Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Tuesdaythe13th/AccessibleDeepAgent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tuesdaythe13th/AccessibleDeepAgent/discussions)
- **Email**: general@artifex.fun

---

## Project Status

🚀 **Active Development** | ⭐ **Research Preview**

This is a research project under active development. APIs may change, and some features are experimental. Production deployment should include additional safety measures and testing.

### Recent Updates

- ✅ Bidirectional reasoning for emotion AI fairness
- ✅ BeTaL automated fairness testing
- ✅ Multi-benchmark evaluation harness
- ✅ Neuroadaptive accessibility agent (ADK)
- ✅ Humane Intelligence Bias Bounty submission

### Roadmap

- [ ] Real-world validation with Valence partnership
- [ ] Integration with production-grade LLM serving
- [ ] Browser extension for signal collection
- [ ] Mobile app support
- [ ] Multi-language accessibility support
- [ ] Large-scale fairness evaluation
- [ ] A/B testing framework
- [ ] User feedback integration
- [ ] Extended benchmark coverage

---

**Built with fairness and accessibility at the core. Innovating on the love of making ai for everyone.**
