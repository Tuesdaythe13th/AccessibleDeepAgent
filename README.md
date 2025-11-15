# AccessibleDeepAgent

AccessibleDeepAgent is a fairness-focused fork of the DeepAgent research stack. It keeps the original
multi-tool reasoning agent (tool search, action planning, evaluation harness) and augments it with the
bias-mitigation workflows used for the Humane Intelligence Accessibility Bias Bounty. The codebase is
organized as a reusable toolkit so you can: (1) reproduce the competition submissions, (2) run the agent
against any of the supported benchmark suites, and (3) experiment with new neuroinclusive policies.

---

## Key capabilities

* **LLM Orchestration** – `src/run_deep_agent.py` coordinates the reasoning model, auxiliary model, tool
  search, tool calling, and "thought folding" so the agent can plan, act, and self-reflect while solving
  tasks.
* **Tooling layer** – `src/tools/` contains executable adapters for ToolBench / ToolHop APIs, RapidAPI,
  Python execution, web search, audio/vision utilities, and other calls surfaced to the agent.
* **Prompt suites** – `src/prompts/` holds the system prompts for open- and closed-set evaluation modes,
  tool intent classification, folding instructions, and dataset-specific templates (e.g., ToolHop).
* **Evaluation harness** – `src/evaluate/` provides dataset-aligned scripts that ingest agent logs and
  compute metrics (ToolBench success, GAIA accuracy, ALFWorld scores, WebShop rewards, etc.).
* **Accessibility workflow** – `docs/ADVANCED_DATA_NOTEBOOK.md` documents how the repo backs the Advanced
  Data Track submission, including the baseline Valence API run and the AccessibleDeepAgent mitigation.

---

## Repository layout

| Path | Description |
| --- | --- |
| `config/` | YAML templates for API credentials, dataset paths, and runtime flags. `base_config.yaml` is loaded by default. |
| `data/` | Expected locations for ToolBench, GAIA, API-Bank, RestBench, ToolHop, ALFWorld, WebShop, and other benchmarks. |
| `docs/` | Competition and notebook guides (currently the Advanced Data Track walkthrough). |
| `figures/` | Placeholder for plots or paper figures referenced by the notebook. |
| `src/envs/` | Lightweight wrappers for environment-specific bookkeeping (e.g., GAIA file access). |
| `src/evaluate/` | Metric computation entry points for each dataset; imported by `evaluate/evaluate_base.py`. |
| `src/prompts/` | Global and dataset-specific prompt templates used by the reasoning stack. |
| `src/tools/` | Concrete tool implementations and the `ToolManager` used at inference time. |
| `src/utils/` | Helper utilities for formatting tool responses, caching search results, and parsing generations. |
| `src/run_deep_agent.py` | Main CLI runner for single questions or dataset sweeps, with optional evaluation. |
| `src/run_tool_search_server.py` | Launches the tool-search retriever as a standalone service. |

---

## Getting started

### 1. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project targets Python 3.9+ and relies on GPU-accelerated serving for the primary reasoning models.
If you are calling hosted models instead, make sure the networking credentials in the config file are set.

### 2. Configure credentials and paths

Edit `config/base_config.yaml` (or create your own YAML and pass `--config_path`). Key sections:

* **API keys**: ToolBench service URL, Google Serper, Jina, TMDB/Spotify tokens, WebShop endpoint, etc.
* **Model servers**: URLs and API keys for the reasoning model, auxiliary model, VQA model, and tool retriever.
* **Dataset paths**: File locations for each benchmark JSON plus associated assets (GAIA files, HLE images, …).

All configuration values cascade through `src/run_deep_agent.py`, so a single file captures both credentials
and dataset-specific knobs.

### 3. Run the agent

#### Open-set or closed-set QA

```bash
python src/run_deep_agent.py \
    --config_path ./config/base_config.yaml \
    --dataset_name gaia \
    --split test \
    --subset_num 32 \
    --enable_tool_search \
    --enable_thought_folding \
    --max_action_limit 30 \
    --eval
```

Helpful flags:

* `--single_question` – bypass datasets and answer a manually supplied prompt.
* `--top_k` – limit the number of suggested tools returned by tool search.
* `--concurrent_limit` – rate-limit async LLM completions when evaluating large splits.
* `--max_fold_limit` – cap the number of "thought folding" self-reflection passes per example.
* `--seed` – ensure reproducibility when you need to compare fairness metrics.

#### Tool search retriever service

If you want to pre-warm the retriever or expose it remotely:

```bash
python src/run_tool_search_server.py --config_path ./config/base_config.yaml --port 9000
```

`ToolManager` will connect to that endpoint whenever `--enable_tool_search` is set in the main runner.

### 4. Evaluate results

When the `--eval` flag is provided, `run_deep_agent.py` stores model generations and then calls the matching
script from `src/evaluate/`. You can also run the evaluators manually:

```bash
python src/evaluate/evaluate_toolbench.py --prediction_path path/to/preds.json --split test
```

Each evaluator (ToolBench, API-Bank, RestBench, ALFWorld, WebShop) shares the helper logic in
`src/evaluate/evaluate_base.py` (answer extraction, dataset adapters, metric printing). For dataset-specific
scoring guidelines check the header comments inside every evaluator file.

---

## Accessibility documentation

The fairness notebook and reporting workflow for the Humane Intelligence Accessibility Bias Bounty is captured in
[`docs/ADVANCED_DATA_NOTEBOOK.md`](docs/ADVANCED_DATA_NOTEBOOK.md). It contains:

1. Baseline Valence API analysis of the alexithymic vs. neurotypical split.
2. Simulated AccessibleDeepAgent mitigation and comparative FNR calculation.
3. Reporting templates, trade-offs, and deployment recommendations you can drop into a Colab or paper.

Use that document whenever you need a narrative version of how the reasoning stack integrates with neuroinclusive
UX changes.

---

## Contributing & license

Issues and PRs are welcome—especially improvements to the fairness evaluation harness and additional datasets.
This repository is released under the [MIT License](LICENSE).
