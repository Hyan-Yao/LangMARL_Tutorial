# LangMARL

**Language-space Multi-Agent Reinforcement Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/langmarl/badge/?version=latest)](https://langmarl.readthedocs.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00722-b31b1b.svg)](https://arxiv.org/abs/2604.00722)
[![GitHub](https://img.shields.io/badge/GitHub-DaRL--GenAI%2FLangMARL-181717.svg?logo=github)](https://github.com/DaRL-GenAI/LangMARL)

[**Paper**](https://arxiv.org/abs/2604.00722) | [**Documentation**](https://langmarl.readthedocs.io/) | [**GitHub**](https://github.com/DaRL-GenAI/LangMARL)

LangMARL applies multi-agent credit assignment and policy gradient optimization from classical MARL into natural language space. It enables principled autonomous optimization of multi-agent LLM-based systems via **Centralized Training with Decentralized Execution (CTDE)**.

---

## Key Features

- **Language Policies** -- Agent policies are natural language instructions, not numeric parameters
- **Centralized Credit Assignment** -- A centralized critic assigns per-agent credit using trajectory-level language analysis
- **Language Policy Optimization** -- Policies evolve via language gradients instead of numeric gradients
- **Multi-Provider LLM Support** -- 18+ predefined models across OpenAI, Google, Together, DeepSeek, and local Ollama
- **Plugin Environment System** -- Register custom environments via `@register_env` decorator
- **Resumable Training** -- Auto-detect checkpoints and resume from any iteration
- **Token & Cost Tracking** -- Built-in per-model pricing and usage statistics

## Installation

```bash
pip install langmarl
```

For environment-specific dependencies:

```bash
# Pistonball / PettingZoo environments
pip install langmarl[pettingzoo]

# All optional dependencies
pip install langmarl[all]
```

From source:

```bash
git clone https://github.com/DaRL-GenAI/LangMARL.git
cd LangMARL
pip install -e ".[all]"
```

Set up your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

**One-line training from config:**

```python
import langmarl

langmarl.train("configs/language_task/qa_central_credit.json")
```

**Programmatic usage:**

```python
import langmarl

config = langmarl.LanguageTaskConfig(
    task_type="qa",
    paradigm="central_credit",
    llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
)

env = langmarl.make_env("language", config)
trainer = langmarl.MonteCarloTrainer(
    config=config,
    env=env,
    critic=langmarl.CentralizedCritic(config),
    optimizer=langmarl.PolicyGradientOptimizer(config.get_optimizer_llm()),
)
trainer.train()
```

## Training Paradigms

| Paradigm | Description |
|---|---|
| `central_global` | A shared critic evaluates overall team performance; all agents receive the same gradient |
| `central_credit` | A shared critic evaluates each agent's individual contribution; per-agent gradients |

## Supported Environments

| Environment | Type | Agents |
|---|---|---|
| Language Tasks (QA, Math, Writing, Coding) | Sequential collaboration | 2+ |
| Overcooked-AI | Cooperative cooking | 2 |
| Pistonball | Large-scale cooperative control | 10-20 |

## Documentation

Full documentation is available at [langmarl.readthedocs.io](https://langmarl.readthedocs.io/).

Source code, issues, and contributions: [github.com/DaRL-GenAI/LangMARL](https://github.com/DaRL-GenAI/LangMARL).

## Citation

LangMARL is described in [*Language-space Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/2604.00722).
If you use it in your research, please cite:

```bibtex
@article{langmarl2026,
  title   = {Language-space Multi-Agent Reinforcement Learning},
  author  = {LangMARL Authors},
  journal = {arXiv preprint arXiv:2604.00722},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.00722}
}
```

## License

MIT
