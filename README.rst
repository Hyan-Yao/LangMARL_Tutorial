LangMARL
========

**Language-space Multi-Agent Reinforcement Learning**

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://readthedocs.org/projects/langmarl/badge/?version=latest
   :target: https://langmarl.readthedocs.io/

.. image:: https://img.shields.io/badge/arXiv-2604.00722-b31b1b.svg
   :target: https://arxiv.org/abs/2604.00722

.. image:: https://img.shields.io/badge/GitHub-DaRL--GenAI%2FLangMARL-181717.svg?logo=github
   :target: https://github.com/DaRL-GenAI/LangMARL

`Paper <https://arxiv.org/abs/2604.00722>`_ |
`Documentation <https://langmarl.readthedocs.io/>`_ |
`GitHub <https://github.com/DaRL-GenAI/LangMARL>`_

LangMARL applies multi-agent credit assignment and policy gradient optimization
from classical MARL into natural language space. It enables principled autonomous
optimization of multi-agent LLM-based systems via **Centralized Training with
Decentralized Execution (CTDE)**.

Key Features
------------

- **Language Policies** -- Agent policies are natural language instructions, not numeric parameters
- **Centralized Credit Assignment** -- A centralized critic assigns per-agent credit using trajectory-level language analysis
- **Language Policy Optimization** -- Policies evolve via language gradients instead of numeric gradients
- **Multi-Provider LLM Support** -- 18+ predefined models across OpenAI, Google, Together, DeepSeek, and local Ollama
- **Plugin Environment System** -- Register custom environments via ``@register_env`` decorator
- **Resumable Training** -- Auto-detect checkpoints and resume from any iteration
- **Token & Cost Tracking** -- Built-in per-model pricing and usage statistics

Installation
------------

.. code-block:: console

   $ pip install langmarl

From source:

.. code-block:: console

   $ git clone https://github.com/DaRL-GenAI/LangMARL.git
   $ cd LangMARL
   $ pip install -e ".[all]"

Set up your API key:

.. code-block:: console

   $ export OPENAI_API_KEY="your-api-key"

Quick Start
-----------

One-line training from config:

.. code-block:: python

   import langmarl

   langmarl.train("configs/language_task/qa_central_credit.json")

Programmatic usage:

.. code-block:: python

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

Documentation
-------------

Full documentation: https://langmarl.readthedocs.io/

Source code and issues: https://github.com/DaRL-GenAI/LangMARL

Citation
--------

LangMARL is described in `Language-space Multi-Agent Reinforcement Learning
<https://arxiv.org/abs/2604.00722>`_. If you use it in your research, please cite:

.. code-block:: bibtex

   @article{langmarl2026,
     title   = {Language-space Multi-Agent Reinforcement Learning},
     author  = {LangMARL Authors},
     journal = {arXiv preprint arXiv:2604.00722},
     year    = {2026},
     url     = {https://arxiv.org/abs/2604.00722}
   }

License
-------

MIT
