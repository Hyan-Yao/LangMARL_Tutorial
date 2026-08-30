LangMARL Documentation
======================

`Paper <https://arxiv.org/abs/2604.00722>`_ |
`GitHub <https://github.com/DaRL-GenAI/LangMARL>`_

**LangMARL** brings multi-agent credit assignment and policy gradient optimization
from classical MARL into natural language space. It optimizes multi-agent LLM systems
under the **Centralized Training with Decentralized Execution (CTDE)** paradigm.

.. code-block:: python

   import langmarl

   langmarl.train("configs/language_task/qa_central_credit.json")

The Core Idea
-------------

LangMARL keeps the structure of classical MARL, but replaces every numeric object
with a language object:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Classical MARL
     - LangMARL
     - Meaning
   * - Policy (numeric weights)
     - Natural language instruction
     - An agent's policy *is* its system prompt
   * - Credit / advantage
     - Trajectory-level language analysis
     - A centralized critic explains *what each agent did wrong*
   * - Numeric gradient
     - Language gradient
     - A concrete improvement instruction, applied to the policy text

Because the policy is text, optimization is interpretable: you can read every
gradient and every policy revision produced during training.

How Training Works
------------------

Training runs a Monte Carlo loop over four components:

1. **LLM Actors** -- each agent acts from its own language policy and its own local
   observation. This is the *decentralized execution* half of CTDE.
2. **Centralized Critic** -- training only. Sees the full episode trajectory and
   assigns credit per agent (LLM-as-judge).
3. **Policy Gradient Optimizer** -- turns credit into language gradients and applies
   them to the policies.
4. **Monte Carlo Trainer** -- collects trajectories, evaluates, aggregates gradients,
   updates policies, checkpoints, repeats.

Training Paradigms
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Paradigm
     - Description
   * - ``central_global``
     - The critic evaluates overall team performance; all agents receive the same
       shared gradient.
   * - ``central_credit``
     - The critic evaluates each agent's individual contribution; each agent receives
       its own targeted gradient.

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   training
   environments
   api

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

.. note::

   This project is under active development. Source code and issue tracker:
   https://github.com/DaRL-GenAI/LangMARL
