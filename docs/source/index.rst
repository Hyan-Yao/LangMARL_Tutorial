LangMARL Documentation
======================

**LangMARL** is a language-space multi-agent reinforcement learning library that applies
credit assignment and policy gradient optimization from classical MARL into natural
language space. It enables principled autonomous optimization of multi-agent LLM-based
systems via **Centralized Training with Decentralized Execution (CTDE)**.

.. code-block:: python

   import langmarl

   langmarl.train("configs/language_task/qa_central_credit.json")

Core Concepts
-------------

LangMARL treats **natural language as a first-class optimization space**:

* **Policies are Language** -- Each agent's policy is a natural language instruction (system prompt),
  not a numeric parameter vector.
* **Credits are Language** -- A centralized critic assigns agent-level credit via trajectory-level
  language analysis, producing causal and interpretable feedback.
* **Optimization is Language Evolution** -- Policies are updated via language gradients
  (improvement instructions) instead of numeric gradients.

Architecture
------------

LangMARL implements the CTDE paradigm with four components:

1. **LLM Actors** -- Each agent is an LLM whose behavior is governed by a natural language
   policy. During execution, each agent observes only its local information and acts independently.

2. **Centralized Critic** -- Used only during training. Evaluates full episode trajectories
   and generates per-agent credit using LLM-as-judge.

3. **Policy Gradient Optimizer** -- Converts credit signals into language gradients
   (concrete improvement instructions) and applies them to agent policies.

4. **Monte Carlo Trainer** -- Orchestrates the training loop: collect trajectories,
   evaluate, generate gradients, aggregate, and update policies.

Training Paradigms
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Paradigm
     - Description
   * - ``central_global``
     - A shared critic evaluates overall team performance. All agents receive the same
       shared gradient signal.
   * - ``central_credit``
     - A shared critic evaluates each agent's individual contribution to team success.
       Each agent receives a targeted per-agent gradient.

Supported Environments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Environment
     - Agents
     - Description
   * - Language Tasks
     - 2+
     - Sequential collaboration on QA (HotPotQA), Math, Creative Writing, and Coding (HumanEval).
   * - Overcooked-AI
     - 2
     - Cooperative cooking with sparse team rewards and role differentiation.
   * - Pistonball
     - 10--20
     - Large-scale cooperative control with partial observability.

Custom environments can be registered via the ``@langmarl.register_env`` decorator.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
