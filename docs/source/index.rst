Welcome to LangMARL's documentation!
=====================================

**LangMARL** is a language-native multi-agent reinforcement learning (MARL) framework
that reformulates policy representation, credit assignment, and policy optimization
entirely in natural language. It enables large language model (LLM)-based agents to learn
cooperative behaviors under the **Centralized Training and Decentralized Execution (CTDE)**
paradigm with improved interpretability, sample efficiency, and scalability.

Core Idea
---------

Traditional MARL relies on numeric parameters, scalar rewards, and gradient-based optimization.
LangMARL introduces a paradigm shift by treating **natural language as a first-class optimization space**:

* **Policies are Language**: Each agent's policy is represented as natural language rules or instructions.
* **Credits are Language**: A centralized critic assigns agent-level credit using trajectory-level language analysis.
* **Optimization is Language Evolution**: Policies are updated via language critiques instead of numeric gradients.

Framework Overview
------------------

1. **LLM Actors with Language Policies** -- Each agent is instantiated as an LLM. The policy
   is encoded as natural language (rules, heuristics, preferences). Input: textual local
   observation. Output: action description mapped to environment actions.

2. **Centralized Language Critic** -- Used only during training with access to the full
   episode trajectory. Produces agent-specific, causal, and interpretable language credits.

3. **Cross-Trajectory Credit Aggregation** -- A summary LLM aggregates language credits from
   multiple Monte Carlo rollouts, replacing numeric expectation with semantic abstraction.

4. **Language Policy Optimizer** -- Converts credit signals into policy critiques and updates
   policies via language revision (editing rules, adding constraints, refining priorities).

Training Paradigms
------------------

LangMARL supports three CTDE paradigms:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Paradigm
     - Description
   * - ``independent``
     - Each piston has a separate critic evaluating its individual performance.
   * - ``central_global``
     - A shared critic evaluates overall team performance with global reward.
   * - ``central_credit``
     - A shared critic evaluates each agent's individual contribution to team success.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
