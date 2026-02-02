# LangMARL: Natural Language Multi-Agent Reinforcement Learning

LangMARL is a **language-native multi-agent reinforcement learning (MARL) framework** that reformulates **policy representation, credit assignment, and policy optimization entirely in natural language**.
It enables large language model (LLM)–based agents to learn cooperative behaviors under the **Centralized Training and Decentralized Execution (CTDE)** paradigm with improved interpretability, sample efficiency, and scalability.

---

## ✨ Core Idea

Traditional MARL relies on numeric parameters, scalar rewards, and gradient-based optimization. LangMARL introduces a paradigm shift by treating **natural language as a first-class optimization space**:

* **Policies are Language**: Each agent’s policy is represented as natural language rules or instructions
* **Credits are Language**: A centralized critic assigns agent-level credit using trajectory-level language analysis
* **Optimization is Language Evolution**: Policies are updated via language critiques instead of numeric gradients

> Natural language is not just a communication or explanation layer—it is the medium for learning itself.

---

## 🧩 Framework Overview

### 1. LLM Actors with Language Policies

* Each agent is instantiated as an LLM
* The policy is encoded as natural language (rules, heuristics, preferences)
* Input: textual local observation
* Output: action description (mapped to environment actions)

### 2. Centralized Language Critic

* Used only during training
* Has access to the full episode trajectory
* Produces **agent-specific, causal, and interpretable language credits**
* Explicitly explains how each agent’s behavior contributed to success or failure

### 3. Cross-Trajectory Credit Aggregation

* A summary LLM aggregates language credits from multiple Monte Carlo rollouts
* Replaces numeric expectation with semantic abstraction
* Produces stable, high-level credit signals per agent

### 4. Language Policy Optimizer

* Converts credit signals into policy critiques
* Updates policies via language revision (editing rules, adding constraints, refining priorities)

---

## 🔁 Algorithm: LangMARL (CTDE)

1. **Decentralized Execution & Experience Collection**

   * Each agent independently acts according to its own language policy
   * Full episode trajectories are collected

2. **Centralized Language-Based Credit Assignment**

   * A centralized critic analyzes full trajectories
   * Generates natural language credit for each agent

3. **Cross-Trajectory Credit Summarization**

   * A summary LLM abstracts consistent behavioral patterns across rollouts

4. **Language-Level Policy Optimization**

   * Credits are transformed into critiques
   * Each agent’s language policy is updated accordingly

At test time, **all agents execute fully decentralized**, without access to centralized components.

---

## 🧪 Experimental Environments

### Overcooked-AI

* Two-agent cooperative environment
* Sparse team rewards
* Requires long-horizon coordination and role differentiation
* Hierarchical action space (semantic subgoals + low-level controls)

### Pistonball

* Large-scale cooperative control (10–20 agents)
* Severe partial observability
* Shared global reward with complex local causality
* Emphasizes scalability and credit assignment under many agents

---

## 📊 Results Summary

### Baselines

* **Zero-shot LLM**: No learning or adaptation
* **TextGrad**: Treats language feedback as gradient-like signals, without explicit multi-agent credit modeling

### Key Findings

* LangMARL consistently outperforms baselines in:

  * Final performance
  * Sample efficiency
  * Training stability
  * Scalability with increasing number of agents
* Performance gains are especially large in:

  * Sparse-reward settings
  * Asymmetric-role tasks
  * Large-team coordination scenarios

---

## 🔍 Ablation and Analysis

* **Removing agent-wise credit assignment leads to significant performance drops**
* LangMARL is robust across different LLM backbones
* Rollout count affects stability but exhibits non-monotonic behavior
* Improvements stem primarily from **language-based credit assignment**, not raw model scale

---

## 📌 Advantages of LangMARL

* ✅ Interpretable, agent-level learning signals
* ✅ Causal, trajectory-level credit assignment
* ✅ No reliance on numeric gradients or model fine-tuning
* ✅ Compatible with black-box LLMs
* ✅ Effective for long-horizon, sparse-reward, and large-scale cooperative tasks
