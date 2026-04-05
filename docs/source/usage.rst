Usage
=====

.. _installation:

Installation
------------

Install LangMARL:

.. code-block:: console

   $ pip install langmarl

For environment-specific dependencies:

.. code-block:: console

   $ pip install langmarl[pettingzoo]   # Pistonball / PettingZoo support
   $ pip install langmarl[all]          # All optional dependencies

Set up your LLM API key as an environment variable:

.. code-block:: console

   $ export OPENAI_API_KEY="your-api-key"

For other providers, set the corresponding key:

.. code-block:: console

   $ export GOOGLE_API_KEY="your-key"     # Google Gemini
   $ export TOGETHER_API_KEY="your-key"   # Together (Llama, Qwen)
   $ export DEEPSEEK_API_KEY="your-key"   # DeepSeek

Quick Start
-----------

One-line training
~~~~~~~~~~~~~~~~~

Train from a JSON config file with a single call:

.. code-block:: python

   import langmarl

   langmarl.train("configs/language_task/qa_central_credit.json")

Programmatic usage
~~~~~~~~~~~~~~~~~~

For full control over the training pipeline:

.. code-block:: python

   import langmarl

   # 1. Configure
   config = langmarl.LanguageTaskConfig(
       task_type="qa",
       paradigm="central_credit",
       llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
       num_agents=2,
       num_iterations=5,
       trajectories_per_iteration=10,
   )

   # 2. Create components
   env = langmarl.make_env("language", config)
   critic = langmarl.CentralizedCritic(config)
   optimizer = langmarl.PolicyGradientOptimizer(config.get_optimizer_llm())

   # 3. Train
   trainer = langmarl.MonteCarloTrainer(
       config=config,
       env=env,
       critic=critic,
       optimizer=optimizer,
   )
   metrics = trainer.train()

Custom environment
~~~~~~~~~~~~~~~~~~

Register a custom environment by subclassing ``BaseEnvironment``:

.. code-block:: python

   import langmarl

   @langmarl.register_env("my_env")
   class MyEnv(langmarl.BaseEnvironment):
       def __init__(self, config):
           self.num_agents = config.num_agents
           self.llm_client = langmarl.LLMClient(config.get_actor_llm())

       def reset(self, task: dict) -> dict:
           return {"task": task}

       def step(self, agent_id: str, action: str):
           return {}, 0.0, False, {}

       def sample_tasks(self, num_samples: int) -> list[dict]:
           return [{"question": "What is 2+2?", "ground_truth": "4"}] * num_samples

       def collect_trajectory(self, policies, task) -> langmarl.Trajectory:
           steps = []
           for i, (agent, policy) in enumerate(policies.items()):
               response = self.llm_client.chat(policy, task["question"])
               steps.append({"agent_id": agent, "input": task["question"], "output": response})
           reward = 1.0 if task["ground_truth"] in steps[-1]["output"] else 0.0
           return langmarl.Trajectory(task=task, steps=steps, reward=reward)

   # Use the custom environment
   config = langmarl.BaseConfig(
       paradigm="central_credit",
       llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
   )
   env = langmarl.make_env("my_env", config)

Using callbacks
~~~~~~~~~~~~~~~

Add callbacks to hook into the training loop:

.. code-block:: python

   trainer = langmarl.MonteCarloTrainer(
       config=config,
       env=env,
       critic=critic,
       optimizer=optimizer,
       callbacks=[
           langmarl.LoggingCallback(),
           langmarl.EarlyStoppingCallback(patience=3, min_delta=0.01),
       ],
   )
   trainer.train()

Using different LLMs per role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assign different models to actors, critics, and optimizers:

.. code-block:: python

   config = langmarl.LanguageTaskConfig(
       task_type="qa",
       paradigm="central_credit",
       actor_llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),    # cheap for actors
       critic_llm=langmarl.LLMConfig.from_preset("gpt-4o"),         # strong for critic
       optimizer_llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"), # cheap for optimizer
       num_agents=2,
       num_iterations=5,
   )

If only ``llm`` is set, it is used as the fallback for all three roles.

Configuration
-------------

JSON config files
~~~~~~~~~~~~~~~~~

Training runs can be configured via JSON files:

.. code-block:: json

   {
       "exp_name": "qa_central_credit",
       "paradigm": "central_credit",
       "task_type": "qa",
       "benchmark_path": "env/lang_benchmark/HotPotQA",
       "num_agents": 2,
       "num_iterations": 5,
       "trajectories_per_iteration": 10,
       "llm": "gpt-4o-mini",
       "log_level": "INFO"
   }

The ``llm`` field accepts either a predefined model name (string) or a full LLM config object:

.. code-block:: json

   {
       "llm": {
           "name": "my-model",
           "model_string": "Qwen/Qwen2.5-72B-Instruct",
           "base_url": "https://api.together.xyz/v1",
           "api_key_env_var": "TOGETHER_API_KEY"
       }
   }

Supported Models
----------------

All providers use the OpenAI-compatible API format. Predefined models:

.. list-table::
   :header-rows: 1
   :widths: 20 35 20

   * - Name
     - Model String
     - Provider
   * - ``gpt-4o``
     - gpt-4o
     - OpenAI
   * - ``gpt-4o-mini``
     - gpt-4o-mini
     - OpenAI
   * - ``gpt-5``
     - gpt-5
     - OpenAI
   * - ``o1``
     - o1
     - OpenAI
   * - ``o1-mini``
     - o1-mini
     - OpenAI
   * - ``gemini-pro``
     - gemini-1.5-pro
     - Google
   * - ``gemini-flash``
     - gemini-1.5-flash
     - Google
   * - ``gemini-2.0-flash``
     - gemini-2.0-flash
     - Google
   * - ``llama-3.1-70b``
     - meta-llama/llama-3.1-70b-instruct
     - Together
   * - ``llama-3.1-8b``
     - meta-llama/llama-3.1-8b-instruct
     - Together
   * - ``llama-3.3-70b``
     - meta-llama/Llama-3.3-70B-Instruct-Turbo
     - Together
   * - ``qwen-72b``
     - Qwen/Qwen2.5-72B-Instruct-Turbo
     - Together
   * - ``qwen-7b``
     - Qwen/Qwen2.5-7B-Instruct-Turbo
     - Together
   * - ``qwen-coder-32b``
     - Qwen/Qwen2.5-Coder-32B-Instruct
     - Together
   * - ``deepseek-chat``
     - deepseek-chat
     - DeepSeek
   * - ``deepseek-reasoner``
     - deepseek-reasoner
     - DeepSeek
   * - ``ollama-llama3``
     - llama3
     - Local Ollama
   * - ``ollama-qwen2``
     - qwen2
     - Local Ollama

You can also create a custom ``LLMConfig`` for any OpenAI-compatible endpoint:

.. code-block:: python

   custom_llm = langmarl.LLMConfig(
       name="my-custom-model",
       model_string="my-model-id",
       base_url="http://localhost:8000/v1",
       api_key="my-key",
       max_tokens=4096,
       input_price_per_million=0.0,
       output_price_per_million=0.0,
   )

Output Structure
----------------

Training runs produce the following directory structure:

.. code-block:: text

   experiments/
   +-- {exp_name}_{timestamp}/
       +-- config.json
       +-- run.log
       +-- metrics.json
       +-- trajectories/
       |   +-- iter_0/
       |   |   +-- episode_0.json
       |   |   +-- episode_1.json
       |   +-- iter_1/
       +-- checkpoints/
       |   +-- iter_0/
       |   |   +-- agent_1.txt
       |   |   +-- agent_2.txt
       |   |   +-- metadata.json
       |   +-- iter_1/
       +-- gradients/
       |   +-- iter_0/
       |       +-- agent_1_gradients.json
       |       +-- agent_1_aggregated.txt
       +-- evaluations/
           +-- iter_0/
               +-- episode_0_eval.txt
