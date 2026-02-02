Usage
=====

.. _installation:

Installation
------------

Install the required dependencies:

.. code-block:: console

   $ pip install pettingzoo[butterfly] supersuit imageio numpy openai

Install the TextGrad package (included in ``src/textgrad``):

.. code-block:: console

   $ cd src && pip install -e .

Set up your LLM API key as an environment variable. For OpenAI:

.. code-block:: console

   $ export OPENAI_API_KEY="your-api-key"

For other providers, set the corresponding key (e.g. ``GOOGLE_API_KEY``,
``TOGETHER_API_KEY``, ``DEEPSEEK_API_KEY``).

Quick Start
-----------

Running a single episode
~~~~~~~~~~~~~~~~~~~~~~~~

Generate a single episode with a rule-based agent and save a GIF:

.. code-block:: python

   from training.episode_generator import EpisodeGenerator

   generator = EpisodeGenerator(
       num_pistons=20,
       max_cycles=125,
       action_mode="discrete",
       agent_type="rule",
   )

   reward, steps, gif_path = generator.render_single_episode(
       save_path="./my_episode.gif",
       fps=30,
   )
   print(f"Reward: {reward}, Steps: {steps}, GIF: {gif_path}")

Generating multiple episodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   episodes = generator.generate_episodes(
       num_episodes=5,
       iteration=0,
       save_gif=True,
       gif_fps=30,
   )

   for ep in episodes:
       print(f"Episode {ep['episode_id']}: reward={ep['total_reward']:.2f}")

Training with TextGrad
~~~~~~~~~~~~~~~~~~~~~~

Use the ``MonteCarloTrainer`` to train multi-agent policies with language-based
optimization:

.. code-block:: python

   from training import TrainingConfig, MonteCarloTrainer

   config = TrainingConfig(
       exp_name="my_experiment",
       paradigm="central_credit",   # "independent" | "central_global" | "central_credit"
       policy_path="./policies/policy.json",
       experiments_dir="./experiments",
       num_iterations=5,
       trajectories_per_iteration=3,
       model="gpt-4o-mini",
       num_pistons=10,
   )

   trainer = MonteCarloTrainer(config)
   all_stats = trainer.train_full()

   for i, stats in enumerate(all_stats):
       print(f"Iteration {i}: avg_reward={stats['avg_reward']:.2f}")

Command-line training
~~~~~~~~~~~~~~~~~~~~~

The ``train_pistonball.py`` script provides a CLI interface:

.. code-block:: console

   # Basic training with central global paradigm
   $ python src/train_pistonball.py --paradigm central_global --num_iterations 5

   # Use a specific model
   $ python src/train_pistonball.py --model gpt-4o --paradigm central_credit

   # Use a different provider (e.g. Gemini)
   $ python src/train_pistonball.py --model gemini-pro

   # Resume training from a specific iteration
   $ python src/train_pistonball.py --config experiments/my_exp/config.json --start_iteration 3

   # List all available models
   $ python src/train_pistonball.py --list_models

   # Use rule-based agents (no LLM required)
   $ python src/train_pistonball.py --agent_type rule --num_pistons 20

Supported Models
----------------

All providers use the OpenAI-compatible API format. Available predefined models:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25

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
   * - ``gemini-pro``
     - gemini-1.5-pro
     - Google (via OpenAI-compatible endpoint)
   * - ``gemini-flash``
     - gemini-1.5-flash
     - Google
   * - ``gemini-2.0-flash``
     - gemini-2.0-flash
     - Google
   * - ``llama-3.1-70b``
     - meta-llama/llama-3.1-70b-instruct
     - Together
   * - ``llama-3.3-70b``
     - meta-llama/Llama-3.3-70B-Instruct-Turbo
     - Together
   * - ``qwen-72b``
     - Qwen/Qwen2.5-72B-Instruct-Turbo
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

You can also create a custom ``LLMConfig`` JSON file for any OpenAI-compatible endpoint:

.. code-block:: json

   {
       "name": "my-custom-model",
       "model_string": "my-model-id",
       "base_url": "http://localhost:8000/v1",
       "api_key": "my-key",
       "max_tokens": 4096,
       "input_price_per_million": 0.0,
       "output_price_per_million": 0.0
   }

Then pass it via:

.. code-block:: console

   $ python src/train_pistonball.py --llm_config path/to/my_llm.json

Action Space
------------

The Pistonball environment supports two action modes:

**Discrete** (3 actions):

- ``0``: retract_down -- Move piston down / retract
- ``1``: stay -- Hold current position
- ``2``: push_up -- Move piston up to push the ball left

**Continuous** (``[-1.0, 1.0]``):

- ``+1.0``: Maximum push up
- ``0.0``: Stay in place
- ``-1.0``: Maximum retract down

Observation Space (CTDE Architecture)
--------------------------------------

**Actor (Local Observation -- Decentralized Execution)**

Each agent observes only local information: its own vertical height, neighbor heights,
and the ball's relative position if visible within its local field of view.

**Critic (Global Observation -- Centralized Training)**

The critic has access to the full global state including ball position, ball velocity,
and all piston positions. This information is used during training for credit assignment
and evaluation, but not during decentralized execution.

Output File Structure
---------------------

Training runs produce the following directory structure:

.. code-block:: text

   experiments/
   +-- exp_name/
       +-- config.json
       +-- iteration_0/
       |   +-- episode_0/
       |   |   +-- episode_0.gif
       |   |   +-- episode_log.txt
       |   |   +-- episode.json
       |   |   +-- trajectory.pkl
       |   +-- episode_1/
       |   +-- textgrad_evaluations.txt
       +-- iteration_1/
       +-- ...
   policies/
   +-- policy.json
