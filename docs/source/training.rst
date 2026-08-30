Training
========

Choosing a paradigm
-------------------

The ``paradigm`` field selects how the centralized critic distributes credit.

``central_global``
   The critic scores the episode as a whole. Every agent receives the same shared
   gradient. Cheaper, and a good baseline.

``central_credit``
   The critic attributes the outcome to individual agents and returns per-agent
   feedback, so each agent receives its own gradient. Use this when agents play
   different roles or when only some agents are responsible for a failure.

.. code-block:: python

   config = langmarl.LanguageTaskConfig(task_type="qa", paradigm="central_credit")

Configuration
-------------

Every environment config inherits the same training fields from ``BaseConfig``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``exp_name``
     - Experiment name; used for the output directory.
   * - ``paradigm``
     - ``"central_global"`` or ``"central_credit"``.
   * - ``num_agents``
     - Number of agents (default ``2``).
   * - ``num_iterations``
     - Training iterations (default ``5``).
   * - ``trajectories_per_iteration``
     - Episodes collected per iteration (default ``10``).
   * - ``mini_batch_size``
     - Subsample this many trajectories for the gradient step (default: use all).
   * - ``max_workers``
     - Parallelism for trajectory collection (default ``5``).
   * - ``experiment_dir`` / ``checkpoint_dir``
     - Where runs and policy checkpoints are written.

Configs are plain JSON and round-trip through ``from_json`` / ``to_json``:

.. code-block:: json

   {
       "exp_name": "qa_central_credit",
       "paradigm": "central_credit",
       "task_type": "qa",
       "benchmark_path": "env/lang_benchmark/HotPotQA",
       "num_agents": 2,
       "num_iterations": 5,
       "trajectories_per_iteration": 10,
       "llm": "gpt-4o-mini"
   }

.. code-block:: python

   config = langmarl.load_config("configs/qa.json", {"num_iterations": 10})

Choosing models
---------------

All providers -- OpenAI, Google, Together, DeepSeek, and local Ollama -- are reached
through the same OpenAI-compatible client, so switching models is a one-line change.
``LLMConfig.from_preset`` covers 18+ predefined models; ``langmarl.list_available_models()``
prints the current list.

Actors, the critic, and the optimizer can each use a different model. This is the
main cost lever: actors run on every step of every episode, while the critic runs
once per episode.

.. code-block:: python

   config = langmarl.LanguageTaskConfig(
       task_type="qa",
       paradigm="central_credit",
       actor_llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),      # cheap, high volume
       critic_llm=langmarl.LLMConfig.from_preset("gpt-4o"),          # strong judge
       optimizer_llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
   )

Setting ``llm`` alone makes it the fallback for all three roles. Any OpenAI-compatible
endpoint works without a preset:

.. code-block:: python

   custom = langmarl.LLMConfig(
       name="my-model",
       model_string="Qwen/Qwen2.5-72B-Instruct",
       base_url="https://api.together.xyz/v1",
       api_key_env_var="TOGETHER_API_KEY",
   )

The same object can be written inline in a JSON config in place of the model-name string.

Resuming a run
--------------

``trainer.train()`` detects the latest checkpoint under ``checkpoint_dir`` and continues
from the next iteration, so an interrupted run needs no special handling -- rerun the
same command. Use ``start_iteration`` to force a specific starting point, and
``PolicyCheckpoint.get_policies(iteration)`` to load the policies from any past iteration.

Token and cost tracking
-----------------------

Usage and USD cost are tracked per run from built-in per-model pricing, reported at the
end of each iteration alongside rewards, and persisted to ``metrics.json``:

.. code-block:: python

   stats = trainer.train_one_iteration(0)
   print(stats["avg_reward"], stats["total_tokens"], stats["cost_usd"])

Callbacks
---------

Callbacks hook into the training loop for logging, early stopping, or your own logic:

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

Subclass ``langmarl.Callback`` and override ``on_iteration_start``, ``on_iteration_end``,
``on_episode_complete``, or ``on_policy_update``.
