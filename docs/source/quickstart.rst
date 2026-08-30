Quickstart
==========

Installation
------------

.. code-block:: console

   $ pip install langmarl

   $ pip install langmarl[pettingzoo]   # Pistonball / PettingZoo environments
   $ pip install langmarl[all]          # everything

Set the API key for whichever provider you use:

.. code-block:: console

   $ export OPENAI_API_KEY="your-api-key"

Train from a config file
------------------------

The shortest path to a training run is a JSON config plus one call. Any config field
can be overridden inline:

.. code-block:: python

   import langmarl

   langmarl.train("configs/language_task/qa_central_credit.json")
   langmarl.train("configs/qa.json", num_iterations=10, paradigm="central_global")

The same thing from the command line:

.. code-block:: console

   $ langmarl-train configs/language_task/qa_central_credit.json

Train programmatically
----------------------

Use the components directly when you want to swap in your own critic, optimizer,
or environment:

.. code-block:: python

   import langmarl

   config = langmarl.LanguageTaskConfig(
       task_type="qa",                 # qa | math | writing | coding
       paradigm="central_credit",
       llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
       num_agents=2,
       num_iterations=5,
       trajectories_per_iteration=10,
   )

   trainer = langmarl.MonteCarloTrainer(
       config=config,
       env=langmarl.make_env("language", config),
       critic=langmarl.CentralizedCritic(config),
       optimizer=langmarl.PolicyGradientOptimizer(config.get_optimizer_llm()),
   )
   metrics = trainer.train()

What a run produces
-------------------

Every run writes a self-contained, human-readable directory. The policies are plain
text files -- you can read exactly how each agent's prompt evolved:

.. code-block:: text

   experiments/{exp_name}_{timestamp}/
   |-- config.json          # the exact config used
   |-- run.log
   |-- metrics.json         # per-iteration rewards, tokens, cost
   |-- trajectories/        # every episode, per iteration
   |-- evaluations/         # critic output
   |-- gradients/           # language gradients, per agent
   `-- checkpoints/         # policy text per agent, per iteration

Next steps
----------

* :doc:`training` -- paradigms, configuration, resuming, cost tracking
* :doc:`environments` -- built-in environments and registering your own
