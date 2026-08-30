API Reference
=============

.. module:: langmarl

This page covers the public API. Everything below is importable directly from
``langmarl``.

Entry points
------------

.. function:: train(config_path, **overrides)

   Load a JSON config, build the environment, critic, optimizer and trainer, and run
   training. Keyword arguments override config fields. Returns the training metrics.

.. function:: load_config(path, overrides=None)

   Load a config from JSON, returning ``BaseConfig`` or the matching subclass.

.. function:: make_env(name, config)

   Create a registered environment by name. Raises ``ValueError`` if unknown.

.. function:: register_env(name)

   Class decorator that registers a ``BaseEnvironment`` subclass under ``name``.

.. function:: list_envs()

   List all registered environment names.

.. function:: list_available_models()

   Map every predefined model name to its description.

Configuration
-------------

.. class:: BaseConfig

   Shared training configuration.

   **Fields:** ``exp_name``, ``paradigm`` (``"central_global"`` | ``"central_credit"``),
   ``num_agents``, ``num_iterations``, ``trajectories_per_iteration``, ``mini_batch_size``,
   ``start_iteration``, ``llm``, ``actor_llm``, ``critic_llm``, ``optimizer_llm``,
   ``experiment_dir``, ``checkpoint_dir``, ``max_workers``, ``log_level``.

   .. method:: get_actor_llm()
   .. method:: get_critic_llm()
   .. method:: get_optimizer_llm()

      Return the role's ``LLMConfig``, falling back to ``llm`` when unset.

   .. method:: from_json(path, overrides=None)
      :classmethod:

      Load from JSON. LLM fields accept a preset name or a full config dict.

   .. method:: to_json(path)

      Save the config to JSON.

.. class:: LanguageTaskConfig

   ``BaseConfig`` plus ``task_type`` (``"qa"`` | ``"math"`` | ``"writing"`` | ``"coding"``),
   ``benchmark_path``, ``data_limit``, ``use_verified_reward``,
   ``episode_generation_workers``, ``optimizer_workers``.

.. class:: OvercookedConfig

   ``BaseConfig`` plus ``layout``, ``episode_horizon``, ``p0_agent``, ``p1_agent``.

.. class:: PistonballConfig

   ``BaseConfig`` plus ``num_pistons``, ``max_cycles``, ``frame_size``, ``action_mode``.

.. class:: LLMConfig

   One model, described in OpenAI-compatible terms.

   **Fields:** ``name``, ``model_string``, ``base_url``, ``api_key``, ``api_key_env_var``,
   ``max_tokens``, ``is_multimodal``, ``input_price_per_million``,
   ``output_price_per_million``, ``extra_params``.

   .. method:: from_preset(name)
      :classmethod:

      Build a config from a predefined model name, e.g. ``"gpt-4o-mini"``.

   .. method:: from_dict(data)
      :classmethod:
   .. method:: to_dict()
   .. method:: get_api_key()

Core abstractions
-----------------

.. class:: Trajectory

   One episode, as a dataclass: ``task`` (dict), ``steps`` (list of per-agent step
   dicts), ``reward`` (float), ``metadata`` (dict).

.. class:: BaseEnvironment

   Base class for environments.

   .. method:: collect_trajectory(policies, task)
      :abstractmethod:

      Run a full episode under ``policies`` (agent name -> policy text) and return a
      :class:`Trajectory`.

   .. method:: reset(task)
      :abstractmethod:
   .. method:: step(agent_id, action)
      :abstractmethod:

      Returns ``(observation, reward, done, info)``.

.. class:: BaseCritic

   .. method:: evaluate(trajectory, policies)
      :abstractmethod:

      Return an evaluation dict with ``raw_response`` and ``per_agent`` credits.

.. class:: BaseOptimizer

   .. method:: generate_gradient(policy, evaluation, context)
      :abstractmethod:
   .. method:: aggregate_gradients(gradients)
      :abstractmethod:
   .. method:: apply_gradient(policy, gradient)
      :abstractmethod:

Implementations
---------------

.. class:: CentralizedCritic(config, prompts_dir=None)

   LLM-as-judge critic for both paradigms. ``evaluate(trajectory, policies)`` returns
   ``{"raw_response", "paradigm", "per_agent"}`` -- per-agent causal credit under
   ``central_credit``, one shared evaluation under ``central_global``.
   Pass ``prompts_dir`` to override the evaluation prompt templates.

.. class:: PolicyGradientOptimizer(llm_config)

   Turns critic feedback into language gradients.

   .. method:: generate_gradient(policy, evaluation, context, agent_name="agent")

      A few sentences of specific improvement advice for one agent.

   .. method:: generate_shared_gradient(evaluation, task_context)

      The ``central_global`` counterpart: one gradient for the whole team.

   .. staticmethod:: aggregate_gradients(gradients)

      Combine the gradients from an iteration's episodes into one.

   .. staticmethod:: apply_gradient(base_policy, gradient)

      Return the updated policy. The base policy is never mutated: the gradient is
      written into a ``[CASE-SPECIFIC FEEDBACK]`` section that is replaced each call.

.. class:: MonteCarloTrainer(config, env, critic, optimizer, reward_fn=None, store=None, callbacks=None)

   The training loop. Each iteration: load policies, collect trajectories, evaluate and
   generate gradients, aggregate and apply them, checkpoint.

   .. method:: train(num_iterations=None)

      Run training, resuming automatically from the latest checkpoint.

   .. method:: train_one_iteration(iteration)

      Run one iteration and return its stats: ``avg_reward``, ``min_reward``,
      ``max_reward``, ``rewards``, ``num_episodes``, ``paradigm``, ``input_tokens``,
      ``output_tokens``, ``total_tokens``, ``cost_usd``.

Callbacks
---------

.. class:: Callback

   Override ``on_iteration_start(iteration, trainer)``,
   ``on_iteration_end(iteration, stats, trainer)``,
   ``on_episode_complete(trajectory, trainer)``, or
   ``on_policy_update(agent_id, old_policy, new_policy)``.

.. class:: LoggingCallback

   Logs iteration start/end through the run logger.

.. class:: EarlyStoppingCallback(patience=3, min_delta=0.01)

   Stop once the average reward stops improving.

LLM client
----------

.. class:: LLMClient(llm_config)

   Unified client for any OpenAI-compatible endpoint.

   .. method:: chat(system_prompt, user_input, max_tokens=None)

      Return the response text.

   .. method:: chat_with_usage(system_prompt, user_input, max_tokens=None)

      Return ``(text, {"input": n, "output": n})``.

.. class:: TokenTracker(model="gpt-4o-mini", input_price=None, output_price=None)

   Accumulates token usage and estimates cost from built-in per-model pricing.
   ``add_usage``, ``get_stats``, ``estimate_cost``, ``get_summary_string``, ``reset``.

Storage
-------

.. class:: LocalStore(base_dir)

   Filesystem backend holding a run's trajectories, evaluations, gradients,
   checkpoints, metrics, and logs. Passed to the trainer as ``store``.

.. class:: PolicyCheckpoint(store, run_id, num_agents)

   Versioned policy snapshots, one text file per agent per iteration.

   .. method:: get_policies(iteration=None)

      Load the latest (or a specific) iteration's policies, generating defaults when
      no checkpoint exists.

   .. method:: save_policies(iteration, policies, stats=None)
   .. method:: diff_policies(iter_a, iter_b)

      Per-agent diff between two iterations -- the readable record of what training changed.

.. class:: TrajectoryStore(store, run_id)

   Episode persistence: ``save(iteration, episode_id, trajectory)``,
   ``load(iteration, limit=None)``, ``count(iteration)``.
