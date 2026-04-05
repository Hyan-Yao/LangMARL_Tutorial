API Reference
=============

.. module:: langmarl

Convenience Functions
---------------------

.. function:: train(config_path, **overrides)

   One-line training: load config, create components, run training.

   :param config_path: Path to a JSON config file.
   :type config_path: str
   :param overrides: Key=value overrides applied to the loaded config.
   :returns: Training metrics.

   .. code-block:: python

      langmarl.train("configs/language_task/qa_central_credit.json")
      langmarl.train("configs/qa.json", num_iterations=10, paradigm="central_global")

.. function:: load_config(path, overrides=None)

   Load a configuration from a JSON file.

   :param path: Path to JSON config file.
   :type path: str
   :param overrides: Optional dictionary of overrides.
   :type overrides: dict or None
   :returns: ``BaseConfig`` or subclass instance (``LanguageTaskConfig``, etc.).

.. function:: make_env(name, config)

   Create an environment instance by registered name.

   :param name: Environment name (e.g., ``"language"``, ``"overcooked"``, ``"pistonball"``).
   :type name: str
   :param config: Configuration object.
   :type config: BaseConfig
   :returns: Environment instance.
   :rtype: BaseEnvironment
   :raises ValueError: If the environment name is not registered.

.. function:: register_env(name)

   Decorator to register a custom environment class.

   :param name: Name to register the environment under.
   :type name: str

   .. code-block:: python

      @langmarl.register_env("my_env")
      class MyEnv(langmarl.BaseEnvironment):
          ...

.. function:: list_envs()

   List all registered environment names.

   :returns: List of environment name strings.
   :rtype: list[str]

Configuration
-------------

BaseConfig
~~~~~~~~~~

.. class:: BaseConfig

   Shared configuration fields across all environments.

   :param exp_name: Experiment name (default: ``"experiment"``).
   :type exp_name: str
   :param paradigm: Training paradigm. One of ``"central_global"``, ``"central_credit"`` (default: ``"central_credit"``).
   :type paradigm: str
   :param num_iterations: Number of training iterations (default: ``5``).
   :type num_iterations: int
   :param trajectories_per_iteration: Episodes per iteration (default: ``10``).
   :type trajectories_per_iteration: int
   :param mini_batch_size: If set, subsample this many trajectories for gradient computation (default: ``None`` = use all).
   :type mini_batch_size: int or None
   :param start_iteration: Iteration to start from (default: ``0``).
   :type start_iteration: int
   :param llm: Default LLM config, used as fallback for actor/critic/optimizer (default: ``None``).
   :type llm: LLMConfig or None
   :param actor_llm: LLM config for actors. Falls back to ``llm`` if not set.
   :type actor_llm: LLMConfig or None
   :param critic_llm: LLM config for the critic. Falls back to ``llm`` if not set.
   :type critic_llm: LLMConfig or None
   :param optimizer_llm: LLM config for the optimizer. Falls back to ``llm`` if not set.
   :type optimizer_llm: LLMConfig or None
   :param num_agents: Number of agents (default: ``2``).
   :type num_agents: int
   :param experiment_dir: Directory to save experiment data (default: ``"./experiments"``).
   :type experiment_dir: str
   :param checkpoint_dir: Directory to save policy checkpoints (default: ``"./ckpt_policy"``).
   :type checkpoint_dir: str
   :param max_workers: Maximum parallel workers for trajectory generation (default: ``5``).
   :type max_workers: int
   :param log_level: Logging level (default: ``"INFO"``).
   :type log_level: str

   .. method:: get_actor_llm()

      Get LLM config for actors. Returns ``actor_llm`` if set, otherwise ``llm``.

      :rtype: LLMConfig

   .. method:: get_critic_llm()

      Get LLM config for the critic. Returns ``critic_llm`` if set, otherwise ``llm``.

      :rtype: LLMConfig

   .. method:: get_optimizer_llm()

      Get LLM config for the optimizer. Returns ``optimizer_llm`` if set, otherwise ``llm``.

      :rtype: LLMConfig

   .. method:: from_json(path, overrides=None)
      :classmethod:

      Load config from a JSON file with optional overrides. LLM fields can be
      specified as predefined model name strings or full config dicts.

      :param path: Path to JSON config file.
      :type path: str
      :param overrides: Optional dictionary of overrides.
      :type overrides: dict or None
      :returns: Config instance.

   .. method:: to_json(path)

      Save config to a JSON file.

      :param path: Output file path.
      :type path: str

LanguageTaskConfig
~~~~~~~~~~~~~~~~~~

.. class:: LanguageTaskConfig

   Configuration for language task environments. Inherits all fields from :class:`BaseConfig`.

   :param task_type: Task type. One of ``"qa"``, ``"math"``, ``"writing"``, ``"coding"`` (default: ``"qa"``).
   :type task_type: str
   :param benchmark_path: Path to the benchmark dataset directory.
   :type benchmark_path: str
   :param data_limit: Limit the number of tasks loaded from the benchmark (default: ``None`` = all).
   :type data_limit: int or None
   :param use_verified_reward: Enable LLM-as-judge for reward verification (default: ``False``).
   :type use_verified_reward: bool
   :param episode_generation_workers: Parallel workers for episode generation (default: ``8``).
   :type episode_generation_workers: int
   :param optimizer_workers: Parallel workers for gradient generation (default: ``1``).
   :type optimizer_workers: int

OvercookedConfig
~~~~~~~~~~~~~~~~

.. class:: OvercookedConfig

   Configuration for the Overcooked environment. Inherits all fields from :class:`BaseConfig`.

   :param layout: Kitchen layout name (default: ``"cramped_room"``).
   :type layout: str
   :param episode_horizon: Maximum timesteps per episode (default: ``400``).
   :type episode_horizon: int
   :param p0_agent: Agent 0 type (default: ``"ProAgent"``).
   :type p0_agent: str
   :param p1_agent: Agent 1 type (default: ``"ProAgent"``).
   :type p1_agent: str

PistonballConfig
~~~~~~~~~~~~~~~~

.. class:: PistonballConfig

   Configuration for the Pistonball environment. Inherits all fields from :class:`BaseConfig`.

   :param num_pistons: Number of pistons (default: ``20``).
   :type num_pistons: int
   :param max_cycles: Maximum cycles per episode (default: ``125``).
   :type max_cycles: int
   :param frame_size: Observation frame size (default: ``84``).
   :type frame_size: int
   :param action_mode: ``"discrete"`` or ``"continuous"`` (default: ``"discrete"``).
   :type action_mode: str

LLMConfig
~~~~~~~~~

.. class:: LLMConfig

   Configuration for a single LLM model. All providers use the OpenAI-compatible API format.

   :param name: Human-readable model identifier (e.g., ``"gpt-4o"``).
   :type name: str
   :param model_string: Model string passed to the API (e.g., ``"gpt-4o"``).
   :type model_string: str
   :param base_url: Custom API base URL. ``None`` means default OpenAI endpoint.
   :type base_url: str or None
   :param api_key: API key. ``None`` means use environment variable.
   :type api_key: str or None
   :param api_key_env_var: Environment variable name for the API key (default: ``"OPENAI_API_KEY"``).
   :type api_key_env_var: str
   :param is_multimodal: Whether the model supports multimodal input (default: ``False``).
   :type is_multimodal: bool
   :param max_tokens: Maximum tokens per response (default: ``4096``).
   :type max_tokens: int
   :param input_price_per_million: Input token price per million (USD).
   :type input_price_per_million: float
   :param output_price_per_million: Output token price per million (USD).
   :type output_price_per_million: float
   :param extra_params: Additional provider-specific parameters.
   :type extra_params: dict

   .. method:: get_api_key()

      Get API key from config or environment variable.

      :returns: API key string.
      :raises ValueError: If no API key is found.

   .. method:: from_preset(name)
      :classmethod:

      Create an ``LLMConfig`` from a predefined model name.

      :param name: Predefined model name (e.g., ``"gpt-4o-mini"``).
      :type name: str
      :returns: LLMConfig instance.

   .. method:: from_dict(data)
      :classmethod:

      Create an ``LLMConfig`` from a dictionary.

   .. method:: to_dict()

      Convert to a dictionary for JSON serialization.

.. function:: get_llm_config(name_or_path)

   Get LLM config by predefined name or load from a JSON file path.

   :param name_or_path: Predefined model name or path to JSON config file.
   :type name_or_path: str
   :returns: LLMConfig instance.
   :raises ValueError: If the name is not recognized and the path does not exist.

.. function:: list_available_models()

   List all available predefined models with their descriptions.

   :returns: Dict mapping model name to description string.
   :rtype: dict[str, str]

Core Abstractions
-----------------

Trajectory
~~~~~~~~~~

.. class:: Trajectory

   A single episode trajectory (dataclass).

   :param task: Task description dict (e.g., ``{"question": "...", "ground_truth": "..."}``).
   :type task: dict
   :param steps: List of step dicts, each containing ``agent_id``, ``observation``, ``action``, etc.
   :type steps: list[dict]
   :param reward: Episode reward.
   :type reward: float
   :param metadata: Optional metadata dict (default: ``{}``).
   :type metadata: dict

BaseEnvironment
~~~~~~~~~~~~~~~

.. class:: BaseEnvironment

   Abstract base class for environment adapters. Subclass this to add new environments.

   .. method:: reset(task)
      :abstractmethod:

      Reset the environment with the given task.

      :param task: Task description dict.
      :type task: dict
      :returns: Initial observations dict.

   .. method:: step(agent_id, action)
      :abstractmethod:

      Execute an agent's action.

      :param agent_id: Agent identifier string.
      :param action: Action string.
      :returns: Tuple of ``(observation, reward, done, info)``.

   .. method:: collect_trajectory(policies, task)
      :abstractmethod:

      Run a full episode with the given policies and return a trajectory.

      :param policies: Dict mapping agent name to policy text.
      :type policies: dict[str, str]
      :param task: Task description dict.
      :type task: dict
      :returns: Episode trajectory.
      :rtype: Trajectory

BaseAgent
~~~~~~~~~

.. class:: BaseAgent

   Abstract base class for agents with language policies.

   .. method:: act(observation, policy)
      :abstractmethod:

      Given an observation and a policy, produce an action.

      :param observation: Text observation.
      :type observation: str
      :param policy: Language policy (system prompt).
      :type policy: str
      :returns: Action text.
      :rtype: str

BaseCritic
~~~~~~~~~~

.. class:: BaseCritic

   Abstract base class for trajectory evaluators.

   .. method:: evaluate(trajectory, policies)
      :abstractmethod:

      Evaluate a trajectory and return per-agent credits.

      :param trajectory: Episode trajectory.
      :type trajectory: Trajectory
      :param policies: Current agent policies.
      :type policies: dict[str, str]
      :returns: Evaluation dict with ``raw_response`` and ``per_agent`` credits.
      :rtype: dict

BaseReward
~~~~~~~~~~

.. class:: BaseReward

   Abstract base class for reward computation.

   .. method:: compute(trajectory)
      :abstractmethod:

      Compute reward for a trajectory.

      :param trajectory: Episode trajectory.
      :type trajectory: Trajectory
      :returns: Reward value.
      :rtype: float

BaseOptimizer
~~~~~~~~~~~~~

.. class:: BaseOptimizer

   Abstract base class for language gradient optimizers.

   .. method:: generate_gradient(policy, evaluation, context)
      :abstractmethod:

      Generate a language gradient (improvement instruction) for one agent.

      :param policy: Current policy text.
      :param evaluation: Evaluation feedback text.
      :param context: Task context text.
      :returns: Gradient text.
      :rtype: str

   .. method:: apply_gradient(policy, gradient)
      :abstractmethod:

      Apply a gradient to a policy, returning the updated policy.

      :param policy: Current policy text.
      :param gradient: Gradient text.
      :returns: Updated policy text.
      :rtype: str

   .. method:: aggregate_gradients(gradients)
      :abstractmethod:

      Aggregate multiple gradients into a single gradient.

      :param gradients: List of gradient texts.
      :type gradients: list[str]
      :returns: Aggregated gradient text.
      :rtype: str

Concrete Implementations
------------------------

CentralizedCritic
~~~~~~~~~~~~~~~~~

.. class:: CentralizedCritic(config, prompts_dir=None)

   Centralized critic supporting ``central_global`` and ``central_credit`` paradigms.
   Evaluates multi-agent sequential collaboration trajectories using LLM-as-judge.

   :param config: Configuration with ``paradigm``, ``num_agents``, and LLM fields.
   :type config: BaseConfig
   :param prompts_dir: Optional custom path to evaluation prompt templates.
   :type prompts_dir: Path or None

   .. method:: evaluate(trajectory, policies)

      Evaluate a trajectory. For ``central_credit``, returns per-agent causal credits.
      For ``central_global``, returns the same evaluation for all agents.

      :param trajectory: Episode trajectory.
      :type trajectory: Trajectory
      :param policies: Current agent policies.
      :type policies: dict[str, str]
      :returns: Dict with keys ``raw_response``, ``paradigm``, and ``per_agent``.
      :rtype: dict

PolicyGradientOptimizer
~~~~~~~~~~~~~~~~~~~~~~~

.. class:: PolicyGradientOptimizer(llm_config)

   LLM-based policy gradient optimizer. Generates language gradients (concrete
   improvement instructions) and applies them to agent policies.

   :param llm_config: LLM configuration for the optimizer model.
   :type llm_config: LLMConfig

   .. method:: generate_gradient(policy, evaluation, context, agent_name="agent")

      Generate a per-agent improvement instruction based on evaluation feedback.

      :param policy: Agent's current policy text.
      :param evaluation: Evaluation feedback for this agent.
      :param context: Task context (truncated to 800 chars).
      :param agent_name: Agent name for logging.
      :returns: Gradient text (2-4 sentences of specific improvement advice).
      :rtype: str

   .. method:: generate_shared_gradient(evaluation, task_context)

      Generate a shared gradient for all agents (used in ``central_global`` paradigm).

      :param evaluation: Team-level evaluation feedback.
      :param task_context: Task context (truncated to 800 chars).
      :returns: Shared gradient text.
      :rtype: str

   .. staticmethod:: apply_gradient(base_policy, gradient)

      Apply a gradient to a base policy. Appends the gradient as a
      ``[CASE-SPECIFIC FEEDBACK]`` section. The base policy is never modified;
      the feedback section is replaced on every call.

      :param base_policy: Original policy text.
      :param gradient: Gradient text to append.
      :returns: Updated policy text.
      :rtype: str

   .. staticmethod:: aggregate_gradients(gradients)

      Aggregate multiple gradients (from multiple episodes) into one by
      joining with separator markers.

      :param gradients: List of gradient texts.
      :type gradients: list[str]
      :returns: Aggregated gradient text.
      :rtype: str

   .. staticmethod:: parse_credit_response(response, agent_names)

      Parse per-agent evaluations from a credit-assignment LLM response.
      Handles JSON format (language tasks), bracket markers (Overcooked, Pistonball),
      and falls back to returning the full response for all agents.

      :param response: Raw LLM evaluation response.
      :type response: str
      :param agent_names: List of agent name strings.
      :type agent_names: list[str]
      :returns: Dict mapping agent name to evaluation text.
      :rtype: dict[str, str]

TrajectoryFormatter
~~~~~~~~~~~~~~~~~~~

.. class:: TrajectoryFormatter

   Formats episode trajectories for LLM evaluation.

   .. staticmethod:: format_trajectory(episode)

      Format an episode into a text string for global evaluation.

      :param episode: Episode dict with ``task``, ``transitions``, ``reward``.
      :returns: Formatted trajectory string.
      :rtype: str

   .. staticmethod:: format_for_credit_assignment(episode)

      Format an episode for per-agent credit assignment, with structured
      per-agent contribution sections.

      :param episode: Episode dict.
      :returns: Formatted trajectory string.
      :rtype: str

   .. staticmethod:: format_trajectory_minimal(episode)

      Format a concise episode summary.

      :param episode: Episode dict.
      :returns: Minimal trajectory string.
      :rtype: str

Training
--------

MonteCarloTrainer
~~~~~~~~~~~~~~~~~

.. class:: MonteCarloTrainer(config, env, critic, optimizer, reward_fn=None, store=None, callbacks=None)

   Generic Monte Carlo trainer for any LangMARL environment. Implements a five-phase
   training iteration: load policies, generate trajectories, evaluate and generate
   gradients, aggregate and apply gradients, save checkpoint.

   :param config: Training configuration.
   :type config: BaseConfig
   :param env: Environment instance.
   :type env: BaseEnvironment
   :param critic: Critic instance.
   :type critic: BaseCritic
   :param optimizer: Optimizer instance.
   :type optimizer: BaseOptimizer
   :param reward_fn: Optional reward function for verified rewards.
   :type reward_fn: BaseReward or None
   :param store: Storage backend (default: ``LocalStore``).
   :type store: BaseStore or None
   :param callbacks: List of training callbacks.
   :type callbacks: list[Callback] or None

   .. method:: train(num_iterations=None)

      Main training loop. Automatically resumes from the latest checkpoint.

      :param num_iterations: Number of iterations to train. Defaults to ``config.num_iterations``.
      :type num_iterations: int or None
      :returns: Training metrics from the store.

   .. method:: train_one_iteration(iteration)

      Run a single training iteration (five phases).

      :param iteration: Iteration number.
      :type iteration: int
      :returns: Statistics dict with keys: ``paradigm``, ``num_episodes``, ``avg_reward``,
         ``min_reward``, ``max_reward``, ``rewards``, ``input_tokens``, ``output_tokens``,
         ``total_tokens``, ``cost_usd``.
      :rtype: dict

Callbacks
~~~~~~~~~

.. class:: Callback

   Abstract base class for training callbacks. Override any of the following methods:

   .. method:: on_iteration_start(iteration, trainer)
   .. method:: on_iteration_end(iteration, stats, trainer)
   .. method:: on_episode_complete(trajectory, trainer)
   .. method:: on_policy_update(agent_id, old_policy, new_policy)

.. class:: LoggingCallback

   Callback that logs iteration start/end events via the trainer's ``RunLogger``.

.. class:: CheckpointCallback

   Callback for checkpoint management (saving is handled by the trainer by default).

.. class:: EarlyStoppingCallback(patience=3, min_delta=0.01)

   Stop training when the average reward stops improving.

   :param patience: Number of iterations to wait without improvement before stopping.
   :type patience: int
   :param min_delta: Minimum improvement to qualify as progress.
   :type min_delta: float

LLM Client
-----------

LLMClient
~~~~~~~~~

.. class:: LLMClient(llm_config)

   Unified LLM client wrapping OpenAI-compatible APIs.

   :param llm_config: LLM configuration.
   :type llm_config: LLMConfig

   .. method:: chat(system_prompt, user_input, max_tokens=None)

      Send a chat completion request and return the response text.

      :param system_prompt: System message.
      :type system_prompt: str
      :param user_input: User message.
      :type user_input: str
      :param max_tokens: Maximum tokens (default: from config).
      :type max_tokens: int or None
      :returns: Response text.
      :rtype: str

   .. method:: chat_with_usage(system_prompt, user_input, max_tokens=None)

      Chat and return both response text and token usage.

      :returns: Tuple of ``(response_text, {"input": N, "output": N})``.
      :rtype: tuple[str, dict[str, int]]

   .. attribute:: raw_client

      Access the underlying ``OpenAI`` client instance.

TokenTracker
~~~~~~~~~~~~

.. class:: TokenTracker(model="gpt-4o-mini", input_price=None, output_price=None)

   Tracks token usage and estimates costs for LLM API calls. Has built-in pricing
   for common models.

   :param model: Model name for pricing lookup.
   :type model: str
   :param input_price: Custom input price per million tokens (overrides lookup).
   :type input_price: float or None
   :param output_price: Custom output price per million tokens (overrides lookup).
   :type output_price: float or None

   .. method:: add_usage(input_tokens, output_tokens)

      Add token usage counts.

      :param input_tokens: Number of input tokens.
      :param output_tokens: Number of output tokens.

   .. method:: get_stats()

      Get usage statistics and cost estimate.

      :returns: Dict with ``model``, ``input_tokens``, ``output_tokens``, ``total_tokens``,
         ``input_cost_usd``, ``output_cost_usd``, ``cost_usd``.
      :rtype: dict

   .. method:: estimate_cost(input_tokens, output_tokens)

      Estimate cost for given token counts without adding to the tracker.

      :returns: Estimated cost in USD.
      :rtype: float

   .. method:: reset()

      Reset all token counters to zero.

   .. method:: get_summary_string()

      Get a human-readable summary of token usage and costs.

      :returns: Formatted summary string.
      :rtype: str

Storage
-------

LocalStore
~~~~~~~~~~

.. class:: LocalStore(base_dir)

   Filesystem-backed storage backend. Manages training run data including
   trajectories, checkpoints, evaluations, gradients, metrics, and logs.

   :param base_dir: Base directory for all storage.
   :type base_dir: str

PolicyCheckpoint
~~~~~~~~~~~~~~~~

.. class:: PolicyCheckpoint(store, run_id, num_agents)

   Manages versioned policy snapshots. Stores one text file per agent per iteration.

   :param store: Storage backend.
   :type store: BaseStore
   :param run_id: Training run identifier.
   :type run_id: str
   :param num_agents: Number of agents.
   :type num_agents: int

   .. method:: get_policies(iteration=None)

      Load policies from the latest (or a specific) iteration. Generates default
      policies if no checkpoint exists.

      :param iteration: Specific iteration to load. ``None`` loads the latest.
      :type iteration: int or None
      :returns: Dict mapping agent name to policy text.
      :rtype: dict[str, str]

   .. method:: save_policies(iteration, policies, stats=None)

      Save policies as a new iteration checkpoint.

      :param iteration: Iteration number.
      :param policies: Dict mapping agent name to policy text.
      :param stats: Optional training statistics.

   .. method:: diff_policies(iter_a, iter_b)

      Compare policies across two iterations.

      :param iter_a: First iteration number.
      :param iter_b: Second iteration number.
      :returns: Dict mapping agent name to diff text.

TrajectoryStore
~~~~~~~~~~~~~~~

.. class:: TrajectoryStore(store, run_id)

   Manages episode trajectory persistence.

   .. method:: save(iteration, episode_id, trajectory)

      Save a trajectory for a given iteration and episode.

   .. method:: load(iteration, limit=None)

      Load trajectories for a given iteration.

      :param iteration: Iteration number.
      :param limit: Maximum number of trajectories to load.
      :returns: List of Trajectory objects.
      :rtype: list[Trajectory]

   .. method:: count(iteration)

      Count stored trajectories for a given iteration.

      :returns: Number of stored trajectories.
      :rtype: int

RunLogger
~~~~~~~~~

.. class:: RunLogger(store, run_id)

   Structured training logger. Writes to both file and console.

   .. method:: info(message)
   .. method:: debug(message)
   .. method:: warning(message)
   .. method:: error(message)
   .. method:: iteration_start(iteration, policies)
   .. method:: iteration_end(iteration, stats)
   .. method:: episode_saved(iteration, episode_id, reward)
   .. method:: evaluation_done(iteration, episode_id, paradigm, response)
   .. method:: gradient_saved(iteration, agent_name, num_gradients)
