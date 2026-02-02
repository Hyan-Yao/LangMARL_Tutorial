API Reference
=============

Training Module
---------------

The ``training`` module provides the core components for Monte Carlo training
on the Pistonball environment using TextGrad optimization.

TrainingConfig
~~~~~~~~~~~~~~

.. class:: training.TrainingConfig

   Configuration dataclass for Monte Carlo training on Pistonball.

   :param exp_name: Experiment name.
   :type exp_name: str
   :param paradigm: Training paradigm. One of ``"independent"``, ``"central_global"``, ``"central_credit"``.
   :type paradigm: str
   :param policy_path: Path to the policy JSON file.
   :type policy_path: str
   :param experiments_dir: Directory to save experiment data.
   :type experiments_dir: str
   :param num_iterations: Number of training iterations (default: ``5``).
   :type num_iterations: int
   :param trajectories_per_iteration: Number of episodes per iteration (default: ``3``).
   :type trajectories_per_iteration: int
   :param model: LLM model name (default: ``"gpt-4o-mini"``).
   :type model: str
   :param llm_config: Optional LLM configuration dict or predefined model name. Takes precedence over ``model``.
   :type llm_config: dict or None
   :param episode_llm_config: Optional separate LLM config for episode generation.
   :type episode_llm_config: dict or None
   :param textgrad_llm_config: Optional separate LLM config for TextGrad backward engine.
   :type textgrad_llm_config: dict or None
   :param num_pistons: Number of pistons in the environment (default: ``20``).
   :type num_pistons: int
   :param max_cycles: Maximum cycles per episode (default: ``125``).
   :type max_cycles: int
   :param agent_type: Agent type, ``"llm"`` or ``"rule"`` (default: ``"llm"``).
   :type agent_type: str
   :param action_mode: Action mode, ``"discrete"`` or ``"continuous"`` (default: ``"discrete"``).
   :type action_mode: str

   .. method:: from_json(config_path)
      :classmethod:

      Load configuration from a JSON file.

      :param config_path: Path to JSON configuration file.
      :returns: TrainingConfig object.

   .. method:: to_json(output_path)

      Save configuration to a JSON file.

   .. method:: get_llm_config()

      Get the resolved ``LLMConfig`` object.

   .. method:: get_episode_llm_config()

      Get the LLM configuration for episode generation. Falls back to ``get_llm_config()``
      if ``episode_llm_config`` is not set.

   .. method:: get_textgrad_llm_config()

      Get the LLM configuration for TextGrad backward engine. Falls back to ``get_llm_config()``
      if ``textgrad_llm_config`` is not set.

   .. method:: list_available_models()
      :staticmethod:

      List all available predefined LLM models.

MonteCarloTrainer
~~~~~~~~~~~~~~~~~

.. class:: training.MonteCarloTrainer(config)

   Main training orchestrator for Pistonball using Monte Carlo trajectory
   evaluation and TextGrad optimization.

   :param config: Training configuration.
   :type config: TrainingConfig

   .. method:: train_full()

      Train for all iterations defined in config. Returns a list of stats
      dictionaries, one per iteration.

      :returns: List of training statistics (``avg_reward``, ``rewards``, ``total_tokens``, ``cost_usd``, etc.).
      :rtype: list[dict]

   .. method:: train_one_iteration(iteration)

      Train a single iteration. Consists of:

      1. Loading policies for the iteration
      2. Generating episodes via simulation (or loading existing ones)
      3. Evaluating episodes and accumulating gradients
      4. Optimizing policies with TextGrad
      5. Saving checkpoints

      :param iteration: Iteration number.
      :type iteration: int
      :returns: Training statistics dictionary.
      :rtype: dict

LLMConfig
~~~~~~~~~

.. class:: training.LLMConfig

   Configuration dataclass for a single LLM model. All providers use
   OpenAI-compatible API format.

   :param name: Model identifier (e.g., ``"gpt-4o"``, ``"gemini-pro"``).
   :type name: str
   :param model_string: Model string to pass to the API.
   :type model_string: str
   :param base_url: Custom API base URL. ``None`` means default OpenAI endpoint.
   :type base_url: str or None
   :param api_key: API key. ``None`` means use environment variable.
   :type api_key: str or None
   :param api_key_env_var: Environment variable name for API key (default: ``"OPENAI_API_KEY"``).
   :type api_key_env_var: str
   :param is_multimodal: Whether the model supports multimodal input (default: ``False``).
   :type is_multimodal: bool
   :param max_tokens: Maximum tokens per response (default: ``4096``).
   :type max_tokens: int
   :param input_price_per_million: Input token price per million (USD).
   :type input_price_per_million: float
   :param output_price_per_million: Output token price per million (USD).
   :type output_price_per_million: float

   .. method:: get_api_key()

      Get API key from config or environment variable.

   .. method:: to_dict()

      Convert to dictionary for JSON serialization.

   .. method:: from_dict(data)
      :classmethod:

      Create ``LLMConfig`` from a dictionary.

.. function:: training.get_llm_config(name_or_path)

   Get LLM config by predefined name (e.g., ``"gpt-4o"``) or load from a JSON file path.

   :param name_or_path: Predefined model name or path to JSON config file.
   :type name_or_path: str
   :returns: LLMConfig object.

.. function:: training.list_available_models()

   List all available predefined models with their descriptions.

   :returns: Dict mapping model name to description string.
   :rtype: dict[str, str]

EpisodeGenerator
~~~~~~~~~~~~~~~~

.. class:: training.EpisodeGenerator

   Generates episodes by running simulations in the Pistonball environment.

   :param num_pistons: Number of pistons (default: ``20``).
   :param max_cycles: Maximum cycles per episode (default: ``125``).
   :param agent_type: ``"rule"`` for formula-based control, ``"llm"`` for LLM agents (default: ``"rule"``).
   :param action_mode: ``"discrete"`` (3 actions) or ``"continuous"`` (default: ``"discrete"``).
   :param gpt_model: Model name for LLM agents (default: ``"gpt-4o-mini"``).
   :param llm_config: Optional LLM configuration (``LLMConfig``, dict, or string).

   .. method:: generate_episodes(num_episodes, iteration, agent_policies=None, parallel=True, save_gif=True)

      Generate multiple episodes. Supports parallel execution via multiprocessing.

      :param num_episodes: Number of episodes to generate.
      :param iteration: Current training iteration.
      :param agent_policies: Dict mapping agent name to policy text.
      :param parallel: If ``True``, run episodes in parallel.
      :param save_gif: Whether to save GIF for each episode.
      :returns: List of episode data dictionaries.
      :rtype: list[dict]

   .. method:: evaluate_policy(agent_policies=None, num_episodes=1, save_dir=None)

      Evaluate a policy by running episodes and recording metrics.

      :param agent_policies: Dict mapping agent name to policy text.
      :param num_episodes: Number of evaluation episodes.
      :param save_dir: Directory to save evaluation results.
      :returns: Evaluation metrics (mean_reward, std_reward, etc.).
      :rtype: dict

   .. method:: render_single_episode(agent_policies=None, save_path=None, fps=30)

      Render a single episode and save as GIF.

      :returns: Tuple of ``(total_reward, num_steps, gif_path)``.

PolicyOptimizer
~~~~~~~~~~~~~~~

.. class:: training.PolicyOptimizer(model="gpt-4o-mini", llm_config=None)

   TextGrad policy optimizer for Pistonball. Wraps TextGrad's backward engine
   and Textual Gradient Descent (TGD) optimizer.

   :param model: Model name for TextGrad backward engine.
   :param llm_config: Optional LLM configuration (takes precedence over ``model``).

   .. method:: initialize_policies(agent_policies)

      Initialize agent policies as TextGrad ``Variable`` objects with gradient tracking.

      :param agent_policies: Dict mapping agent name to policy text.
      :returns: Dict mapping agent name to ``tg.Variable``.

   .. method:: accumulate_gradients(policy_vars, feedback, reduce_group_id=None)

      Accumulate gradients from a single episode's evaluation feedback. Supports
      batch gradient reduction when ``reduce_group_id`` is provided.

      :param policy_vars: List of policy ``Variable`` objects.
      :param feedback: Evaluation prompt text.
      :param reduce_group_id: Optional ID for batch gradient aggregation.

   .. method:: accumulate_gradients_credit(policy_vars, feedback, agent_names=None, reduce_group_id=None)

      Accumulate gradients for the credit assignment paradigm. Makes a single LLM
      call to evaluate all agents jointly, then parses and distributes per-agent gradients.

      :param policy_vars: List of policy variables.
      :param feedback: Joint evaluation prompt.
      :param agent_names: List of agent names corresponding to policy variables.

   .. method:: optimize_step(policy_vars)

      Execute one optimization step using Textual Gradient Descent (TGD).

      :param policy_vars: List of policy variables with accumulated gradients.
      :returns: List of optimized policy strings.
      :rtype: list[str]

ParadigmEvaluator
~~~~~~~~~~~~~~~~~

.. class:: training.ParadigmEvaluator(paradigm, num_pistons=20, prompts_dir=None)

   Evaluator for the three training paradigms. Loads prompt templates from
   external JSON files for easy customization.

   :param paradigm: ``"independent"``, ``"central_global"``, or ``"central_credit"``.
   :param num_pistons: Number of pistons in the environment.
   :param prompts_dir: Optional custom path to prompts directory.

   .. method:: create_evaluation_prompt(trajectory, agent_id=None)

      Create an evaluation prompt based on the configured paradigm.

      :param trajectory: Formatted trajectory string.
      :param agent_id: Agent name (required for ``independent`` paradigm).
      :returns: Complete LLM evaluation prompt.
      :rtype: str

TrajectoryFormatter
~~~~~~~~~~~~~~~~~~~

.. class:: training.TrajectoryFormatter

   Formats episode trajectories for LLM evaluation. Samples every 10 timesteps
   and includes global state, agent actions, and reward information.

   .. method:: format_trajectory(episode)
      :staticmethod:

      Format an episode's trajectory into a text string for LLM evaluation.

      :param episode: Episode dict containing transitions and total reward.
      :returns: Formatted trajectory string.
      :rtype: str

PolicyManager
~~~~~~~~~~~~~

.. class:: training.PolicyManager(policy_path, num_agents=20)

   Manages policy checkpoints and versioning. Stores per-agent policies across
   training iterations in a JSON file.

   :param policy_path: Path to the policy JSON file.
   :param num_agents: Number of agents (pistons).

   .. method:: get_current_policies()

      Get policies for the current iteration.

      :returns: Dict mapping agent name to policy text.
      :rtype: dict[str, str]

   .. method:: save_updated_policies(agent_policies, stats=None)

      Save optimized policies as a new iteration checkpoint.

      :param agent_policies: Dict mapping agent name to optimized policy text.
      :param stats: Optional training statistics.

   .. method:: load_specific_iteration(iteration)

      Load policies from a specific iteration.

      :param iteration: Iteration number.
      :returns: Dict mapping agent name to policy text.

   .. method:: get_training_history()

      Get training statistics across all iterations.

      :returns: List of stats dictionaries.
      :rtype: list[dict]

TokenTracker
~~~~~~~~~~~~

.. class:: training.TokenTracker(model="gpt-4o-mini", input_price=None, output_price=None)

   Tracks token usage and estimates costs for LLM API calls.

   :param model: Model name for pricing lookup.
   :param input_price: Custom input price per million tokens (overrides lookup).
   :param output_price: Custom output price per million tokens (overrides lookup).

   .. method:: add_usage(input_tokens, output_tokens)

      Add token usage counts.

   .. method:: get_stats()

      Get usage statistics and cost estimate.

      :returns: Dict with ``input_tokens``, ``output_tokens``, ``total_tokens``, ``cost_usd``.
      :rtype: dict

   .. method:: estimate_cost(input_tokens, output_tokens)

      Estimate cost for a given number of tokens without adding to the tracker.

      :returns: Estimated cost in USD.
      :rtype: float

TextGrad Module
---------------

The ``textgrad`` module implements language-based automatic differentiation.
It provides the core primitives for treating natural language as an optimization space.

Variable
~~~~~~~~

.. class:: textgrad.Variable(value, requires_grad=False, role_description="")

   Core computational graph node. Represents a text value that can optionally
   track gradients for optimization.

   :param value: The text content.
   :param requires_grad: Whether to track gradients for this variable.
   :param role_description: Description of this variable's role in the computation.

TextLoss
~~~~~~~~

.. class:: textgrad.TextLoss(eval_prompt)

   Evaluation module for text-based loss computation. Takes an evaluation prompt
   and produces a loss variable when called with input variables.

BlackboxLLM
~~~~~~~~~~~~

.. class:: textgrad.BlackboxLLM(engine=None, system_prompt=None)

   Wrapper for LLM calls within the TextGrad computation graph.

TextualGradientDescent
~~~~~~~~~~~~~~~~~~~~~~

.. class:: textgrad.TextualGradientDescent(parameters)

   Language-based optimizer that updates text variables based on accumulated
   gradient feedback. Also available as ``textgrad.TGD``.

   :param parameters: List of ``Variable`` objects to optimize.

   .. method:: step()

      Execute one optimization step, updating all parameter values based on
      their accumulated gradients.

Engine
~~~~~~

.. function:: textgrad.get_engine(model_string)

   Factory function to create an LLM engine. Supports OpenAI, Anthropic, Gemini,
   Together, Cohere, Groq, Ollama, vLLM, and other providers.

   :param model_string: Model identifier string.
   :returns: Engine instance.

.. function:: textgrad.set_backward_engine(engine, override=False)

   Set the global backward engine used for gradient computation.

   :param engine: Engine instance to use.
   :param override: Whether to override an existing engine.
