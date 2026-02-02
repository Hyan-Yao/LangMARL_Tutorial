"""Main Monte Carlo trainer module for Pistonball"""

import logging
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from .config import TrainingConfig
from .trajectory import TrajectoryFormatter
from .evaluators import ParadigmEvaluator
from .optimizer import PolicyOptimizer
from .token_tracker import TokenTracker
from .episode_loader import EpisodeLoader
from .episode_generator import EpisodeGenerator, estimate_tokens
from .policy_manager import PolicyManager


class MonteCarloTrainer:
    """Monte Carlo trainer for Pistonball - main orchestrator"""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize components
        self.policy_manager = PolicyManager(config.policy_path, num_agents=config.num_pistons)
        self.episode_loader = EpisodeLoader(config.experiments_dir)
        # Get separate LLM configs for episode generation and textgrad
        episode_llm_config = config.get_episode_llm_config()
        textgrad_llm_config = config.get_textgrad_llm_config()

        self.episode_generator = EpisodeGenerator(
            num_pistons=config.num_pistons,
            max_cycles=config.max_cycles,
            frame_size=config.frame_size,
            stack_size=config.stack_size,
            continuous=config.continuous,
            experiments_dir=config.experiments_dir,
            exp_name=config.exp_name,
            agent_type=config.agent_type,
            gpt_model=episode_llm_config.model_string,
            action_mode=config.action_mode,
            rule_k=config.rule_k,
            rule_tau=config.rule_tau,
            llm_config=episode_llm_config,
        )
        self.formatter = TrajectoryFormatter()
        self.evaluator = ParadigmEvaluator(
            config.paradigm,
            num_pistons=config.num_pistons
        )

        # Initialize optimizer with textgrad LLM config
        self.optimizer = PolicyOptimizer(
            model=textgrad_llm_config.model_string,
            llm_config=textgrad_llm_config
        )

        # Token tracker uses episode generation model pricing (primary token consumer)
        self.token_tracker = TokenTracker(
            model=episode_llm_config.model_string,
            input_price=episode_llm_config.input_price_per_million,
            output_price=episode_llm_config.output_price_per_million
        )

        # Store configs for logging
        self.episode_llm_config = episode_llm_config
        self.textgrad_llm_config = textgrad_llm_config

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.logger.info(f"Initialized {config.paradigm} trainer for Pistonball")
        self.logger.info(f"Episode generation model: {episode_llm_config.name} ({episode_llm_config.model_string})")
        self.logger.info(f"TextGrad model: {textgrad_llm_config.name} ({textgrad_llm_config.model_string})")
        self.logger.info(f"Current iteration: {self.policy_manager.current_iteration}")

        # Track textgrad evaluation texts and tokens
        self.textgrad_eval_texts = []
        self.textgrad_token_stats = {
            'total_estimated_tokens': 0,
            'total_queries': 0
        }

    def train_one_iteration(self, iteration: int) -> Dict:
        """
        Train one iteration

        Args:
            iteration: Iteration number

        Returns:
            Training statistics dictionary
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting Iteration {iteration}")
        self.logger.info(f"{'='*60}")

        # Reset textgrad tracking for this iteration
        self.textgrad_eval_texts = []
        self.textgrad_token_stats = {
            'total_estimated_tokens': 0,
            'total_queries': 0
        }

        # Create iteration directory for textgrad logs
        iter_dir = Path(self.config.experiments_dir) / self.config.exp_name / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        textgrad_log_file = iter_dir / "textgrad_evaluations.txt"

        # Setup TextGrad logger to write to iteration directory
        self.optimizer.setup_textgrad_logger(iter_dir)

        # Phase 1: Load policies for this iteration
        self.logger.info(f"Phase 1: Loading policies for iteration {iteration}...")
        agent_policies = None

        # Try to load policies for the specific iteration
        try:
            agent_policies = self.policy_manager.load_specific_iteration(iteration)
            self.logger.info(f"Loaded existing policies from iteration_{iteration}")
        except ValueError:
            pass

        # Fallback: try previous iteration's policies (if iteration > 0)
        if agent_policies is None and iteration > 0:
            try:
                agent_policies = self.policy_manager.load_specific_iteration(iteration - 1)
                self.logger.info(
                    f"No policies for iteration_{iteration}, "
                    f"using iteration_{iteration - 1} policies"
                )
            except ValueError:
                pass

        # Final fallback: use current policies
        if agent_policies is None:
            self.logger.info(
                f"No policies found for iteration_{iteration} or {iteration - 1}, "
                f"using current policies (iteration_{self.policy_manager.current_iteration})"
            )
            agent_policies = self.policy_manager.get_current_policies()

        agent_policy_vars = self.optimizer.initialize_policies(agent_policies)
        self.logger.info(f"Loaded policies for {len(agent_policies)} agents")

        # Phase 2: Generate episodes by running simulation (or load existing)
        existing_episode_count = self.episode_loader.count_episodes(
            self.config.exp_name, iteration
        )

        if existing_episode_count >= self.config.trajectories_per_iteration:
            # Episodes already exist, load them instead of re-running simulation
            self.logger.info(
                f"Phase 2: Found {existing_episode_count} existing episodes for iteration {iteration}, "
                f"loading from disk (skipping simulation)..."
            )
            episodes = self.episode_loader.load_episodes(
                self.config.exp_name,
                iteration,
                num_episodes=self.config.trajectories_per_iteration
            )
            self.logger.info(f"Loaded {len(episodes)} episodes from disk")
        else:
            # Generate new episodes via simulation
            self.logger.info(
                f"Phase 2: Generating {self.config.trajectories_per_iteration} episodes "
                f"via simulation..."
            )
            episodes = self.episode_generator.generate_episodes(
                num_episodes=self.config.trajectories_per_iteration,
                iteration=iteration,
                agent_policies=agent_policies,
            )

            # Track token usage from simulation
            for episode_data in episodes:
                token_stats = episode_data.get("token_stats", {})
                estimated_tokens = token_stats.get("estimated_tokens_simulation", {})
                self.token_tracker.add_usage(
                    estimated_tokens.get("input", 0),
                    estimated_tokens.get("output", 0)
                )

        # Validate episodes
        if not self.episode_loader.validate_episodes(episodes):
            raise ValueError("Invalid episode data")

        if len(episodes) != self.config.trajectories_per_iteration:
            raise ValueError(
                f"Expected {self.config.trajectories_per_iteration} episodes, "
                f"got {len(episodes)}"
            )

        # Show episode stats
        episode_stats = self.episode_loader.get_episode_stats(episodes)
        self.logger.info(f"Episodes generated: {episode_stats}")

        # Phase 3: Evaluate episodes and accumulate gradients
        self.logger.info("Phase 3: Evaluating episodes and accumulating gradients...")

        # Create unique reduce group IDs for this batch of episodes
        reduce_group_ids = self._create_reduce_group_ids(iteration, agent_policy_vars)

        for i, episode in enumerate(tqdm(episodes, desc="Evaluating episodes")):
            self._evaluate_and_accumulate_gradients(
                episode, agent_policy_vars, iteration, reduce_group_ids
            )

        # Log gradient accumulation
        self.logger.info(f"Accumulated gradients from {len(episodes)} episodes")
        sample_agent = list(agent_policy_vars.keys())[0]
        self.logger.info(f"Sample agent ({sample_agent}) has {len(agent_policy_vars[sample_agent].gradients)} gradients")

        # Save textgrad evaluation texts with enhanced formatting
        with open(textgrad_log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"TEXTGRAD EVALUATIONS - Iteration {iteration}\n")
            f.write(f"Paradigm: {self.config.paradigm}\n")
            f.write(f"Number of Agents: {self.config.num_pistons}\n")
            f.write(f"Episodes Evaluated: {len(episodes)}\n")
            f.write("="*80 + "\n\n")

            for idx, eval_text in enumerate(self.textgrad_eval_texts):
                f.write(f"--- Evaluation {idx + 1} ---\n")
                f.write(eval_text)
                f.write("\n" + "-"*80 + "\n\n")

            # Add summary section
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total Evaluations: {len(self.textgrad_eval_texts)}\n")
            f.write(f"Estimated Tokens: {self.textgrad_token_stats['total_estimated_tokens']:,}\n")
            f.write(f"Total Queries: {self.textgrad_token_stats['total_queries']}\n")

        self.logger.info(f"Saved textgrad evaluations to {textgrad_log_file}")

        # Save credit assignment specific responses if in credit paradigm
        if self.config.paradigm == "central_credit":
            credit_file = self.optimizer.save_credit_responses(iter_dir / "credit_assignment_responses.txt")
            if credit_file:
                self.logger.info(f"Saved credit assignment responses to {credit_file}")

        # Phase 4: Optimize policies
        self.logger.info("Phase 4: Optimizing policies...")

        # Gather all policy variables for optimization
        all_policy_vars = list(agent_policy_vars.values())

        optimized_policies = self.optimizer.optimize_step(all_policy_vars)

        # Extract optimized policies as dict
        opt_agent_policies = {}
        for agent_name, opt_policy in zip(agent_policy_vars.keys(), optimized_policies):
            opt_agent_policies[agent_name] = opt_policy

        # Phase 5: Collect stats and save
        self.logger.info("Phase 5: Saving checkpoint...")
        stats = self._collect_stats(episodes)

        # Add textgrad token stats
        stats['textgrad_estimated_tokens'] = self.textgrad_token_stats['total_estimated_tokens']
        stats['textgrad_queries'] = self.textgrad_token_stats['total_queries']

        self.policy_manager.save_updated_policies(opt_agent_policies, stats)

        # Phase 6: Log results
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Iteration {iteration} completed")
        self.logger.info(f"Average reward: {stats['avg_reward']:.2f}")
        self.logger.info(f"Tokens: {stats['total_tokens']:,}")
        self.logger.info(f"TextGrad estimated tokens: {stats['textgrad_estimated_tokens']:,}")
        self.logger.info(f"Cost: ${stats['cost_usd']:.4f}")
        self.logger.info(f"{'='*60}\n")

        return stats

    def _create_reduce_group_ids(self, iteration, agent_policy_vars):
        """Create unique reduce group IDs for gradient aggregation"""
        ids = {
            'global': id((iteration, 'global'))
        }

        # Create reduce group ID for each agent
        for agent_name, policy_var in agent_policy_vars.items():
            ids[agent_name] = id((iteration, policy_var))

        return ids

    def _evaluate_and_accumulate_gradients(
        self, episode, agent_policy_vars, iteration, reduce_group_ids
    ):
        """
        Evaluate single episode and accumulate gradients

        Args:
            episode: Episode dict
            agent_policy_vars: Dict mapping agent_name -> tg.Variable
            iteration: Current iteration number
            reduce_group_ids: Dict of IDs for batch gradient reduction groups
        """
        if self.config.paradigm == "independent":
            self._evaluate_independent(
                episode, agent_policy_vars, iteration, reduce_group_ids
            )
        elif self.config.paradigm == "central_global":
            self._evaluate_global(
                episode, agent_policy_vars, iteration, reduce_group_ids
            )
        elif self.config.paradigm == "central_credit":
            self._evaluate_credit(
                episode, agent_policy_vars, iteration, reduce_group_ids
            )

    def _evaluate_independent(
        self, episode, agent_policy_vars, iteration, reduce_group_ids
    ):
        """Independent learning evaluation - separate evaluators for each agent"""
        # Format trajectory once (same for all agents)
        traj = self.formatter.format_trajectory(episode)

        # Evaluate each agent independently
        for agent_name, policy_var in agent_policy_vars.items():
            # Create evaluation prompt
            prompt = self.evaluator.create_evaluation_prompt(traj, agent_id=agent_name)

            # Save evaluation text and estimate tokens
            eval_header = f"Episode {episode['episode_id']} - {agent_name} Evaluation:\n"
            self.textgrad_eval_texts.append(eval_header + prompt)
            token_est = estimate_tokens(prompt)
            self.textgrad_token_stats['total_estimated_tokens'] += token_est['estimated_tokens']
            self.textgrad_token_stats['total_queries'] += 1

            # Get reduce group ID for this agent
            reduce_id = reduce_group_ids.get(agent_name, reduce_group_ids['global'])

            # Accumulate gradients
            self.optimizer.accumulate_gradients(
                [policy_var], prompt, reduce_group_id=reduce_id
            )

    def _evaluate_global(
        self, episode, agent_policy_vars, iteration, reduce_group_ids
    ):
        """Central global reward evaluation - all agents share the same global feedback"""
        # Format trajectory
        traj = self.formatter.format_trajectory(episode)
        prompt = self.evaluator.create_evaluation_prompt(traj)

        # Save evaluation text and estimate tokens
        eval_header = f"Episode {episode['episode_id']} - Global Team Evaluation:\n"
        self.textgrad_eval_texts.append(eval_header + prompt)
        token_est = estimate_tokens(prompt)
        self.textgrad_token_stats['total_estimated_tokens'] += token_est['estimated_tokens']
        self.textgrad_token_stats['total_queries'] += 1

        # Accumulate gradients for all agent policies with global feedback
        all_policy_vars = list(agent_policy_vars.values())
        self.optimizer.accumulate_gradients(
            all_policy_vars, prompt, reduce_group_id=reduce_group_ids['global']
        )

    def _evaluate_credit(
        self, episode, agent_policy_vars, iteration, reduce_group_ids
    ):
        """Central credit assignment evaluation - per-agent credit assignment"""
        # Format trajectory
        traj = self.formatter.format_trajectory(episode)

        # Create joint evaluation prompt
        prompt = self.evaluator.create_evaluation_prompt(traj)

        # Save evaluation text and estimate tokens
        eval_header = f"Episode {episode['episode_id']} - Credit Assignment:\n"
        self.textgrad_eval_texts.append(eval_header + prompt)
        token_est = estimate_tokens(prompt)
        self.textgrad_token_stats['total_estimated_tokens'] += token_est['estimated_tokens']
        self.textgrad_token_stats['total_queries'] += 1

        # Use per-agent credit assignment
        # Pass all agent policy variables for credit assignment
        all_policy_vars = list(agent_policy_vars.values())
        agent_names = list(agent_policy_vars.keys())

        self.optimizer.accumulate_gradients_credit(
            all_policy_vars, prompt,
            agent_names=agent_names,
            reduce_group_id=reduce_group_ids['global']
        )

    def _collect_stats(self, episodes: List[Dict]) -> Dict:
        """
        Collect training statistics

        Args:
            episodes: List of episodes

        Returns:
            Stats dictionary
        """
        rewards = [ep['total_reward'] for ep in episodes]
        token_stats = self.token_tracker.get_stats()

        return {
            "paradigm": self.config.paradigm,
            "avg_reward": sum(rewards) / len(rewards),
            "rewards": rewards,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "input_tokens": token_stats['input_tokens'],
            "output_tokens": token_stats['output_tokens'],
            "total_tokens": token_stats['total_tokens'],
            "cost_usd": token_stats['cost_usd'],
        }

    def train_full(self):
        """
        Train for all iterations

        Returns:
            List of stats for each iteration
        """
        all_stats = []

        start_iter = self.config.start_iteration
        end_iter = self.config.num_iterations

        # Sync PolicyManager's current_iteration with start_iteration
        # This ensures save_updated_policies uses the correct iteration number
        if start_iter > 0:
            self.logger.info(f"Resuming training from iteration {start_iter}")
            # Set current_iteration to start_iter - 1 so that after training,
            # save_updated_policies will increment it to start_iter + 1
            self.policy_manager.current_iteration = start_iter

        for iteration in range(start_iter, end_iter):
            stats = self.train_one_iteration(iteration)
            all_stats.append(stats)

        self.logger.info("\n" + "="*60)
        self.logger.info("Training completed!")
        self.logger.info(f"Total iterations: {len(all_stats)}")
        self.logger.info(f"Final average reward: {all_stats[-1]['avg_reward']:.2f}")
        self.logger.info(f"Total cost: ${sum(s['cost_usd'] for s in all_stats):.4f}")
        self.logger.info("="*60)

        # Cleanup TextGrad logger
        self.optimizer.cleanup_textgrad_logger()

        return all_stats
