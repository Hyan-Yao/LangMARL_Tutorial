#!/usr/bin/env python3
"""
Example training script for Pistonball using Monte Carlo TextGrad optimization.

This script demonstrates how to use the training framework to train
multi-agent policies in the Pistonball environment.

Usage:
    # Using command line arguments
    python train_pistonball.py --paradigm central_global --num_iterations 5

    # Using a config file
    python train_pistonball.py --config experiments/my_exp/config.json

    # Using different LLM models
    python train_pistonball.py --model gpt-4o
    python train_pistonball.py --model gemini-pro
    python train_pistonball.py --model llama-3.1-70b
    python train_pistonball.py --model qwen-72b

    # Using custom LLM config file
    python train_pistonball.py --llm_config configs/my_llm.json

    # List available models
    python train_pistonball.py --list_models

Paradigms:
    - independent: Each piston group has a separate critic
    - central_global: Shared critic for team performance
    - central_credit: Evaluate individual contributions to team success

Supported Models (all use OpenAI-compatible API):
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-5, o1, o1-mini
    - Google: gemini-pro, gemini-flash, gemini-2.0-flash
    - Llama: llama-3.1-70b, llama-3.1-8b, llama-3.3-70b
    - Qwen: qwen-72b, qwen-7b, qwen-coder-32b
    - DeepSeek: deepseek-chat, deepseek-reasoner
    - Local: ollama-llama3, ollama-qwen2
"""

import argparse
import os
import sys
from pathlib import Path

from training import TrainingConfig, MonteCarloTrainer, list_available_models

os.environ['OPENAI_API_KEY'] = ""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train multi-agent policies on Pistonball using TextGrad',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config file option (takes precedence over other args)
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to JSON config file. If provided, other args are ignored except --start_iteration.'
    )

    # List available models
    parser.add_argument(
        '--list_models', action='store_true',
        help='List all available predefined LLM models and exit'
    )

    # Experiment settings
    parser.add_argument(
        '--exp_name', type=str, default='pistonball_exp',
        help='Experiment name'
    )
    parser.add_argument(
        '--paradigm', type=str, default='central_global',
        choices=['independent', 'central_global', 'central_credit'],
        help='Training paradigm'
    )

    # Training parameters
    parser.add_argument(
        '--num_iterations', type=int, default=10,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--trajectories_per_iteration', type=int, default=3,
        help='Number of episodes per iteration'
    )
    parser.add_argument(
        '--start_iteration', type=int, default=0,
        help='Starting iteration (for resuming)'
    )

    # Environment parameters
    parser.add_argument(
        '--num_pistons', type=int, default=10,
        help='Number of pistons in the environment'
    )
    parser.add_argument(
        '--max_cycles', type=int, default=125,
        help='Maximum cycles per episode'
    )
    parser.add_argument(
        '--agent_type', type=str, default='llm',
        choices=['rule', 'llm'],
        help='Agent type: rule (formula-based control), llm (LLM parses actions from prompts)'
    )
    parser.add_argument(
        '--action_mode', type=str, default='discrete',
        choices=['discrete', 'continuous'],
        help='Action mode: discrete (0-retract, 1-stay, 2-push) or continuous [-1, 1]'
    )
    parser.add_argument(
        '--rule_k', type=float, default=1.0,
        help='Slope coefficient k for rule-based scoring function S_i = k*(x_i - x_b) - smoothing'
    )
    parser.add_argument(
        '--rule_tau', type=float, default=4.0,
        help='Dead-zone threshold tau (pixels) for rule-based action quantization'
    )

    # Model settings
    parser.add_argument(
        '--model', type=str, default='gpt-4o-mini',
        help='LLM model name (predefined: gpt-4o, gemini-pro, llama-3.1-70b, qwen-72b, etc.)'
    )
    parser.add_argument(
        '--llm_config', type=str, default=None,
        help='Path to custom LLM config JSON file (overrides --model)'
    )

    # API configuration (for custom endpoints)
    parser.add_argument(
        '--base_url', type=str, default=None,
        help='Custom API base URL (overrides model default)'
    )
    parser.add_argument(
        '--api_key_env', type=str, default=None,
        help='Environment variable name for API key (overrides model default)'
    )

    # Path settings
    parser.add_argument(
        '--experiments_dir', type=str, default='./experiments',
        help='Directory to save experiment data'
    )
    parser.add_argument(
        '--policy_path', type=str, default='./policies/policy.json',
        help='Path to policy file'
    )

    # Logging
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --list_models
    if args.list_models:
        print("\nAvailable predefined LLM models:")
        print("=" * 60)
        models = list_available_models()
        for name, desc in sorted(models.items()):
            print(f"  {name:20s} -> {desc}")
        print("=" * 60)
        print("\nUsage examples:")
        print("  python train_pistonball.py --model gpt-4o")
        print("  python train_pistonball.py --model gemini-pro")
        print("  python train_pistonball.py --model llama-3.1-70b")
        print("\nFor custom models, create a JSON config file and use --llm_config")
        sys.exit(0)

    # Load config from file or create from args
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = TrainingConfig.from_json(args.config)
        # Allow overriding start_iteration from command line
        if args.start_iteration > 0:
            config.start_iteration = args.start_iteration
    else:
        # Create directories
        experiments_dir = Path(args.experiments_dir)
        policy_dir = Path(args.policy_path).parent
        experiments_dir.mkdir(parents=True, exist_ok=True)
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Build llm_config if custom parameters provided
        llm_config = None
        if args.llm_config:
            # Use provided LLM config file path
            llm_config = args.llm_config
        elif args.base_url or args.api_key_env:
            # Build inline config with custom parameters
            llm_config = {
                'name': args.model,
                'model_string': args.model,
            }
            if args.base_url:
                llm_config['base_url'] = args.base_url
            if args.api_key_env:
                llm_config['api_key_env_var'] = args.api_key_env

        # Create configuration
        config = TrainingConfig(
            exp_name=args.exp_name,
            paradigm=args.paradigm,
            policy_path=args.policy_path,
            experiments_dir=str(experiments_dir),
            num_iterations=args.num_iterations,
            trajectories_per_iteration=args.trajectories_per_iteration,
            model=args.model,
            llm_config=llm_config,
            num_pistons=args.num_pistons,
            max_cycles=args.max_cycles,
            agent_type=args.agent_type,
            action_mode=args.action_mode,
            rule_k=args.rule_k,
            rule_tau=args.rule_tau,
            log_level=args.log_level,
            start_iteration=args.start_iteration,
        )

    # Save configuration
    config_path = config.experiments_dir / config.exp_name / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_json(str(config_path))
    print(f"Configuration saved to: {config_path}")

    # Get LLM config for display
    llm_cfg = config.get_llm_config()

    # Create and run trainer
    print(f"\n{'='*60}")
    print(f"Starting Pistonball Training")
    print(f"{'='*60}")
    print(f"Paradigm: {config.paradigm}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Pistons: {config.num_pistons}")
    print(f"Agent Type: {config.agent_type}")
    print(f"Action Mode: {config.action_mode}")
    if config.agent_type == 'rule':
        print(f"Rule parameters: k={config.rule_k}, tau={config.rule_tau}")
    print(f"Model: {config.get_model_display_name()}")
    if llm_cfg.base_url:
        print(f"  Base URL: {llm_cfg.base_url}")
        print(f"  API Key Env: {llm_cfg.api_key_env_var}")
    print(f"{'='*60}\n")

    trainer = MonteCarloTrainer(config)
    all_stats = trainer.train_full()

    # Print final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total iterations: {len(all_stats)}")
    print(f"Average rewards per iteration:")
    for i, stats in enumerate(all_stats):
        print(f"  Iteration {i}: {stats['avg_reward']:.2f}")
    print(f"\nTotal cost: ${sum(s['cost_usd'] for s in all_stats):.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
