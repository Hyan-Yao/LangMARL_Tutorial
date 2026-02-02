#!/usr/bin/env python3
"""
Test script to verify the rule-based policy for Pistonball.

This script runs a single episode with the rule-based policy to verify
that the scoring function and action quantization work correctly.

Usage:
    python test_rule_policy.py --agent_type rule --rule_k 0.1 --rule_tau 4.0
"""

import argparse
import numpy as np
from pathlib import Path

from training.episode_generator import EpisodeGenerator, PistonballObservationFormatter


def test_rule_policy(
    num_pistons: int = 20,
    max_cycles: int = 125,
    agent_type: str = "rule",
    rule_k: float = 0.1,
    rule_tau: float = 4.0,
    verbose: bool = True,
):
    """
    Test the rule-based policy by running a single episode.

    Args:
        num_pistons: Number of pistons
        max_cycles: Maximum cycles per episode
        agent_type: 'rule' (formula-based) or 'llm' (LLM-based)
        rule_k: Slope coefficient for scoring function
        rule_tau: Dead-zone threshold
        verbose: Print detailed information
    """
    print("="*60)
    print("Rule-Based Policy Test")
    print("="*60)
    print(f"Agent type: {agent_type}")
    print(f"Parameters: k={rule_k}, tau={rule_tau}")
    print(f"Pistons: {num_pistons}, Max cycles: {max_cycles}")
    print("="*60)

    # Create episode generator
    generator = EpisodeGenerator(
        num_pistons=num_pistons,
        max_cycles=max_cycles,
        experiments_dir=Path("./test_experiments"),
        exp_name="rule_policy_test",
        agent_type=agent_type,
        action_mode="discrete",
        rule_k=rule_k,
        rule_tau=rule_tau,
        random_drop=False,  # Fixed initial position for reproducibility
        pistons_start_low=True,
    )

    # Reset environment
    obs_dict, info = generator.env.reset(seed=42)

    # Get initial state
    formatter = PistonballObservationFormatter()
    state = formatter.get_env_state(generator.env)
    ball_x = state["ball_position"][0]
    print(f"\nInitial ball x position: {ball_x:.1f}")

    # Run episode and collect statistics
    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}  # 0=retract_down, 1=stay, 2=push_up
    scores_log = []

    for t in range(max_cycles):
        # Get state for logging
        state = formatter.get_env_state(generator.env)
        ball_x = state["ball_position"][0]

        # Collect actions for all agents
        actions = {}
        step_scores = []

        for agent_name in generator.env.possible_agents:
            if agent_name not in obs_dict:
                continue

            agent_idx = generator.env.possible_agents.index(agent_name)

            # Use rule-based action
            action = generator._get_rule_based_action(obs_dict[agent_name], agent_idx)

            actions[agent_name] = action
            action_counts[action] += 1

            # Log score for detailed analysis (first few pistons)
            if verbose and t < 5 and agent_idx < 5:
                piston_x = state["pistons"][agent_idx]["position"][0]
                piston_y = state["pistons"][agent_idx]["position"][1]
                step_scores.append((agent_idx, piston_x, ball_x, action))

        # Log detailed scores for first few timesteps
        if verbose and t < 5:
            print(f"\n--- Timestep {t} (ball_x={ball_x:.1f}) ---")
            for idx, px, bx, act in step_scores:
                action_name = {0: "retract_down", 1: "stay", 2: "push_up"}[act]
                print(f"  Piston {idx}: x={px:.1f}, action={action_name}")

        # Convert actions to environment format
        env_actions = {}
        for name, act in actions.items():
            env_actions[name] = act

        # Step environment
        obs_dict, rewards, terms, truncs, infos = generator.env.step(env_actions)

        # Track reward
        step_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
        total_reward += step_reward

        # Check termination
        if any(terms.values()) or any(truncs.values()):
            print(f"\nEpisode terminated at step {t}")
            break

    # Final state
    state = formatter.get_env_state(generator.env)
    final_ball_x = state["ball_position"][0]

    # Print summary
    print("\n" + "="*60)
    print("Episode Summary")
    print("="*60)
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final ball x position: {final_ball_x:.1f}")
    print(f"Ball moved: {ball_x - final_ball_x:.1f} pixels to the left")
    print(f"\nAction distribution (0=retract_down, 1=stay, 2=push_up):")
    total_actions = sum(action_counts.values())
    for act, count in action_counts.items():
        name = {0: "retract_down", 1: "stay", 2: "push_up"}[act]
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Test rule-based policy")
    parser.add_argument("--num_pistons", type=int, default=20)
    parser.add_argument("--max_cycles", type=int, default=125)
    parser.add_argument("--agent_type", type=str, default="rule",
                        choices=["rule", "llm"])
    parser.add_argument("--rule_k", type=float, default=0.1)
    parser.add_argument("--rule_tau", type=float, default=4.0)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    test_rule_policy(
        num_pistons=args.num_pistons,
        max_cycles=args.max_cycles,
        agent_type=args.agent_type,
        rule_k=args.rule_k,
        rule_tau=args.rule_tau,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
