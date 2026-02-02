"""Policy checkpoint management module for Pistonball"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class PolicyManager:
    """Manage policy checkpoints and versioning for Pistonball

    Policy format (all agents active):
    {
        "current_iteration": 0,
        "num_agents": 20,
        "policies": {
            "iteration_0": {
                "timestamp": "...",
                "agent_policies": {
                    "piston_0": {"policy": "..."},
                    "piston_1": {"policy": "..."},
                    ...
                },
                "training_stats": null
            }
        }
    }
    """

    def __init__(self, policy_path: Path, num_agents: int = 20):
        """
        Initialize policy manager

        Args:
            policy_path: Path to policy.json file
            num_agents: Number of agents (pistons) in the environment
        """
        self.policy_path = Path(policy_path)
        self.num_agents = num_agents
        self.policy_data = self._load_policy()
        self.current_iteration = self.policy_data.get('current_iteration', 0)

    def _load_policy(self) -> Dict:
        """Load policy.json file or create default"""
        if not self.policy_path.exists():
            # Create default policy file
            default_policy = self._get_default_policy()
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.policy_path, 'w') as f:
                json.dump(default_policy, f, indent=2)
            return default_policy

        with open(self.policy_path) as f:
            return json.load(f)

    def _get_default_policy(self) -> Dict:
        """Get default policy structure for Pistonball with per-agent policies"""
        # Default policy template for each agent
        default_policy_template = """The piston adjusts its vertical position cooperatively to push the ball toward the left boundary."""

        # Generate per-agent policies
        agent_policies = {}
        for i in range(self.num_agents):
            agent_name = f"piston_{i}"
            agent_policies[agent_name] = {
                "policy": f"{default_policy_template}"
            }

        return {
            "current_iteration": 0,
            "num_agents": self.num_agents,
            "policies": {
                "iteration_0": {
                    "timestamp": datetime.now().isoformat(),
                    "agent_policies": agent_policies,
                    "training_stats": None
                }
            }
        }

    def _save_policy(self):
        """Save policy data to file"""
        with open(self.policy_path, 'w') as f:
            json.dump(self.policy_data, f, indent=2)

    def get_current_policies(self) -> Dict[str, str]:
        """
        Get policies for current iteration

        Returns:
            Dict mapping agent_name -> policy_text
            e.g., {"piston_0": "policy...", "piston_1": "policy...", ...}
        """
        iter_key = f"iteration_{self.current_iteration}"

        if iter_key not in self.policy_data['policies']:
            raise ValueError(f"No policy found for {iter_key}")

        policies = self.policy_data['policies'][iter_key]

        # New format: agent_policies
        if 'agent_policies' in policies:
            return {
                agent_name: agent_data['policy']
                for agent_name, agent_data in policies['agent_policies'].items()
            }

        # Legacy format compatibility: convert team_policy to all agents
        if 'team_policy' in policies:
            team_policy = policies['team_policy']['policy']
            return {
                f"piston_{i}": team_policy
                for i in range(self.num_agents)
            }

        raise ValueError(f"Invalid policy format for {iter_key}")

    def save_updated_policies(
        self,
        agent_policies: Dict[str, str],
        stats: Dict = None
    ):
        """
        Save optimized policies for next iteration

        Args:
            agent_policies: Dict mapping agent_name -> optimized policy text
                e.g., {"piston_0": "policy...", "piston_1": "policy...", ...}
            stats: Training statistics
        """
        new_iter = self.current_iteration + 1
        iter_key = f"iteration_{new_iter}"

        # Create new iteration entry with per-agent policies
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_policies": {
                agent_name: {"policy": policy}
                for agent_name, policy in agent_policies.items()
            },
            "training_stats": stats
        }

        self.policy_data['policies'][iter_key] = new_entry

        # Update current iteration pointer
        self.policy_data['current_iteration'] = new_iter

        # Update num_agents if needed
        self.policy_data['num_agents'] = len(agent_policies)

        # Save to file
        self._save_policy()

        # Update internal state
        self.current_iteration = new_iter

        print(f"\nSaved updated policies to iteration_{new_iter} ({len(agent_policies)} agents)")

    def get_iteration_count(self) -> int:
        """
        Get total number of iterations

        Returns:
            Number of iterations
        """
        return len(self.policy_data.get('policies', {}))

    def load_specific_iteration(self, iteration: int) -> Dict[str, str]:
        """
        Load policies from a specific iteration

        Args:
            iteration: Iteration number

        Returns:
            Dict mapping agent_name -> policy_text
        """
        iter_key = f"iteration_{iteration}"

        if iter_key not in self.policy_data['policies']:
            raise ValueError(f"No policy found for iteration {iteration}")

        policies = self.policy_data['policies'][iter_key]

        # New format: agent_policies
        if 'agent_policies' in policies:
            return {
                agent_name: agent_data['policy']
                for agent_name, agent_data in policies['agent_policies'].items()
            }

        # Legacy format compatibility: convert team_policy to all agents
        if 'team_policy' in policies:
            team_policy = policies['team_policy']['policy']
            return {
                f"piston_{i}": team_policy
                for i in range(self.num_agents)
            }

        raise ValueError(f"Invalid policy format for {iter_key}")

    def get_training_history(self) -> List[Dict]:
        """
        Get training history across all iterations

        Returns:
            List of training stats dictionaries
        """
        history = []
        for iter_key in sorted(self.policy_data.get('policies', {}).keys()):
            iter_data = self.policy_data['policies'][iter_key]
            if iter_data.get('training_stats'):
                history.append({
                    'iteration': iter_key,
                    'timestamp': iter_data.get('timestamp'),
                    **iter_data['training_stats']
                })
        return history

    def reset_to_iteration(self, iteration: int):
        """
        Reset current iteration pointer to a specific iteration

        Args:
            iteration: Target iteration number
        """
        iter_key = f"iteration_{iteration}"
        if iter_key not in self.policy_data['policies']:
            raise ValueError(f"No policy found for iteration {iteration}")

        self.policy_data['current_iteration'] = iteration
        self.current_iteration = iteration
        self._save_policy()

        print(f"Reset to iteration {iteration}")
