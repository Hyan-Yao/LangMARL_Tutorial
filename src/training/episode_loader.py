"""Episode loading and validation module for Pistonball"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class EpisodeLoader:
    """Load and validate episode data from experiments directory"""

    def __init__(self, experiments_dir: Path):
        """
        Initialize episode loader

        Args:
            experiments_dir: Path to experiments directory
        """
        self.experiments_dir = Path(experiments_dir)

    def load_episodes(
        self,
        exp_name: str,
        iteration: int,
        num_episodes: Optional[int] = None
    ) -> List[Dict]:
        """
        Load episodes from an experiment iteration

        Args:
            exp_name: Experiment name
            iteration: Iteration number
            num_episodes: Number of episodes to load (None = all)

        Returns:
            List of episode dictionaries
        """
        iter_dir = self.experiments_dir / exp_name / f"iteration_{iteration}"

        if not iter_dir.exists():
            raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")

        episodes = []
        episode_dirs = sorted(iter_dir.glob("episode_*"))

        if num_episodes is not None:
            episode_dirs = episode_dirs[:num_episodes]

        for episode_dir in episode_dirs:
            episode_file = episode_dir / "episode.json"
            if episode_file.exists():
                with open(episode_file) as f:
                    episodes.append(json.load(f))

        return episodes

    def validate_episodes(self, episodes: List[Dict]) -> bool:
        """
        Validate episode data structure

        Args:
            episodes: List of episode dictionaries

        Returns:
            True if all episodes are valid
        """
        required_fields = ['episode_id', 'total_reward', 'transitions']

        for i, ep in enumerate(episodes):
            for field in required_fields:
                if field not in ep:
                    print(f"Episode {i} missing required field: {field}")
                    return False

            # Validate transitions
            if not isinstance(ep['transitions'], list):
                print(f"Episode {i}: transitions must be a list")
                return False

            if len(ep['transitions']) == 0:
                print(f"Episode {i}: no transitions recorded")
                return False

        return True

    def get_episode_stats(self, episodes: List[Dict]) -> Dict:
        """
        Get statistics about a batch of episodes

        Args:
            episodes: List of episode dictionaries

        Returns:
            Statistics dictionary
        """
        if not episodes:
            return {"error": "No episodes provided"}

        rewards = [ep['total_reward'] for ep in episodes]
        lengths = [len(ep['transitions']) for ep in episodes]

        return {
            "num_episodes": len(episodes),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_transitions": sum(lengths)
        }

    def load_latest_iteration(self, exp_name: str) -> tuple:
        """
        Load episodes from the latest iteration

        Args:
            exp_name: Experiment name

        Returns:
            (iteration_number, episodes)
        """
        exp_dir = self.experiments_dir / exp_name

        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        # Find all iteration directories
        iter_dirs = sorted(exp_dir.glob("iteration_*"))

        if not iter_dirs:
            raise FileNotFoundError(f"No iterations found in {exp_dir}")

        # Get latest iteration number
        latest_dir = iter_dirs[-1]
        iteration = int(latest_dir.name.split("_")[1])

        episodes = self.load_episodes(exp_name, iteration)

        return iteration, episodes

    def count_episodes(self, exp_name: str, iteration: int) -> int:
        """
        Count number of complete episodes in an iteration.

        Only counts episode directories that contain a valid episode.json file,
        matching the behavior of load_episodes.

        Args:
            exp_name: Experiment name
            iteration: Iteration number

        Returns:
            Number of complete episodes with episode.json files
        """
        iter_dir = self.experiments_dir / exp_name / f"iteration_{iteration}"

        if not iter_dir.exists():
            return 0

        count = 0
        for episode_dir in iter_dir.glob("episode_*"):
            if (episode_dir / "episode.json").exists():
                count += 1
        return count

    def list_experiments(self) -> List[str]:
        """
        List all experiments in the experiments directory

        Returns:
            List of experiment names
        """
        if not self.experiments_dir.exists():
            return []

        return [d.name for d in self.experiments_dir.iterdir() if d.is_dir()]

    def get_experiment_info(self, exp_name: str) -> Dict:
        """
        Get information about an experiment

        Args:
            exp_name: Experiment name

        Returns:
            Experiment information dictionary
        """
        exp_dir = self.experiments_dir / exp_name

        if not exp_dir.exists():
            return {"error": f"Experiment not found: {exp_name}"}

        # Count iterations
        iter_dirs = list(exp_dir.glob("iteration_*"))
        num_iterations = len(iter_dirs)

        # Get episode counts per iteration
        episode_counts = {}
        total_episodes = 0
        for iter_dir in sorted(iter_dirs):
            iter_name = iter_dir.name
            count = len(list(iter_dir.glob("episode_*")))
            episode_counts[iter_name] = count
            total_episodes += count

        return {
            "experiment_name": exp_name,
            "num_iterations": num_iterations,
            "total_episodes": total_episodes,
            "episodes_per_iteration": episode_counts
        }
