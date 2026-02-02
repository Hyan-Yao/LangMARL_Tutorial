"""Paradigm evaluators for Monte Carlo training on Pistonball"""

import json
from pathlib import Path
from typing import Optional


class PromptLoader:
    """
    Loader for evaluation prompts from external files.
    Supports loading game contexts, role descriptions, and evaluation templates.
    """

    # Default path to evaluation prompts directory
    DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts" / "evaluation"

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt loader.

        Args:
            prompts_dir: Path to evaluation prompts directory.
                         If None, uses default path.
        """
        self.prompts_dir = Path(prompts_dir) if prompts_dir else self.DEFAULT_PROMPTS_DIR
        self._cache = {}

    def _load_json(self, filepath: Path) -> dict:
        """Load and cache JSON file."""
        str_path = str(filepath)
        if str_path not in self._cache:
            with open(filepath, 'r') as f:
                self._cache[str_path] = json.load(f)
        return self._cache[str_path]

    def load_game_context(self, game_name: str = "pistonball") -> str:
        """
        Load game context for Pistonball.

        Args:
            game_name: Game name (default 'pistonball')

        Returns:
            Game context string
        """
        # Try game-specific file first, fall back to default
        game_file = self.prompts_dir / "game_contexts" / f"{game_name}.json"
        if not game_file.exists():
            game_file = self.prompts_dir / "game_contexts" / "default.json"

        if not game_file.exists():
            raise ValueError("Return no game context file exists")

        data = self._load_json(game_file)
        return data.get("game_context", "")


    def load_role_descriptions(self) -> dict:
        """
        Load role descriptions for pistons.

        Returns:
            Dict containing role descriptions
        """
        roles_file = self.prompts_dir / "role_descriptions.json"
        if not roles_file.exists():
            # Return default roles
            role_desc = "You are a cooperative piston agent responsible for adjusting your vertical position based on local observations to help the team collectively push the ball toward the left boundary."
            return {
                "roles": {
                    "left_pistons": {
                        "short": role_desc,
                        "detailed": role_desc,
                        "dependencies": role_desc
                    },
                    "middle_pistons": {
                        "short": role_desc,
                        "detailed": role_desc,
                        "dependencies": role_desc
                    },
                    "right_pistons": {
                        "short": role_desc,
                        "detailed": role_desc,
                        "dependencies": role_desc
                    }
                },
                "role_context_summary": role_desc
            }

        return self._load_json(roles_file)

    def load_evaluation_template(self, paradigm: str) -> str:
        """
        Load evaluation template for a specific paradigm.

        Args:
            paradigm: Paradigm name ('independent', 'central_global', 'central_credit')

        Returns:
            Template string
        """
        template_file = self.prompts_dir / "templates" / f"{paradigm}.json"
        if not template_file.exists():
            # Return default templates
            raise ValueError("No template files found.")

        data = self._load_json(template_file)
        return data.get("template", "")

    def clear_cache(self):
        """Clear the loaded prompts cache."""
        self._cache.clear()


class ParadigmEvaluator:
    """
    Evaluator for three training paradigms in Pistonball:
    - Independent Learning: Each individual piston evaluated separately
    - Central Global Reward: All pistons evaluated as one team with shared feedback
    - Central Credit Assignment: Evaluate each piston's individual contribution

    All pistons are active and each has its own policy to optimize.
    Loads prompts from external files for easy customization.
    """

    def __init__(
        self,
        paradigm: str,
        num_pistons: int = 20,
        prompts_dir: Optional[Path] = None
    ):
        """
        Initialize evaluator

        Args:
            paradigm: 'independent' | 'central_global' | 'central_credit'
            num_pistons: Number of pistons in the environment
            prompts_dir: Optional custom path to prompts directory
        """
        valid_paradigms = ['independent', 'central_global', 'central_credit']
        if paradigm not in valid_paradigms:
            raise ValueError(
                f"Invalid paradigm '{paradigm}'. Must be one of {valid_paradigms}"
            )

        self.paradigm = paradigm
        self.num_pistons = num_pistons

        # Initialize prompt loader
        self.prompt_loader = PromptLoader(prompts_dir)

        # Load prompts
        self._load_prompts()

        # Define piston groups for reference (left/middle/right regions)
        self._define_piston_groups()

        # Create list of all agent names
        self.all_agents = [f"piston_{i}" for i in range(num_pistons)]

    def _define_piston_groups(self):
        """Define piston groups based on position."""
        third = self.num_pistons // 3
        self.piston_groups = {
            'left': [f'piston_{i}' for i in range(third)],
            'middle': [f'piston_{i}' for i in range(third, 2*third)],
            'right': [f'piston_{i}' for i in range(2*third, self.num_pistons)],
        }

    def _load_prompts(self):
        """Load all necessary prompts from files."""
        # Load game context
        self.game_context = self.prompt_loader.load_game_context("pistonball")

        # Load role descriptions
        roles_data = self.prompt_loader.load_role_descriptions()
        self.role_descriptions = roles_data.get("roles", {})
        self.role_context_summary = roles_data.get("role_context_summary", "")

        # Load evaluation template for the paradigm
        self.evaluation_template = self.prompt_loader.load_evaluation_template(self.paradigm)

    def reload_prompts(self):
        """Reload prompts from files (useful after modifying prompt files)."""
        self.prompt_loader.clear_cache()
        self._load_prompts()

    def get_agent_group(self, agent_name: str) -> str:
        """
        Determine which group (left/middle/right) an agent belongs to.

        Args:
            agent_name: Agent name like 'piston_0'

        Returns:
            Group name: 'left', 'middle', or 'right'
        """
        for group_name, members in self.piston_groups.items():
            if agent_name in members:
                return group_name
        # Default to middle if not found
        return 'middle'

    def create_evaluation_prompt(
        self,
        trajectory: str,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Create evaluation prompt based on paradigm

        Args:
            trajectory: Formatted trajectory string
            agent_id: Agent name (e.g., 'piston_0') for independent paradigm,
                      None for central_global, not used for central_credit

        Returns:
            Complete LLM evaluation prompt
        """
        if self.paradigm == "independent":
            if agent_id is None:
                raise ValueError("agent_id (agent name) required for independent learning")
            return self._create_independent_prompt(trajectory, agent_id)
        elif self.paradigm == "central_global":
            return self._create_global_prompt(trajectory)
        elif self.paradigm == "central_credit":
            return self._create_credit_prompt(trajectory)

    def _create_independent_prompt(self, trajectory: str, agent_name: str) -> str:
        """
        Paradigm 1: Independent Learning
        Each individual piston has a separate critic evaluating their performance.

        Args:
            trajectory: Formatted trajectory string for this agent
            agent_name: Individual agent name (e.g., 'piston_0')

        Returns:
            Evaluation prompt for this specific agent
        """
        # Determine which group this agent belongs to
        group = self.get_agent_group(agent_name)
        role_key = f"{group}_pistons"

        # Get role description based on position
        role_desc = self.role_descriptions.get(role_key, {}).get(
            "detailed",
            f"{group.capitalize()} region piston"
        )

        # Extract piston index for position context
        try:
            piston_idx = int(agent_name.split('_')[1])
            position_desc = f"{agent_name} (position {piston_idx}/{self.num_pistons-1}, {group} region)"
        except (IndexError, ValueError):
            position_desc = agent_name

        # Format the template
        prompt = f"{self.game_context}\n\n{self.evaluation_template}".format(
            piston_group=position_desc,
            role_description=role_desc,
            trajectory=trajectory
        )

        return prompt

    def _create_global_prompt(self, trajectory: str) -> str:
        """
        Paradigm 2: Central Training with Global Reward
        Shared critic evaluating overall team performance.
        """
        prompt = f"{self.game_context}\n\n{self.evaluation_template}".format(
            trajectory=trajectory,
            role_context=self.role_context_summary
        )

        return prompt

    def _create_credit_prompt(self, trajectory: str) -> str:
        """
        Paradigm 3: Central Training with Credit Assignment
        Shared critic evaluating EACH individual piston's contribution to team success.

        This creates a prompt that asks for separate evaluation for each piston,
        enabling per-agent gradient assignment.
        """
        # Build credit-specific role context with all pistons
        left_role = self.role_descriptions.get("left_pistons", {})
        middle_role = self.role_descriptions.get("middle_pistons", {})
        right_role = self.role_descriptions.get("right_pistons", {})

        # List all pistons with their group assignments
        piston_groups_desc = []
        for group_name, pistons in self.piston_groups.items():
            role_key = f"{group_name}_pistons"
            role_info = self.role_descriptions.get(role_key, {})
            piston_list = ", ".join(pistons)
            piston_groups_desc.append(
                f"* **{group_name.capitalize()} Pistons** ({piston_list}): "
                f"{role_info.get('short', group_name.capitalize())} - "
                f"{role_info.get('detailed', 'No description')}"
            )

        role_context_credit = "\n".join(piston_groups_desc)
        role_context_credit += f"\n* Total {self.num_pistons} pistons, each needs individual evaluation"

        # Build the expected output format for credit assignment
        # Request per-piston evaluation markers
        piston_eval_format = "\n".join([
            f"[{agent.upper()} EVALUATION]\n- Performance assessment:\n- Specific improvements:"
            for agent in self.all_agents
        ])

        prompt = f"{self.game_context}\n\n{self.evaluation_template}".format(
            trajectory=trajectory,
            role_context_credit=role_context_credit,
            piston_eval_format=piston_eval_format,
            num_pistons=self.num_pistons,
            all_agents=", ".join(self.all_agents)
        )

        return prompt

    def get_piston_groups(self):
        """
        Get the piston group definitions.

        Returns:
            Dictionary mapping group names to piston lists
        """
        return self.piston_groups
