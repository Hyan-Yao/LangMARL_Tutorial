"""Training configuration module for Pistonball environment"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .llm_config import LLMConfig, get_llm_config, PREDEFINED_MODELS


@dataclass
class TrainingConfig:
    """Configuration for Monte Carlo training on Pistonball"""

    # Experiment configuration
    exp_name: str
    paradigm: str  # 'independent' | 'central_global' | 'central_credit'

    # Path configuration
    policy_path: str
    experiments_dir: str

    # Training parameters
    num_iterations: int = 5
    trajectories_per_iteration: int = 3

    # Model configuration - supports both simple name and full LLM config
    model: str = "gpt-4o-mini"

    # LLM configuration (optional, takes precedence over 'model' if provided)
    # Can be a predefined model name, a path to JSON config, or inline config dict
    llm_config: Optional[Dict[str, Any]] = None

    # Separate LLM configs for episode generation and textgrad (optional)
    # If not provided, falls back to llm_config/model
    episode_llm_config: Optional[Dict[str, Any]] = None  # For episode generation (agent actions)
    textgrad_llm_config: Optional[Dict[str, Any]] = None  # For TextGrad backward engine

    # Pistonball environment parameters
    num_pistons: int = 20
    max_cycles: int = 125
    frame_size: Tuple[int, int] = (64, 64)
    stack_size: int = 4
    continuous: bool = False

    # Agent types for the pistons
    # 'llm': LLM parses actions from prompts
    # 'rule': Formula-based control (no LLM)
    agent_type: str = "llm"  # 'llm' | 'rule'

    # Action mode for the pistons
    # 'discrete': 3 actions (0-retract down, 1-stay, 2-push up)
    # 'continuous': continuous value in [-1, 1]
    action_mode: str = "discrete"  # 'discrete' | 'continuous'

    # Rule-based scoring function parameters (for agent_type 'rule')
    # S_i = k * (x_i - x_b) - (h_i - 0.5 * (h_{i-1} + h_{i+1}))
    # rule_k: slope coefficient controlling "right-high, left-low" trend strength
    # rule_tau: dead-zone threshold (pixels) to suppress action jitter
    rule_k: float = 0.1
    rule_tau: float = 4.0

    # Logging configuration
    log_level: str = "INFO"

    # Optional: current iteration (for resuming)
    start_iteration: int = 0

    def __post_init__(self):
        """Validate configuration"""
        # Validate paradigm
        valid_paradigms = ['independent', 'central_global', 'central_credit']
        if self.paradigm not in valid_paradigms:
            raise ValueError(
                f"Invalid paradigm '{self.paradigm}'. "
                f"Must be one of {valid_paradigms}"
            )

        # Validate trajectories_per_iteration
        if self.trajectories_per_iteration < 1:
            raise ValueError(
                "trajectories_per_iteration must be at least 1"
            )

        # Convert paths to Path objects
        self.policy_path = Path(self.policy_path)
        self.experiments_dir = Path(self.experiments_dir)

        # Convert frame_size to tuple if it's a list
        if isinstance(self.frame_size, list):
            self.frame_size = tuple(self.frame_size)

        # Initialize LLM config cache
        self._resolved_llm_config: Optional[LLMConfig] = None
        self._resolved_episode_llm_config: Optional[LLMConfig] = None
        self._resolved_textgrad_llm_config: Optional[LLMConfig] = None

    def get_llm_config(self) -> LLMConfig:
        """
        Get the resolved LLM configuration.

        Priority:
        1. If llm_config dict is provided, use it directly
        2. If llm_config is a string path to JSON file, load it
        3. If model name matches a predefined model, use that
        4. Otherwise, create a basic config using model as model_string

        Returns:
            LLMConfig object
        """
        if self._resolved_llm_config is not None:
            return self._resolved_llm_config

        if self.llm_config is not None:
            if isinstance(self.llm_config, dict):
                # Inline config dict
                self._resolved_llm_config = LLMConfig.from_dict(self.llm_config)
            elif isinstance(self.llm_config, str):
                # Path to JSON file or predefined model name
                self._resolved_llm_config = get_llm_config(self.llm_config)
            else:
                raise ValueError(
                    f"Invalid llm_config type: {type(self.llm_config)}. "
                    "Expected dict or str."
                )
        elif self.model in PREDEFINED_MODELS:
            # Use predefined model
            self._resolved_llm_config = PREDEFINED_MODELS[self.model]
        else:
            # Create basic config using model name
            self._resolved_llm_config = LLMConfig(
                name=self.model,
                model_string=self.model,
            )

        return self._resolved_llm_config

    def get_episode_llm_config(self) -> LLMConfig:
        """
        Get the LLM configuration for episode generation.

        Priority:
        1. If episode_llm_config is provided, use it
        2. Otherwise, fall back to get_llm_config()

        Returns:
            LLMConfig object for episode generation
        """
        if self._resolved_episode_llm_config is not None:
            return self._resolved_episode_llm_config

        if self.episode_llm_config is not None:
            if isinstance(self.episode_llm_config, dict):
                self._resolved_episode_llm_config = LLMConfig.from_dict(self.episode_llm_config)
            elif isinstance(self.episode_llm_config, str):
                self._resolved_episode_llm_config = get_llm_config(self.episode_llm_config)
            else:
                raise ValueError(
                    f"Invalid episode_llm_config type: {type(self.episode_llm_config)}. "
                    "Expected dict or str."
                )
        else:
            # Fall back to default llm_config
            self._resolved_episode_llm_config = self.get_llm_config()

        return self._resolved_episode_llm_config

    def get_textgrad_llm_config(self) -> LLMConfig:
        """
        Get the LLM configuration for TextGrad backward engine.

        Priority:
        1. If textgrad_llm_config is provided, use it
        2. Otherwise, fall back to get_llm_config()

        Returns:
            LLMConfig object for TextGrad
        """
        if self._resolved_textgrad_llm_config is not None:
            return self._resolved_textgrad_llm_config

        if self.textgrad_llm_config is not None:
            if isinstance(self.textgrad_llm_config, dict):
                self._resolved_textgrad_llm_config = LLMConfig.from_dict(self.textgrad_llm_config)
            elif isinstance(self.textgrad_llm_config, str):
                self._resolved_textgrad_llm_config = get_llm_config(self.textgrad_llm_config)
            else:
                raise ValueError(
                    f"Invalid textgrad_llm_config type: {type(self.textgrad_llm_config)}. "
                    "Expected dict or str."
                )
        else:
            # Fall back to default llm_config
            self._resolved_textgrad_llm_config = self.get_llm_config()

        return self._resolved_textgrad_llm_config

    def get_model_display_name(self) -> str:
        """Get display name for the model being used."""
        llm_cfg = self.get_llm_config()
        if llm_cfg.base_url:
            return f"{llm_cfg.name} ({llm_cfg.model_string} via {llm_cfg.base_url})"
        return f"{llm_cfg.name} ({llm_cfg.model_string})"

    @classmethod
    def from_json(cls, config_path: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file

        Args:
            config_path: Path to JSON configuration file

        Returns:
            TrainingConfig object
        """
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, output_path: str):
        """
        Save configuration to JSON file

        Args:
            output_path: Path to save configuration
        """
        # Convert Path objects to strings for JSON serialization
        data = {
            'exp_name': self.exp_name,
            'paradigm': self.paradigm,
            'policy_path': str(self.policy_path),
            'experiments_dir': str(self.experiments_dir),
            'num_iterations': self.num_iterations,
            'trajectories_per_iteration': self.trajectories_per_iteration,
            'model': self.model,
            'num_pistons': self.num_pistons,
            'max_cycles': self.max_cycles,
            'frame_size': list(self.frame_size),
            'stack_size': self.stack_size,
            'continuous': self.continuous,
            'agent_type': self.agent_type,
            'action_mode': self.action_mode,
            'rule_k': self.rule_k,
            'rule_tau': self.rule_tau,
            'log_level': self.log_level,
            'start_iteration': self.start_iteration,
        }

        # Include llm_config if provided
        if self.llm_config is not None:
            if isinstance(self.llm_config, dict):
                data['llm_config'] = self.llm_config
            elif isinstance(self.llm_config, str):
                data['llm_config'] = self.llm_config

        # Include episode_llm_config if provided
        if self.episode_llm_config is not None:
            if isinstance(self.episode_llm_config, dict):
                data['episode_llm_config'] = self.episode_llm_config
            elif isinstance(self.episode_llm_config, str):
                data['episode_llm_config'] = self.episode_llm_config

        # Include textgrad_llm_config if provided
        if self.textgrad_llm_config is not None:
            if isinstance(self.textgrad_llm_config, dict):
                data['textgrad_llm_config'] = self.textgrad_llm_config
            elif isinstance(self.textgrad_llm_config, str):
                data['textgrad_llm_config'] = self.textgrad_llm_config

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """List all available predefined LLM models."""
        from .llm_config import list_available_models
        return list_available_models()
