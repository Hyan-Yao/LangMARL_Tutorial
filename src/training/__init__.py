"""
Monte Carlo Training Module for Pistonball Multi-Agent Environment

This module provides modular components for training multi-agent policies using
Monte Carlo trajectory evaluation and TextGrad optimization in the Pistonball
environment from PettingZoo.

Key Components:
- TrainingConfig: Configuration management for training
- TrajectoryFormatter: Format episodes for LLM evaluation
- ParadigmEvaluator: Generate evaluation prompts for three paradigms
- PromptLoader: Load evaluation prompts from external files
- PolicyOptimizer: TextGrad optimization wrapper
- TokenTracker: Token usage tracking and cost estimation
- EpisodeLoader: Load episode data from experiments
- EpisodeGenerator: Generate episodes by running Pistonball simulations
- PolicyManager: Policy checkpoint management
- MonteCarloTrainer: Main training orchestrator

Supported Training Paradigms:
1. Independent Learning: Each piston group has a separate critic
2. Central Global Reward: Shared critic for overall team performance
3. Central Credit Assignment: Evaluate individual contributions to team success
"""

from .config import TrainingConfig
from .llm_config import (
    LLMConfig,
    get_llm_config,
    save_llm_config,
    list_available_models,
    PREDEFINED_MODELS,
)
from .trajectory import TrajectoryFormatter
from .evaluators import ParadigmEvaluator, PromptLoader
from .optimizer import PolicyOptimizer
from .token_tracker import TokenTracker
from .episode_loader import EpisodeLoader
from .episode_generator import EpisodeGenerator, estimate_tokens
from .policy_manager import PolicyManager
from .trainer import MonteCarloTrainer

__all__ = [
    'TrainingConfig',
    'LLMConfig',
    'get_llm_config',
    'save_llm_config',
    'list_available_models',
    'PREDEFINED_MODELS',
    'TrajectoryFormatter',
    'ParadigmEvaluator',
    'PromptLoader',
    'PolicyOptimizer',
    'TokenTracker',
    'EpisodeLoader',
    'EpisodeGenerator',
    'estimate_tokens',
    'PolicyManager',
    'MonteCarloTrainer',
]

__version__ = '0.1.0'
