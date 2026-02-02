"""Episode generation module for running Pistonball simulations"""

import os
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from multiprocessing import Pool, cpu_count
from openai import OpenAI

from .llm_config import LLMConfig, get_llm_config

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1


def estimate_tokens(text: str) -> Dict[str, int]:
    """
    Estimate token count from text using word and character count

    Args:
        text: Input text

    Returns:
        Dictionary with word_count, char_count, and estimated_tokens
    """
    word_count = len(text.split())
    char_count = len(text)
    # Rough estimate: 1 token ≈ 4 characters or 0.75 words
    estimated_tokens = max(int(char_count / 4), int(word_count / 0.75))

    return {
        "word_count": word_count,
        "char_count": char_count,
        "estimated_tokens": estimated_tokens
    }


def parse_action_from_response(response, action_mode: str = "discrete"):
    """
    Parse action from LLM response based on action mode.

    Args:
        response: OpenAI API response object
        action_mode: Action space mode - "discrete" (0-2) or "continuous" (-1.0 to 1.0)

    Returns:
        Parsed action value:
        - discrete: int in [0, 1, 2] (0-retract down, 1-stay, 2-push up)
        - continuous: float in [-1.0, 1.0]
    """
    import re

    # Extract text content from response
    try:
        # Handle OpenAI response format
        if hasattr(response, 'output'):
            # New responses API format
            text = ""
            for item in response.output:
                if hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'text'):
                            text += content.text
        elif hasattr(response, 'choices'):
            # Chat completions format
            text = response.choices[0].message.content
        else:
            text = str(response)
    except Exception:
        text = str(response)

    text = text.strip()

    # Default actions for each mode
    default_actions = {
        "discrete": 1,      # stay
        "continuous": 0.0,  # stay
    }

    if action_mode == "discrete":
        # Parse discrete action (0, 1, or 2)
        # 0 - retract down, 1 - stay, 2 - push up
        # Look for a single digit 0-2 at the start of the response or after "Action:"
        match = re.search(r'(?:^|Action:\s*)([0-2])\b', text, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))

        # Try to find any standalone digit 0-2
        match = re.search(r'\b([0-2])\b', text)
        if match:
            return int(match.group(1))

        # Try to parse action names
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['retract', 'down', 'retract_down']):
            return 0
        elif any(keyword in text_lower for keyword in ['stay', 'hold', 'wait']):
            return 1
        elif any(keyword in text_lower for keyword in ['push_up', 'push up', 'push']):
            return 2

        return default_actions["discrete"]

    elif action_mode == "continuous":
        # Parse continuous action value in [-1.0, 1.0]
        # Look for a float value at the start or after "Action:"
        match = re.search(r'(?:^|Action:\s*)([+-]?\d*\.?\d+)', text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = float(match.group(1))
                return max(-1.0, min(1.0, value))  # Clamp to [-1.0, 1.0]
            except ValueError:
                pass

        # Try to find any float in the response
        match = re.search(r'([+-]?\d*\.?\d+)', text)
        if match:
            try:
                value = float(match.group(1))
                return max(-1.0, min(1.0, value))
            except ValueError:
                pass

        # Try to parse descriptive text
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['maximum push', 'full push', 'push up fast']):
            return 1.0
        elif any(keyword in text_lower for keyword in ['moderate push', 'medium push']):
            return 0.5
        elif any(keyword in text_lower for keyword in ['slight push', 'gentle push', 'light push']):
            return 0.3
        elif any(keyword in text_lower for keyword in ['stay', 'hold', 'no movement']):
            return 0.0
        elif any(keyword in text_lower for keyword in ['slight retract', 'gentle retract', 'light retract']):
            return -0.3
        elif any(keyword in text_lower for keyword in ['moderate retract', 'medium retract']):
            return -0.5
        elif any(keyword in text_lower for keyword in ['maximum retract', 'full retract', 'retract fast']):
            return -1.0

        return default_actions["continuous"]

    return default_actions.get(action_mode, 1)


def reset_pistons_to_lowest(env):
    """
    Reset all pistons to their lowest position (maximum y value).

    This function directly manipulates the pymunk physics bodies to set
    all pistons to the lowest position after environment reset.

    Args:
        env: The Pistonball environment (can be wrapped)
    """
    # Get the unwrapped raw environment
    raw = env.unwrapped

    # Calculate the maximum y position (lowest point for pistons)
    # This matches the calculation in pistonball.py
    maximum_piston_y = raw.screen_height - raw.wall_width - (raw.piston_height - raw.piston_head_height)

    # Set each piston to the lowest position
    for piston in raw.pistonList:
        # Keep x position, set y to maximum (lowest point)
        piston.position = (piston.position.x, maximum_piston_y)
        # Reset velocity to zero
        piston.velocity = (0, 0)


def make_env(num_pistons: int, max_cycles: int, frame_size: Tuple[int, int] = (64, 64),
             stack_size: int = 4, continuous: bool = False, render_mode: str = "rgb_array",
             random_drop: bool = True):
    """
    Create a Pistonball environment with preprocessing

    Args:
        num_pistons: Number of pistons in the environment
        max_cycles: Maximum number of cycles per episode
        frame_size: Size of the observation frames
        stack_size: Number of frames to stack
        continuous: Whether to use continuous action space
        render_mode: Render mode for the environment
        random_drop: If True, ball spawns at random x position; if False, ball spawns at x=800

    Returns:
        Preprocessed Pistonball environment
    """
    env = pistonball_v6.parallel_env(
        n_pistons=num_pistons,
        render_mode=render_mode,
        continuous=continuous,
        max_cycles=max_cycles,
        random_drop=random_drop,
    )
    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)
    return env


def _run_episode_worker(args: Tuple) -> Dict:
    """
    Worker function to run a single episode in parallel.
    This is a top-level function required for multiprocessing.

    Args:
        args: Tuple of (episode_id, iteration, config_dict)

    Returns:
        Episode data dictionary
    """
    episode_id, iteration, config = args

    try:
        # Create a temporary EpisodeGenerator instance in this worker process
        generator = EpisodeGenerator(
            num_pistons=config['num_pistons'],
            max_cycles=config['max_cycles'],
            frame_size=config['frame_size'],
            stack_size=config['stack_size'],
            continuous=config['continuous'],
            experiments_dir=config['experiments_dir'],
            exp_name=config['exp_name'],
            agent_type=config['agent_type'],
            gpt_model=config['gpt_model'],
            action_mode=config.get('action_mode', 'discrete'),
            random_drop=config.get('random_drop', True),
            pistons_start_low=config.get('pistons_start_low', False),
            rule_k=config.get('rule_k', 0.1),
            rule_tau=config.get('rule_tau', 4.0),
            llm_config=config.get('llm_config'),
        )

        # Run the episode with GIF and incremental save support
        episode_data = generator._run_single_episode(
            episode_id=episode_id,
            iteration=iteration,
            agent_policies=config.get('agent_policies'),
            save_gif=config.get('save_gif', True),
            gif_fps=config.get('gif_fps', 30),
            incremental_save=config.get('incremental_save', True),
            save_interval=config.get('save_interval', 10),
        )

        return episode_data

    except Exception as e:
        # Convert OpenAI API errors and other unpicklable exceptions to standard exceptions
        # This ensures errors can be properly passed back through multiprocessing
        error_type = type(e).__name__
        error_msg = str(e)

        # Try to extract additional info from OpenAI errors
        if hasattr(e, 'status_code'):
            error_msg = f"[HTTP {e.status_code}] {error_msg}"
        if hasattr(e, 'body') and e.body:
            try:
                error_msg = f"{error_msg} | Body: {e.body}"
            except Exception:
                pass

        raise RuntimeError(
            f"Episode {episode_id} failed with {error_type}: {error_msg}"
        ) from None


class PistonballObservationFormatter:
    """Format Pistonball observations for LLM understanding using environment state"""

    # Discrete action space (3 actions)
    DISCRETE_ACTION_NAMES = {
        2: "push_up",      # Move piston up to push ball left
        1: "stay",         # Hold current position
        0: "retract_down"  # Move piston down/retract
    }

    DISCRETE_ACTION_DESCRIPTIONS = {
        2: ("push_up", "Move piston UP to push the ball toward the LEFT (goal direction). "
            "Use when ball is above you or approaching."),
        1: ("stay", "HOLD current position. Use to maintain contact with ball or wait."),
        0: ("retract_down", "Move piston DOWN/retract. Use to let ball pass or reset position.")
    }

    # Continuous action space description
    CONTINUOUS_ACTION_RANGE = {
        "min": -1.0,  # Maximum downward (retract)
        "max": 1.0,   # Maximum upward (push)
        "description": "Continuous value from -1.0 (full retract) to 1.0 (full push up)"
    }

    # Legacy mapping for backward compatibility
    ACTION_NAMES = DISCRETE_ACTION_NAMES

    @staticmethod
    def get_env_state(env) -> Dict:
        """
        Extract state information directly from the Pistonball environment.

        Args:
            env: The Pistonball environment (can be wrapped)

        Returns:
            Dictionary containing:
            - ball_position: (x, y) coordinates
            - ball_velocity: (vx, vy) velocity
            - ball_radius: ball radius
            - pistons: list of piston states (position, velocity)
            - screen_width, screen_height: dimensions
            - piston_width, piston_height: piston dimensions
        """
        # Unwrap to get the raw environment
        raw = env.unwrapped

        # Get ball state
        ball = raw.ball
        ball_pos = (float(ball.position.x), float(ball.position.y))
        ball_vel = (float(ball.velocity.x), float(ball.velocity.y))

        # Get piston states
        pistons = []
        for i, piston in enumerate(raw.pistonList):
            pistons.append({
                "index": i,
                "position": (float(piston.position.x), float(piston.position.y)),
                "velocity": (float(piston.velocity.x), float(piston.velocity.y)),
            })

        return {
            "ball_position": ball_pos,
            "ball_velocity": ball_vel,
            "ball_radius": raw.ball_radius,
            "pistons": pistons,
            "screen_width": raw.screen_width,
            "screen_height": raw.screen_height,
            "piston_width": raw.piston_width,
            "piston_height": raw.piston_height,
            "n_pistons": raw.n_pistons,
        }

    @staticmethod
    def format_local_observation(env, agent_idx: int) -> str:
        """
        Format a single agent's local observation as text using environment state.

        Args:
            env: The Pistonball environment
            agent_idx: Index of the agent (0 = leftmost piston)

        Returns:
            Text description of this agent's local observation
        """
        state = PistonballObservationFormatter.get_env_state(env)

        num_agents = state["n_pistons"]
        ball_x, ball_y = state["ball_position"]
        ball_vx, ball_vy = state["ball_velocity"]
        piston = state["pistons"][agent_idx]
        piston_x, piston_y = piston["position"]
        screen_width = state["screen_width"]
        piston_width = state["piston_width"]

        ball_y = 491.0 - ball_y
        piston_y = 491.0 - piston_y

        # Calculate relative position of ball to this piston
        ball_rel_x = ball_x - piston_x  # positive = ball is to the right

        # Determine piston's global position description
        if agent_idx < num_agents * 0.33:
            piston_region = "left side (near goal)"
        elif agent_idx < num_agents * 0.67:
            piston_region = "center"
        else:
            piston_region = "right side (far from goal)"

        # Calculate distances
        distance_to_ball = abs(ball_rel_x)
        ball_above_piston = ball_y < piston_y  # y increases downward in pygame

        lines = [
            f"=== Your Piston Status ===",
            f"Piston index: #{agent_idx} of {num_agents} ({piston_region})",
            f"Your X position: {piston_x:.1f} (screen width: {screen_width})",
            f"Your Y position: {piston_y:.1f}",
            f"",
            f"=== Ball Relative to You ===",
        ]

        # Describe ball position relative to this piston
        if distance_to_ball < piston_width * 1.5:
            lines.append(f"Ball distance: {distance_to_ball:.1f} pixels - VERY CLOSE!")
            if ball_rel_x < 0:
                lines.append(f"Ball is slightly to your LEFT")
            elif ball_rel_x > 0:
                lines.append(f"Ball is slightly to your RIGHT")
            else:
                lines.append(f"Ball is directly ABOVE you")
        elif distance_to_ball < piston_width * 3:
            lines.append(f"Ball distance: {distance_to_ball:.1f} pixels - NEARBY")
            if ball_rel_x < 0:
                lines.append(f"Ball is to your LEFT (closer to goal)")
            else:
                lines.append(f"Ball is to your RIGHT (coming towards you)")
        else:
            lines.append(f"Ball distance: {distance_to_ball:.1f} pixels - FAR")
            if ball_rel_x < 0:
                lines.append(f"Ball has passed you (already closer to goal)")
            else:
                lines.append(f"Ball has not reached you yet")

        # Ball movement
        lines.append(f"")
        lines.append(f"=== Ball Movement ===")
        if abs(ball_vx) < 5:
            h_movement = "nearly stationary horizontally"
        elif ball_vx < 0:
            h_movement = f"moving LEFT at {abs(ball_vx):.1f} px/step (toward goal!)"
        else:
            h_movement = f"moving RIGHT at {ball_vx:.1f} px/step (away from goal)"
        lines.append(f"Horizontal: {h_movement}")

        if abs(ball_vy) < 5:
            v_movement = "nearly stationary vertically"
        elif ball_vy > 0:
            v_movement = f"moving DOWN at {ball_vy:.1f} px/step (toward pistons)"
        else:
            v_movement = f"moving UP at {abs(ball_vy):.1f} px/step (away from pistons)"
        lines.append(f"Vertical: {v_movement}")

        return "\n".join(lines)

    @staticmethod
    def format_global_state(env) -> str:
        """
        Format the global state of the Pistonball environment for LLM.

        Args:
            env: The Pistonball environment

        Returns:
            Formatted state description
        """
        state = PistonballObservationFormatter.get_env_state(env)

        ball_x, ball_y = state["ball_position"]
        ball_y = 491.0 - ball_y
        ball_vx, ball_vy = state["ball_velocity"]
        screen_width = state["screen_width"]
        num_agents = state["n_pistons"]
        piston_width = state["piston_width"]

        # Calculate ball's global position as percentage
        ball_progress = 1.0 - (ball_x / screen_width)  # 1.0 = at left edge (goal)

        # Determine which pistons are near the ball
        nearby_pistons = []
        for p in state["pistons"]:
            dist = abs(p["position"][0] - ball_x)
            if dist < piston_width * 2:
                nearby_pistons.append(p["index"])

        state_lines = [
            f"=== Pistonball Global State ===",
            f"Number of pistons: {num_agents}",
            f"Objective: Push the ball to the LEFT edge (x=0)",
            f"",
            f"=== Ball Status ===",
            f"Position: x={ball_x:.1f}, y={ball_y:.1f}",
            f"Progress toward goal: {ball_progress*100:.1f}% (100% = goal reached)",
            f"Velocity: vx={ball_vx:.1f}, vy={ball_vy:.1f}",
        ]

        # Ball location description
        if ball_progress > 0.8:
            state_lines.append(f"Location: Very close to GOAL! Keep pushing!")
        elif ball_progress > 0.5:
            state_lines.append(f"Location: Past halfway, good progress")
        elif ball_progress > 0.2:
            state_lines.append(f"Location: Still in the right half of the field")
        else:
            state_lines.append(f"Location: Far from goal, need coordinated pushing")

        # Movement summary
        if ball_vx < -10:
            state_lines.append(f"Movement: Moving LEFT toward goal (good!)")
        elif ball_vx > 10:
            state_lines.append(f"Movement: Moving RIGHT away from goal (bad!)")
        else:
            state_lines.append(f"Movement: Nearly stationary horizontally")

        # Nearby pistons
        state_lines.append(f"")
        state_lines.append(f"=== Pistons Near Ball ===")
        if nearby_pistons:
            state_lines.append(f"Pistons that can contact ball: {nearby_pistons}")
        else:
            state_lines.append(f"No pistons currently near the ball")

        # All piston positions summary
        state_lines.append(f"")
        state_lines.append(f"=== All Piston Positions ===")
        for p in state["pistons"]:
            px, py = p["position"]
            py = 491.0 - py
            dist_to_ball = ball_x - px
            # Determine NEAR BALL status with left/right direction
            if p["index"] in nearby_pistons:
                if dist_to_ball > 0:
                    status = ">>NEAR BALL (piston is ON LEFT of the ball)<<"  # Ball is to the right of piston
                elif dist_to_ball < 0:
                    status = ">>NEAR BALL (piston is ON RIGHT of the ball)<<"   # Ball is to the left of piston
                else:
                    status = ">>NEAR BALL (piston is at the same x with the ball)<<"  # Ball is directly above piston
            else:
                status = ""
            state_lines.append(f"  Piston {p['index']}: x={px:.0f}, y={py:.0f} (height), ball is {dist_to_ball:+.0f}px away {status}")

        return "\n".join(state_lines)

    @staticmethod
    def format_agent_prompt(env, agent_name: str, policy: str = None,
                            action_mode: str = "discrete") -> str:
        """
        Generate a complete prompt for a specific agent.

        Args:
            env: The Pistonball environment
            agent_name: Name of the agent to generate prompt for
            policy: Optional policy text
            action_mode: "discrete" (3 actions) or "continuous" [-1, 1]

        Returns:
            Complete prompt text for the LLM
        """
        raw = env.unwrapped
        num_agents = raw.n_pistons
        agent_idx = int(agent_name.split("_")[-1])  # Extract index from "piston_X"

        # Get local observation description
        local_obs_text = PistonballObservationFormatter.format_local_observation(env, agent_idx)

        # Get global state summary
        global_state_text = PistonballObservationFormatter.format_global_state(env)

        prompt = f"""
=== GAME RULES ===
You are operating in a **cooperative piston-based control environment**, where you control **piston #{agent_idx}** in a **Pistonball game**.

### Role and Objective
* The environment consists of **{num_agents} pistons**, all of which **share the same reward**
* The collective goal is to **coordinate and push a ball to the LEFT boundary (x = 0)**
* Once the ball reaches the left edge, **all pistons receive the reward**

### Piston Dynamics
* Pistons can move **only vertically**
  * **UP**: pushes the ball to the LEFT (toward the goal)
  * **DOWN**: retracts the piston
* Effective ball movement requires **coordination with neighboring pistons**

### Local Observations
At each timestep, you can observe only **local information**:
1. **Your own vertical height** (in pixels)
2. **The vertical height of your left neighbor** (in pixels)
3. **The vertical height of your right neighbor** (in pixels)
4. **Whether the ball is visible in your local field of view**
5. If the ball is visible:
   * **The horizontal position of the ball relative to you**
   * A **positive value** means the ball is to your right
   * A **negative value** means the ball is to your left

### Action Requirement
* At every timestep, you must choose one discrete action

=== YOUR LOCAL OBSERVATION ===
{local_obs_text}
"""

        if policy:
            prompt += f"""
=== CURRENT POLICY ===
{policy}
"""

        # Add decision section based on action mode
        if action_mode == "discrete":
            prompt += """
=== YOUR DECISION ===
Choose your action (respond with the number only on the first line):
  0 = RETRACT_DOWN
  1 = STAY (hold position)
  2 = PUSH_UP

Action:"""
        elif action_mode == "continuous":
            prompt += """
=== YOUR DECISION ===
Choose your action as a continuous value between -1.0 and 1.0:
  -1.0 = Maximum retract down
   0.0 = Stay in place
  +1.0 = Maximum push up

Respond with a single decimal number (e.g., 0.5, -0.3, 1.0) on the first line.

Action:"""

        return prompt

    @staticmethod
    def format_action(action, action_mode: str = "discrete") -> str:
        """
        Convert action to readable string.

        Args:
            action: Action value (int for discrete, float for continuous)
            action_mode: "discrete" or "continuous"

        Returns:
            Human-readable action description
        """
        if action_mode == "discrete":
            # 0 - retract down, 1 - stay, 2 - push up
            discrete_names = {0: "retract_down", 1: "stay", 2: "push_up"}
            return discrete_names.get(action, f"unknown({action})")
        elif action_mode == "continuous":
            if action > 0.5:
                return f"push_up_strong({action:+.2f})"
            elif action > 0.1:
                return f"push_up_light({action:+.2f})"
            elif action > -0.1:
                return f"stay({action:+.2f})"
            elif action > -0.5:
                return f"retract_light({action:+.2f})"
            else:
                return f"retract_strong({action:+.2f})"
        return f"unknown({action})"

    @staticmethod
    def discrete_to_continuous(action: int) -> float:
        """
        Convert basic discrete action (0,1,2) to continuous value.

        Args:
            action: Discrete action (0=retract_down, 1=stay, 2=push_up)

        Returns:
            Continuous value in [-1.0, 1.0]
        """
        # 0 - retract down -> -1.0, 1 - stay -> 0.0, 2 - push up -> 1.0
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping.get(action, 0.0)

    @staticmethod
    def continuous_to_discrete(value: float) -> int:
        """
        Convert continuous value to basic discrete action.

        Args:
            value: Continuous value in [-1.0, 1.0]

        Returns:
            Discrete action (0=retract_down, 1=stay, 2=push_up)
        """
        if value > 0.33:
            return 2  # push_up
        elif value < -0.33:
            return 0  # retract_down
        else:
            return 1  # stay

    @staticmethod
    def format_joint_action(actions: Dict, action_mode: str = "discrete") -> str:
        """
        Format joint action of all pistons.

        Args:
            actions: Dictionary of actions keyed by agent name
            action_mode: "discrete" or "continuous"

        Returns:
            Formatted action description
        """
        action_strs = []
        for agent_name, action in sorted(actions.items()):
            action_str = PistonballObservationFormatter.format_action(action, action_mode)
            action_strs.append(f"{agent_name}: {action_str}")
        return ", ".join(action_strs)


class EpisodeGenerator:
    """Generate episodes by running simulations in the Pistonball environment"""

    def __init__(
        self,
        num_pistons: int = 20,
        max_cycles: int = 125,
        frame_size: Tuple[int, int] = (64, 64),
        stack_size: int = 4,
        continuous: bool = False,
        experiments_dir: Path = None,
        exp_name: str = "pistonball_exp",
        agent_type: str = "rule",  # 'llm' | 'rule'
        gpt_model: str = "gpt-4o-mini",
        action_mode: str = "discrete",  # 'discrete' | 'continuous'
        random_drop: bool = True,  # If True, ball spawns at random x; if False, ball spawns at x=800
        pistons_start_low: bool = False,  # If True, all pistons start at lowest position
        rule_k: float = 0.1,  # Slope coefficient for rule-based scoring function
        rule_tau: float = 4.0,  # Dead-zone threshold for rule-based scoring function
        llm_config: Optional[Union[LLMConfig, Dict, str]] = None,  # LLM configuration
    ):
        """
        Initialize episode generator

        Args:
            num_pistons: Number of pistons in the environment
            max_cycles: Maximum cycles per episode
            frame_size: Frame size for observations
            stack_size: Number of frames to stack
            continuous: Whether to use continuous actions (env setting)
            experiments_dir: Directory to save episode data
            exp_name: Experiment name
            agent_type: Type of agent:
                - 'rule': Formula-based agent (no LLM)
                - 'llm': LLM agent that parses actions from prompts
            gpt_model: GPT model name for LLM-based agents (used if llm_config is None)
            action_mode: Action space mode:
                - 'discrete': 3 actions (0=retract_down, 1=stay, 2=push_up)
                - 'continuous': continuous value in [-1.0, 1.0]
            random_drop: If True, ball spawns at random x position; if False, ball spawns at x=800
            pistons_start_low: If True, all pistons start at their lowest position after reset
            llm_config: LLM configuration for API calls. Can be:
                - LLMConfig object
                - Dict with config values
                - String (predefined model name or path to JSON file)
                - None (will use gpt_model to look up config)
        """
        self.num_pistons = num_pistons
        self.max_cycles = max_cycles
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.continuous = continuous
        self.experiments_dir = Path(experiments_dir) if experiments_dir else Path("./experiments")
        self.exp_name = exp_name
        self.agent_type = agent_type
        self.action_mode = action_mode
        self.random_drop = random_drop  # Ball initial x position randomization
        self.pistons_start_low = pistons_start_low  # Piston initial position control
        self.rule_k = rule_k  # Slope coefficient for rule-based scoring
        self.rule_tau = rule_tau  # Dead-zone threshold for rule-based scoring

        # Resolve LLM configuration
        self.llm_config = self._resolve_llm_config(gpt_model, llm_config)
        self.gpt_model = self.llm_config.model_string

        # Initialize OpenAI client with LLM config
        self._llm_client = None

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Validate action_mode
        valid_modes = ["discrete", "continuous"]
        if action_mode not in valid_modes:
            raise ValueError(f"action_mode must be one of {valid_modes}, got {action_mode}")

        # For continuous action mode, we need continuous env
        env_continuous = continuous or action_mode == "continuous"

        # Initialize environment
        self.env = make_env(
            num_pistons=num_pistons,
            max_cycles=max_cycles,
            frame_size=frame_size,
            stack_size=stack_size,
            continuous=env_continuous,
            random_drop=random_drop,
        )

        self.num_agents = len(self.env.possible_agents)

        # Set number of actions based on mode
        if action_mode == "discrete":
            self.num_actions = 3  # 0-retract, 1-stay, 2-push
        else:  # continuous
            self.num_actions = None  # Continuous, no discrete count

        self.logger.info(
            f"Initialized EpisodeGenerator for Pistonball "
            f"(pistons={num_pistons}, max_cycles={max_cycles}, action_mode={action_mode})"
        )

        # Observation formatter
        self.obs_formatter = PistonballObservationFormatter()

    def _resolve_llm_config(
        self,
        gpt_model: str,
        llm_config: Optional[Union[LLMConfig, Dict, str]]
    ) -> LLMConfig:
        """Resolve LLM configuration from various input types."""
        if llm_config is not None:
            if isinstance(llm_config, LLMConfig):
                return llm_config
            elif isinstance(llm_config, dict):
                return LLMConfig.from_dict(llm_config)
            elif isinstance(llm_config, str):
                return get_llm_config(llm_config)
            else:
                raise ValueError(f"Invalid llm_config type: {type(llm_config)}")
        else:
            # Try to get predefined config for the model name
            return get_llm_config(gpt_model)

    def _get_llm_client(self) -> OpenAI:
        """Get or create OpenAI client configured for the LLM provider."""
        if self._llm_client is None:
            api_key = self.llm_config.get_api_key()
            if self.llm_config.base_url:
                self._llm_client = OpenAI(
                    base_url=self.llm_config.base_url,
                    api_key=api_key
                )
            else:
                self._llm_client = OpenAI(api_key=api_key)
        return self._llm_client

    def _get_rule_based_action(self, obs: np.ndarray, agent_idx: int):
        """
        Local discrete control policy for Pistonball with partial observability.

        This implements Algorithm 1 from the paper, handling the case when the ball
        is outside the piston's local observation region.

        Score function with visibility indicator:
            S_i = δ_i · k · (x_i - x_b) - (h_i - h̄_i)

        Where:
            - δ_i ∈ {0, 1}: Ball visibility indicator (1 if ball in local observation)
            - x_b: Ball's horizontal position (pixels)
            - x_i: Piston i's horizontal center position (pixels)
            - h_i: Piston i's height (converted from y-coordinate)
            - h̄_i = (h_{i-1} + h_{i+1}) / 2: Neighbor-averaged height
            - k > 0: Slope coefficient controlling "right-high, left-low" trend
            - τ > 0: Dead-zone threshold to suppress action jitter

        Quantization rule:
            u_i = 1 (up)   if S_i > τ
            u_i = 0 (stay) if |S_i| ≤ τ
            u_i = -1 (down) if S_i < -τ

        Key behaviors:
        - When ball is visible (δ_i = 1): Full scoring function creates local slope
          around the ball, with "right-high, left-low" pattern.
        - When ball is NOT visible (δ_i = 0): Policy degrades to pure neighbor-based
          smoothing control S_i = -(h_i - h̄_i), maintaining a continuous, stable
          height field without ball-following behavior.

        Local observation region:
            In PettingZoo Pistonball, each piston's local observation covers itself
            and its immediate left/right neighbors. The visibility range is computed
            as approximately 1.5 × piston_width from the piston center.

        Args:
            obs: Observation array for the agent (unused, kept for compatibility)
            agent_idx: Index of the agent

        Returns:
            Action based on action_mode:
            - discrete: int (0-2) where 0=retract_down, 1=stay, 2=push_up
            - continuous: float [-1.0, 1.0]
        """
        # Get actual state from environment
        state = self.obs_formatter.get_env_state(self.env)
        ball_x, ball_y = state["ball_position"]
        n_pistons = state["n_pistons"]
        pistons = state["pistons"]
        piston_width = state["piston_width"]

        # Get current piston's position
        piston_x, piston_y = pistons[agent_idx]["position"]

        # ========================================
        # Step 1: Determine ball visibility (δ_i)
        # ========================================
        # Local observation covers this piston and its left/right neighbors.
        # Compute the horizontal bounds of local observation region.
        # For piston i, visible range is approximately from neighbor i-1's center
        # to neighbor i+1's center, which spans about 3 piston widths.
        # We use 1.5 × piston_width as the half-width of visible region.
        visibility_half_width = 1.5 * piston_width

        # Ball is visible if its x-coordinate falls within the local observation
        ball_distance_x = abs(ball_x - piston_x)
        delta_i = 1 if ball_distance_x <= visibility_half_width else 0

        # ========================================
        # Step 2: Compute piston horizontal center x_i
        # ========================================
        x_i = piston_x  # Already the center position

        # ========================================
        # Step 3: Compute neighbor-averaged height h̄_i
        # ========================================
        # Get neighbor pistons' y positions and convert to "height" concept
        # In pygame: y increases downward, so smaller y = higher physical position
        # We convert to "height" (larger = higher) by negating y values
        # Handle boundary cases: use current piston's height for missing neighbors
        if agent_idx == 0:
            # Leftmost piston: no left neighbor, use own y
            neighbor_left_y = piston_y
        else:
            neighbor_left_y = pistons[agent_idx - 1]["position"][1]

        if agent_idx == n_pistons - 1:
            # Rightmost piston: no right neighbor, use own y
            neighbor_right_y = piston_y
        else:
            neighbor_right_y = pistons[agent_idx + 1]["position"][1]

        # Convert pygame y coordinates to "height" (larger value = higher position)
        h_i = -piston_y  # Current piston height
        h_left = -neighbor_left_y  # Left neighbor height
        h_right = -neighbor_right_y  # Right neighbor height

        # Neighbor-averaged height: h̄_i = (h_{i-1} + h_{i+1}) / 2
        h_bar_i = 0.5 * (h_left + h_right)

        # ========================================
        # Step 4: Compute score S_i
        # ========================================
        # Parameters (from instance configuration)
        k = self.rule_k
        tau = self.rule_tau

        # S_i = δ_i · k · (x_i - x_b) - (h_i - h̄_i)
        # When δ_i = 1: Full scoring with ball-following slope
        # When δ_i = 0: Pure smoothing control S_i = -(h_i - h̄_i)
        position_term = delta_i * k * (x_i - ball_x)
        smoothing_term = h_i - h_bar_i
        S_i = position_term - smoothing_term

        # ========================================
        # Step 5: Quantize score into discrete action
        # ========================================
        if S_i > tau:
            u_i = 1  # move piston up
        elif S_i < -tau:
            u_i = -1  # move piston down
        else:
            u_i = 0  # no movement (dead-zone)

        # Convert u_i to environment action format
        if self.action_mode == "discrete":
            # Environment: 0=retract_down, 1=stay, 2=push_up
            # u_i: -1=down, 0=stay, 1=up
            # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
            return u_i + 1
        else:  # continuous
            # Map to continuous value
            if u_i == 1:
                return 0.7
            elif u_i == -1:
                return -0.7
            else:
                return 0.0

    def _get_llm_action(self, obs_dict: Dict, policy: str, agent_name: str):
        """
        Get action from LLM based on environment state and policy.

        Uses the configured LLM provider (OpenAI, Gemini, Llama, Qwen, etc.)
        via OpenAI-compatible API.

        Args:
            obs_dict: Dictionary of observations (kept for compatibility)
            policy: Current policy text for the LLM
            agent_name: Name of the agent

        Returns:
            Tuple of (action, llm_response, prompt) where action type depends on action_mode
        """
        # Generate complete prompt using environment state (not image parsing)
        prompt = self.obs_formatter.format_agent_prompt(
            env=self.env,
            agent_name=agent_name,
            policy=policy,
            action_mode=self.action_mode
        )

        # Get client and make API call
        client = self._get_llm_client()

        # Build API call parameters
        api_params = {
            "model": self.gpt_model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Handle model-specific parameter differences
        # o1, o3, and gpt-5 models require max_completion_tokens instead of max_tokens
        model_lower = self.gpt_model.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            api_params["max_completion_tokens"] = self.llm_config.max_tokens
        else:
            api_params["max_tokens"] = self.llm_config.max_tokens

        try:
            response = client.chat.completions.create(**api_params)
            response_text = response.choices[0].message.content
        except Exception as e:
            # Log detailed error information
            error_msg = str(e)
            if hasattr(e, 'status_code'):
                error_msg = f"HTTP {e.status_code}: {error_msg}"
            if hasattr(e, 'body'):
                try:
                    error_msg = f"{error_msg}\nBody: {e.body}"
                except Exception:
                    pass
            self.logger.error(f"LLM API error for {agent_name}: {error_msg}")
            self.logger.error(f"Model: {self.gpt_model}, API params: {api_params.keys()}")
            raise

        action = parse_action_from_response(response_text, self.action_mode)

        # Format action for logging
        action_str = self.obs_formatter.format_action(action, self.action_mode)
        response_log = f"Action: {action_str} - Following coordination policy"

        return action, response_log, prompt  # Return prompt for logging

    def _convert_action_for_env(self, action):
        """
        Convert action to format expected by environment.

        Args:
            action: Action from policy (int or float depending on action_mode)

        Returns:
            Action in environment format (float for continuous env)
        """
        if self.action_mode == "discrete":
            # Convert discrete (0,1,2) to continuous if env is continuous
            if self.continuous:
                return np.array([self.obs_formatter.discrete_to_continuous(action)])
            return action
        else:  # continuous
            # Already in correct format
            return np.array([action])

    def generate_episodes(
        self,
        num_episodes: int,
        iteration: int,
        agent_policies: Dict[str, str] = None,
        num_workers: Optional[int] = None,
        parallel: bool = True,
        save_gif: bool = True,
        gif_fps: int = 30,
        incremental_save: bool = True,
        save_interval: int = 10,
    ) -> List[Dict]:
        """
        Generate multiple episodes

        Args:
            num_episodes: Number of episodes to generate
            iteration: Current training iteration
            agent_policies: Dict mapping agent_name -> policy_text
                e.g., {"piston_0": "policy...", "piston_1": "policy...", ...}
            num_workers: Number of parallel workers (default: cpu_count)
            parallel: If True, run episodes in parallel (default: True)
            save_gif: Whether to save GIF for each episode
            gif_fps: Frames per second for GIF
            incremental_save: If True, save data incrementally during episode
            save_interval: Save every N steps when incremental_save=True

        Returns:
            List of episode dictionaries
        """
        self.logger.info(
            f"Generating {num_episodes} episodes for iteration {iteration} "
            f"(parallel={parallel}, incremental_save={incremental_save})"
        )

        if not parallel or num_episodes == 1:
            # Sequential execution
            episodes = []
            for i in range(num_episodes):
                episode = self._run_single_episode(
                    i, iteration, agent_policies,
                    save_gif=save_gif, gif_fps=gif_fps,
                    incremental_save=incremental_save,
                    save_interval=save_interval,
                )
                episodes.append(episode)
                self.logger.info(
                    f"Episode {i+1}/{num_episodes} completed: "
                    f"reward={episode['total_reward']:.2f}"
                )
            return episodes

        # Parallel execution using multiprocessing
        if num_workers is None:
            num_workers = min(cpu_count(), num_episodes)

        self.logger.info(f"Using {num_workers} parallel workers (multiprocessing)")

        # Prepare configuration dictionary for workers
        config = {
            'num_pistons': self.num_pistons,
            'max_cycles': self.max_cycles,
            'frame_size': self.frame_size,
            'stack_size': self.stack_size,
            'continuous': self.continuous,
            'experiments_dir': self.experiments_dir,
            'exp_name': self.exp_name,
            'agent_type': self.agent_type,
            'gpt_model': self.gpt_model,
            'action_mode': self.action_mode,
            'random_drop': self.random_drop,
            'pistons_start_low': self.pistons_start_low,
            'rule_k': self.rule_k,
            'rule_tau': self.rule_tau,
            'llm_config': self.llm_config.to_dict() if self.llm_config else None,
            'agent_policies': agent_policies,
            'save_gif': save_gif,
            'gif_fps': gif_fps,
            'incremental_save': incremental_save,
            'save_interval': save_interval,
        }

        # Prepare arguments for each episode
        worker_args = [(i, iteration, config) for i in range(num_episodes)]

        # Run episodes in parallel
        with Pool(processes=num_workers) as pool:
            episodes = pool.map(_run_episode_worker, worker_args)

        # Log completion
        for i, episode in enumerate(episodes):
            self.logger.info(
                f"Episode {i+1}/{num_episodes} completed: "
                f"reward={episode['total_reward']:.2f}"
            )

        return episodes

    def _run_single_episode(
        self,
        episode_id: int,
        iteration: int,
        agent_policies: Dict[str, str] = None,
        save_gif: bool = True,
        gif_fps: int = 30,
        incremental_save: bool = True,
        save_interval: int = 10,
    ) -> Dict:
        """
        Run a single episode simulation

        Args:
            episode_id: Episode ID
            iteration: Current training iteration
            agent_policies: Dict mapping agent_name -> policy_text
                e.g., {"piston_0": "policy...", "piston_1": "policy...", ...}
            save_gif: Whether to save GIF of the episode
            gif_fps: Frames per second for GIF
            incremental_save: If True, save data incrementally during episode
            save_interval: Save every N steps when incremental_save=True

        Returns:
            Episode data dictionary
        """
        # Create episode directory
        iter_dir = self.experiments_dir / self.exp_name / f"iteration_{iteration}"
        episode_dir = iter_dir / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # GIF frames collection
        gif_frames = [] if save_gif and HAS_IMAGEIO else None

        # Use provided agent policies or create default
        if agent_policies is None:
            agent_policies = {
                f"piston_{i}": ""
                for i in range(self.num_pistons)
            }

        # Create log file for this episode
        log_file = episode_dir / "episode_log.txt"
        episode_logs = []

        # Reset environment
        obs_dict, info = self.env.reset()

        # If pistons_start_low is enabled, reset all pistons to lowest position
        if self.pistons_start_low:
            reset_pistons_to_lowest(self.env)

        total_reward = 0.0
        transitions = []
        trajectory = []

        # Token tracking
        total_gpt_queries = 0
        total_estimated_tokens = {"input": 0, "output": 0}

        # Run episode
        for t in range(self.max_cycles):
            # Capture frame for GIF (before taking action)
            if gif_frames is not None:
                try:
                    frame = self.env.render()
                    if frame is not None:
                        gif_frames.append(frame)
                except Exception as e:
                    self.logger.warning(f"Failed to render frame at timestep {t}: {e}")

            # === OBSERVATION SEPARATION ===
            # Global state for Critic (centralized training)
            global_state = self.obs_formatter.get_env_state(self.env)
            global_state_text = self.obs_formatter.format_global_state(self.env)

            # Get actions for all agents
            actions = {}
            agent_infos = {}

            # Process each agent individually
            for agent_name in self.env.possible_agents:
                if agent_name not in obs_dict:
                    continue

                agent_idx = self.env.possible_agents.index(agent_name)

                # === LOCAL OBSERVATION for Actor (decentralized execution) ===
                local_obs_text = self.obs_formatter.format_local_observation(self.env, agent_idx)

                if self.agent_type == "llm":
                    # Get agent-specific policy
                    agent_policy = agent_policies.get(
                        agent_name,
                        "Default: Push up when ball is near, retract after it passes."
                    )
                    action, response, agent_prompt = self._get_llm_action(obs_dict, agent_policy, agent_name)
                    total_gpt_queries += 1
                    token_est = estimate_tokens(agent_prompt + response)
                    total_estimated_tokens["input"] += token_est['estimated_tokens']
                    total_estimated_tokens["output"] += token_est['estimated_tokens'] // 3
                else:
                    # Default rule-based agent (formula-based control)
                    action = self._get_rule_based_action(obs_dict[agent_name], agent_idx)
                    response = None
                    agent_prompt = None

                # Store original action for logging
                agent_infos[agent_name] = {
                    'action': action,
                    'action_name': self.obs_formatter.format_action(action, self.action_mode),
                    'response': response,
                    'prompt': agent_prompt,
                    'local_observation': local_obs_text,  # Local obs for actor
                }

                # Convert action to environment format
                env_action = self._convert_action_for_env(action)
                actions[agent_name] = env_action

            # Step environment
            next_obs_dict, rewards, terms, truncs, infos = self.env.step(actions)

            # Calculate step reward (mean across all agents)
            step_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            total_reward += step_reward

            # Create transition record with BOTH local and global observations
            # Convert numpy types to native Python types for JSON serialization
            def to_native(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [to_native(x) for x in obj]
                return obj

            transition = {
                "timestep": t,
                # Global state for Critic (centralized training)
                "global_state": {
                    "text": global_state_text,
                    "ball_position": to_native(global_state["ball_position"]),
                    "ball_velocity": to_native(global_state["ball_velocity"]),
                    "pistons": to_native(global_state["pistons"]),
                },
                "agents": {},
                "instant_reward": step_reward,
                "cumulative_reward": total_reward,
                "done": any(terms.values()) or any(truncs.values()),
            }

            # Record agent information with local observations
            for agent_name in self.env.possible_agents:
                if agent_name in agent_infos:
                    # Convert action to native Python type for JSON serialization
                    action_val = agent_infos[agent_name]['action']
                    if hasattr(action_val, 'item'):  # numpy type
                        action_val = action_val.item()
                    elif isinstance(action_val, np.ndarray):
                        action_val = action_val.tolist()

                    transition["agents"][agent_name] = {
                        "action": action_val,
                        "action_name": agent_infos[agent_name]['action_name'],
                        # Local observation for Actor (decentralized execution)
                        "local_observation": agent_infos[agent_name]['local_observation'],
                        "observation_to_gpt": agent_infos[agent_name].get('prompt') if self.agent_type == "llm" else None,
                        "gpt_query_response": agent_infos[agent_name]['response'],
                    }

            transitions.append(transition)
            trajectory.append((obs_dict, actions, step_reward, transition["done"], infos))

            # Incremental save: save data every save_interval steps
            if incremental_save and (t + 1) % save_interval == 0:
                self._save_incremental(
                    episode_dir=episode_dir,
                    transitions=transitions,
                    trajectory=trajectory,
                    total_reward=total_reward,
                    timestep=t,
                    episode_id=episode_id,
                    iteration=iteration,
                    gif_frames=gif_frames,
                    gif_fps=gif_fps,
                )

            # Add to episode log
            episode_logs.append(f"\n--- Timestep {t} ---")
            episode_logs.append(f"[GLOBAL STATE (for Critic)]")
            episode_logs.append(global_state_text)
            episode_logs.append(f"\n[ACTIONS]")
            # Log original actions (before conversion)
            original_actions = {name: info['action'] for name, info in agent_infos.items()}
            episode_logs.append(f"Actions: {self.obs_formatter.format_joint_action(original_actions, self.action_mode)}")
            episode_logs.append(f"Reward: {step_reward:.4f} (Total: {total_reward:.4f})")

            # Check for episode end
            if any(terms.values()) or any(truncs.values()):
                episode_logs.append(f"\nEpisode ended at timestep {t}")
                break

            obs_dict = next_obs_dict

        # Final log
        episode_logs.append(f"\n=== Episode {episode_id} Summary ===")
        episode_logs.append(f"Total Reward: {total_reward:.4f}")
        episode_logs.append(f"Steps: {len(transitions)}")

        # Save episode log
        with open(log_file, 'w') as f:
            f.write('\n'.join(episode_logs))

        # Save GIF if frames were collected
        gif_path = None
        if gif_frames is not None and len(gif_frames) > 0:
            gif_path = episode_dir / f"episode_{episode_id}.gif"
            try:
                imageio.mimsave(str(gif_path), gif_frames, fps=gif_fps)
                self.logger.info(f"Saved episode GIF to {gif_path} ({len(gif_frames)} frames)")
            except Exception as e:
                self.logger.warning(f"Failed to save GIF: {e}")
                gif_path = None

        # Create episode data
        episode_data = {
            "episode_id": episode_id,
            "iteration": iteration,
            "total_reward": total_reward,
            "num_pistons": self.num_pistons,
            "max_cycles": self.max_cycles,
            "agent_type": self.agent_type,
            "action_mode": self.action_mode,
            "transitions": transitions,
            "gif_path": str(gif_path) if gif_path else None,
            "token_stats": {
                "total_gpt_queries": total_gpt_queries,
                "estimated_tokens_simulation": {
                    "input": total_estimated_tokens["input"],
                    "output": total_estimated_tokens["output"]
                }
            },
        }

        # Save episode data
        self._save_episode(episode_data, iteration, episode_id)

        # Save trajectory pickle (includes both local obs and global state)
        traj_file = episode_dir / "trajectory.pkl"
        with open(traj_file, "wb") as f:
            pickle.dump({
                "trajectory": trajectory,
                "transitions": transitions,  # Contains local/global obs separation
                "total_reward": total_reward,
                "num_steps": len(transitions),
            }, f)

        self.logger.debug(f"Saved episode {episode_id} to {episode_dir}")

        return episode_data

    def _save_incremental(
        self,
        episode_dir: Path,
        transitions: List[Dict],
        trajectory: List,
        total_reward: float,
        timestep: int,
        episode_id: int,
        iteration: int,
        gif_frames: List = None,
        gif_fps: int = 30,
    ):
        """
        Save episode data incrementally during simulation.

        This saves a checkpoint of the episode at the current timestep,
        allowing recovery if the simulation is interrupted.

        Args:
            episode_dir: Directory to save the checkpoint
            transitions: List of transitions so far
            trajectory: List of trajectory tuples so far
            total_reward: Cumulative reward so far
            timestep: Current timestep
            episode_id: Episode ID
            iteration: Training iteration
            gif_frames: List of frames collected so far for GIF
            gif_fps: Frames per second for GIF
        """
        checkpoint_file = episode_dir / f"checkpoint.json"

        checkpoint_data = {
            "episode_id": episode_id,
            "iteration": iteration,
            "timestep": timestep,
            "total_reward": total_reward,
            "num_transitions": len(transitions),
            "transitions": transitions,
            "is_checkpoint": True,
        }

        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            self.logger.debug(f"Saved incremental checkpoint at step {timestep}")
        except Exception as e:
            self.logger.warning(f"Failed to save incremental checkpoint: {e}")

        # Also save trajectory pickle checkpoint
        traj_checkpoint_file = episode_dir / f"trajectory_checkpoint.pkl"
        try:
            with open(traj_checkpoint_file, "wb") as f:
                pickle.dump({
                    "trajectory": trajectory,
                    "transitions": transitions,
                    "total_reward": total_reward,
                    "num_steps": len(transitions),
                    "is_checkpoint": True,
                    "checkpoint_timestep": timestep,
                }, f)
        except Exception as e:
            self.logger.warning(f"Failed to save trajectory checkpoint: {e}")

        # Save GIF checkpoint if frames are available
        if gif_frames is not None and len(gif_frames) > 0 and HAS_IMAGEIO:
            gif_checkpoint_path = episode_dir / f"episode_{episode_id}_step_{timestep}.gif"
            try:
                imageio.mimsave(str(gif_checkpoint_path), gif_frames, fps=gif_fps)
                self.logger.debug(f"Saved GIF checkpoint to {gif_checkpoint_path} ({len(gif_frames)} frames)")
            except Exception as e:
                self.logger.warning(f"Failed to save GIF checkpoint: {e}")

    def _save_episode(self, episode_data: Dict, iteration: int, episode_id: int):
        """
        Save episode data to disk

        Args:
            episode_data: Episode data dictionary
            iteration: Training iteration
            episode_id: Episode ID
        """
        # Create iteration directory
        iter_dir = self.experiments_dir / self.exp_name / f"iteration_{iteration}"
        episode_dir = iter_dir / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save episode JSON
        episode_file = episode_dir / "episode.json"
        with open(episode_file, "w") as f:
            json.dump(episode_data, f, indent=2)

        self.logger.debug(f"Saved episode to {episode_file}")

    def evaluate_policy(
        self,
        agent_policies: Dict[str, str] = None,
        num_episodes: int = 1,
        save_dir: Path = None,
        gif_fps: int = 30,
    ) -> Dict:
        """
        Evaluate a policy by running episodes and recording detailed metrics.

        This method is designed for evaluation, not training. It:
        - Runs episodes with the given policy
        - Saves GIFs of all episodes
        - Records detailed logs with local/global observation separation
        - Returns comprehensive evaluation metrics

        Args:
            agent_policies: Dict mapping agent_name -> policy_text
            num_episodes: Number of evaluation episodes
            save_dir: Directory to save evaluation results
            gif_fps: Frames per second for GIF

        Returns:
            Evaluation results dictionary with metrics and paths
        """
        if save_dir is None:
            save_dir = self.experiments_dir / self.exp_name / "evaluation"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting policy evaluation: {num_episodes} episodes")

        all_rewards = []
        all_lengths = []
        gif_paths = []
        episode_results = []

        for ep_idx in range(num_episodes):
            # Run episode with GIF rendering
            episode_data = self._run_single_episode(
                episode_id=ep_idx,
                iteration=-1,  # Use -1 for evaluation
                agent_policies=agent_policies,
                save_gif=True,
                gif_fps=gif_fps,
            )

            all_rewards.append(episode_data['total_reward'])
            all_lengths.append(len(episode_data['transitions']))

            if episode_data.get('gif_path'):
                gif_paths.append(episode_data['gif_path'])

            episode_results.append({
                'episode_id': ep_idx,
                'total_reward': episode_data['total_reward'],
                'num_steps': len(episode_data['transitions']),
                'gif_path': episode_data.get('gif_path'),
            })

            self.logger.info(
                f"Eval episode {ep_idx+1}/{num_episodes}: "
                f"reward={episode_data['total_reward']:.2f}, "
                f"steps={len(episode_data['transitions'])}"
            )

        # Calculate evaluation metrics
        eval_metrics = {
            'num_episodes': num_episodes,
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'min_reward': float(np.min(all_rewards)),
            'max_reward': float(np.max(all_rewards)),
            'mean_length': float(np.mean(all_lengths)),
            'std_length': float(np.std(all_lengths)),
            'all_rewards': all_rewards,
            'all_lengths': all_lengths,
            'gif_paths': gif_paths,
            'episode_results': episode_results,
        }

        # Save evaluation summary
        summary_file = save_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(eval_metrics, f, indent=2)

        self.logger.info(
            f"Evaluation complete: mean_reward={eval_metrics['mean_reward']:.2f} "
            f"(±{eval_metrics['std_reward']:.2f})"
        )

        return eval_metrics

    def render_single_episode(
        self,
        agent_policies: Dict[str, str] = None,
        save_path: Path = None,
        fps: int = 30,
    ) -> Tuple[float, int, str]:
        """
        Render a single episode and save as GIF.

        Convenience method for quickly visualizing policy behavior.

        Args:
            agent_policies: Dict mapping agent_name -> policy_text
            save_path: Path to save GIF (defaults to experiments_dir/render.gif)
            fps: Frames per second

        Returns:
            Tuple of (total_reward, num_steps, gif_path)
        """
        if save_path is None:
            save_path = self.experiments_dir / self.exp_name / "render.gif"

        episode_data = self._run_single_episode(
            episode_id=0,
            iteration=-1,
            agent_policies=agent_policies,
            save_gif=True,
            gif_fps=fps,
        )

        # Move GIF to requested location if different
        if episode_data.get('gif_path') and str(save_path) != episode_data['gif_path']:
            import shutil
            shutil.move(episode_data['gif_path'], str(save_path))

        return (
            episode_data['total_reward'],
            len(episode_data['transitions']),
            str(save_path)
        )
