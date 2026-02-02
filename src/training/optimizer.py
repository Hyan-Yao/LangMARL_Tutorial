"""TextGrad optimizer wrapper for Pistonball policy optimization"""

import textgrad as tg
from typing import List, Dict, Optional, Union
import re
import logging
from pathlib import Path
from datetime import datetime

from .llm_config import LLMConfig, get_llm_config


class PolicyOptimizer:
    """TextGrad policy optimizer for Pistonball"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        llm_config: Optional[Union[LLMConfig, Dict, str]] = None
    ):
        """
        Initialize optimizer

        Args:
            model: Model name for TextGrad backward engine (used if llm_config is None)
            llm_config: LLM configuration (LLMConfig object, dict, or predefined name/path)
                       Takes precedence over 'model' parameter
        """
        self.llm_config = self._resolve_llm_config(model, llm_config)
        self.model = self.llm_config.model_string

        # Set up TextGrad backward engine using LLMConfig
        self._setup_backward_engine()

        self.current_log_handler = None
        self.current_iteration_dir = None

        # Track credit assignment responses for saving
        self.credit_responses = []

    def _resolve_llm_config(
        self,
        model: str,
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
            return get_llm_config(model)

    def _setup_backward_engine(self):
        """Set up TextGrad backward engine using the LLM configuration."""
        from textgrad.engine import get_engine_from_config
        engine = get_engine_from_config(self.llm_config)
        tg.set_backward_engine(engine, override=True)

    def setup_textgrad_logger(self, iteration_dir: Path):
        """
        Setup TextGrad logger to write to the iteration directory

        Args:
            iteration_dir: Path to the current iteration directory
        """
        # Get the textgrad logger
        textgrad_logger = logging.getLogger('textgrad')

        # Remove existing file handler if any
        if self.current_log_handler is not None:
            textgrad_logger.removeHandler(self.current_log_handler)
            self.current_log_handler.close()
            self.current_log_handler = None

        # Create new log file in iteration directory
        iteration_dir = Path(iteration_dir)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        self.current_iteration_dir = iteration_dir

        log_file = iteration_dir / f"textgrad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Create and add new file handler
        file_handler = logging.FileHandler(log_file)

        # Use the same formatter as textgrad's default
        try:
            json_formatter = tg.CustomJsonFormatter()
            file_handler.setFormatter(json_formatter)
        except AttributeError:
            # Fallback if CustomJsonFormatter is not available
            file_handler.setFormatter(logging.Formatter('%(message)s'))

        file_handler.setLevel(logging.INFO)

        textgrad_logger.addHandler(file_handler)
        self.current_log_handler = file_handler

        # Reset credit responses tracking
        self.credit_responses = []

        print(f"TextGrad logger configured: {log_file}")

    def cleanup_textgrad_logger(self):
        """
        Cleanup TextGrad logger by removing the current file handler
        """
        if self.current_log_handler is not None:
            textgrad_logger = logging.getLogger('textgrad')
            textgrad_logger.removeHandler(self.current_log_handler)
            self.current_log_handler.close()
            self.current_log_handler = None

    def save_credit_responses(self, output_path: Path = None):
        """
        Save credit assignment responses to a file for analysis.

        Args:
            output_path: Optional path to save file. If None, uses current iteration dir.

        Returns:
            Path to the saved file, or None if no responses to save
        """
        if not self.credit_responses:
            return None

        if output_path is None and self.current_iteration_dir:
            output_path = self.current_iteration_dir / "credit_assignment_responses.txt"
        elif output_path is None:
            output_path = Path("credit_assignment_responses.txt")

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CREDIT ASSIGNMENT LLM RESPONSES\n")
            f.write("="*80 + "\n\n")

            for idx, resp in enumerate(self.credit_responses):
                f.write(f"{'='*40}\n")
                f.write(f"Response {idx + 1}\n")
                f.write(f"{'='*40}\n\n")

                f.write("--- PROMPT (first 500 chars) ---\n")
                f.write(resp['prompt'] + "\n\n")

                f.write("--- LLM RESPONSE ---\n")
                f.write(resp['response'] + "\n\n")

                f.write("--- PARSING RESULTS ---\n")
                f.write(f"Total agents: {resp['num_agents']}\n")
                f.write(f"Successfully parsed: {resp['successfully_parsed']}\n\n")

                f.write("--- PARSED EVALUATIONS ---\n")
                for agent, eval_text in resp['parsed_evals'].items():
                    f.write(f"\n[{agent}]:\n")
                    f.write(eval_text + "\n")

                f.write("\n" + "-"*80 + "\n\n")

        print(f"Saved credit assignment responses to {output_path}")
        return output_path

    def initialize_policies(
        self,
        agent_policies: dict
    ) -> dict:
        """
        Initialize policies as TextGrad variables

        All agents are active, each with their own policy.

        Args:
            agent_policies: Dict mapping agent_name -> policy text
                e.g., {"piston_0": "policy...", "piston_1": "policy...", ...}

        Returns:
            Dict mapping agent_name -> tg.Variable
        """
        agent_policy_vars = {}
        for agent_name, policy_text in agent_policies.items():
            agent_policy_vars[agent_name] = tg.Variable(
                policy_text,
                requires_grad=True,
                role_description=f"Policy for {agent_name}"
            )

        return agent_policy_vars

    def accumulate_gradients(
        self,
        policy_vars: List[tg.Variable],
        feedback: str,
        reduce_group_id: int = None
    ):
        """
        Accumulate gradients from feedback with optional batch reduction

        This method computes gradients for a single episode and accumulates them
        on the policy variables. When reduce_group_id is provided, gradients from
        multiple episodes with the same ID will be automatically reduced/aggregated
        by textgrad's native mechanism during backward pass.

        Args:
            policy_vars: List of policy variables
            feedback: Evaluation prompt for TextLoss
            reduce_group_id: Optional ID for batch gradient reduction group
        """
        # Create intermediate action variable that depends on policy_vars
        # This establishes the computation graph: loss -> action_var -> policy_vars
        action_var = tg.Variable(
            "Actions taken by pistons during episode",
            predecessors=policy_vars,
            role_description="episode actions for gradient computation"
        )

        # Create TextLoss with evaluation prompt
        loss_fn = tg.TextLoss(feedback)
        loss = loss_fn(action_var)

        # Trigger TextGrad backward pass
        loss.backward()

        # Manually propagate gradients from action_var to policy_vars
        for policy_var in policy_vars:
            for grad in action_var.gradients:
                policy_var.gradients.add(grad)
                policy_var.gradients_context[grad] = {
                    "context": f"Gradient feedback:\n{grad.value}",
                    "response_desc": action_var.get_role_description(),
                    "variable_desc": policy_var.get_role_description()
                }

                # Add reduce metadata for batch gradient aggregation
                if reduce_group_id is not None:
                    from textgrad.autograd.algebra import _reduce_gradients_mean
                    reduce_meta = {"op": _reduce_gradients_mean, "id": reduce_group_id}
                    grad._reduce_meta.append(reduce_meta)
                    policy_var._reduce_meta.append(reduce_meta)

    def accumulate_gradients_credit(
        self,
        policy_vars: List[tg.Variable],
        feedback: str,
        agent_names: List[str] = None,
        reduce_group_id: int = None
    ):
        """
        Accumulate gradients for credit assignment paradigm with optional batch reduction.
        One LLM call evaluates ALL agents, then parse and distribute gradients.

        Args:
            policy_vars: List of policy variables for each agent
            feedback: Evaluation prompt that asks for formatted joint evaluation
            agent_names: List of agent names corresponding to policy_vars
            reduce_group_id: Optional ID for batch gradient reduction group
        """
        if agent_names is None:
            agent_names = [f"agent_{i}" for i in range(len(policy_vars))]

        # Call LLM once to get joint evaluation for all agents
        from textgrad.config import SingletonBackwardEngine
        from textgrad import logger
        engine = SingletonBackwardEngine().get_engine()

        # Log the backward pass initiation
        logger.info(
            "Credit Assignment Backward Pass - Joint Evaluation",
            extra={
                "event_type": "credit_backward_start",
                "num_agents": len(policy_vars),
                "agent_names": agent_names,
                "reduce_group_id": reduce_group_id,
                "evaluation_prompt": feedback
            }
        )

        # Get the joint evaluation response
        joint_response = engine(feedback)
        

        # Log the LLM response
        logger.info(
            "Credit Assignment - Joint LLM Response",
            extra={
                "event_type": "credit_llm_response",
                "joint_response": joint_response,
                "response_length": len(joint_response)
            }
        )

        # Parse the response to extract separate evaluations for each agent
        agent_evals = self._parse_agent_credit_evaluation(joint_response, agent_names)

        # Log the parsed gradients
        logger.info(
            "Credit Assignment - Parsed Agent Gradients",
            extra={
                "event_type": "credit_gradient_parsing",
                "num_agents_parsed": len(agent_evals),
                "parse_success": len(agent_evals) == len(agent_names)
            }
        )

        # Save credit response for later file export
        self.credit_responses.append({
            "prompt": feedback,
            "response": joint_response,
            "parsed_evals": agent_evals,
            "num_agents": len(agent_names),
            "successfully_parsed": len(agent_evals)
        })

        # Create gradient variables for each agent
        for policy_var, agent_name in zip(policy_vars, agent_names):
            eval_text = agent_evals.get(agent_name, joint_response)  # Fallback to full response

            grad_var = tg.Variable(
                eval_text,
                requires_grad=False,
                role_description=f"credit assignment feedback for {agent_name}"
            )

            # Add gradients directly to policy variables
            policy_var.gradients.add(grad_var)
            policy_var.gradients_context[grad_var] = {
                "context": f"{agent_name} credit assignment feedback:\n{eval_text}",
                "response_desc": f"credit assignment feedback for {agent_name}",
                "variable_desc": policy_var.get_role_description()
            }

            # Add reduce metadata for batch gradient aggregation
            if reduce_group_id is not None:
                from textgrad.autograd.algebra import _reduce_gradients_mean
                reduce_meta = {"op": _reduce_gradients_mean, "id": reduce_group_id}
                grad_var._reduce_meta.append(reduce_meta)
                policy_var._reduce_meta.append(reduce_meta)

        # Log gradient accumulation
        logger.info(
            "Credit Assignment - Gradients Accumulated",
            extra={
                "event_type": "credit_gradient_accumulation",
                "gradient_counts": [len(p.gradients) for p in policy_vars],
                "num_agents": len(policy_vars)
            }
        )

    @staticmethod
    def _parse_agent_credit_evaluation(response: str, agent_names: List[str]) -> dict:
        """
        Parse credit assignment evaluation into separate per-agent feedbacks.

        Supports multiple formats:
        1. [PISTON_0 EVALUATION] ... [PISTON_1 EVALUATION] ...
        2. [PISTON 0 EVALUATION] ... (with space)
        3. piston_0: ... piston_1: ...
        4. **piston_0**: ... (markdown bold)

        Args:
            response: LLM response containing all agent evaluations
            agent_names: List of agent names to look for (e.g., ['piston_0', 'piston_1', ...])

        Returns:
            Dict mapping agent_name -> feedback_text
        """
        agent_evals = {}

        # Sort agent names to ensure we process them in order (piston_0, piston_1, ...)
        sorted_agents = sorted(agent_names, key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        # Try to find sections marked with agent markers like [PISTON_0 EVALUATION]
        for i, agent_name in enumerate(sorted_agents):
            # Extract piston number
            try:
                piston_num = agent_name.split('_')[1]
            except IndexError:
                piston_num = str(i)

            # Build multiple patterns to match different formats
            patterns = [
                # [PISTON_0 EVALUATION] or [PISTON 0 EVALUATION]
                rf'\[PISTON[_\s]?{piston_num}\s*EVALUATION\](.*?)(?=\[PISTON[_\s]?\d+\s*EVALUATION\]|\Z)',
                # **piston_0**: or **PISTON_0**:
                rf'\*\*{agent_name}\*\*\s*:?\s*(.*?)(?=\*\*piston_\d+\*\*|\Z)',
                # piston_0: followed by content
                rf'{agent_name}\s*:\s*(.*?)(?=piston_\d+\s*:|\Z)',
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content:  # Only use non-empty matches
                        agent_evals[agent_name] = content
                        break

        # If we got less than half the agents, try numbered format (1. ..., 2. ...)
        if len(agent_evals) < len(agent_names) // 2:
            # Try to match numbered sections
            numbered_pattern = rf'(\d+)\.\s*(?:piston[_\s]?(\d+)|agent[_\s]?(\d+))?\s*[:\-]?\s*(.*?)(?=\d+\.\s*(?:piston|agent)|\Z)'
            matches = re.findall(numbered_pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                num_idx = int(match[0]) - 1  # Convert 1-indexed to 0-indexed
                piston_num = match[1] or match[2] or str(num_idx)
                content = match[3].strip()
                agent_name = f"piston_{piston_num}"
                if agent_name in agent_names and agent_name not in agent_evals and content:
                    agent_evals[agent_name] = content

        # For agents without specific feedback, assign based on their group
        # (left/middle/right) if group-level feedback exists
        if len(agent_evals) < len(agent_names):
            # Check for group-level evaluations
            group_patterns = {
                'left': rf'\[LEFT\s*PISTONS?\s*EVALUATION\](.*?)(?=\[(?:MIDDLE|RIGHT)\s*PISTONS?\s*EVALUATION\]|\Z)',
                'middle': rf'\[MIDDLE\s*PISTONS?\s*EVALUATION\](.*?)(?=\[(?:LEFT|RIGHT)\s*PISTONS?\s*EVALUATION\]|\Z)',
                'right': rf'\[RIGHT\s*PISTONS?\s*EVALUATION\](.*?)(?=\[(?:LEFT|MIDDLE)\s*PISTONS?\s*EVALUATION\]|\Z)',
            }

            group_evals = {}
            for group_name, pattern in group_patterns.items():
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    group_evals[group_name] = match.group(1).strip()

            if group_evals:
                # Assign group feedback to individual pistons based on position
                num_agents = len(agent_names)
                third = num_agents // 3
                for agent_name in agent_names:
                    if agent_name in agent_evals:
                        continue
                    try:
                        idx = int(agent_name.split('_')[1])
                        if idx < third:
                            group = 'left'
                        elif idx < 2 * third:
                            group = 'middle'
                        else:
                            group = 'right'
                        if group in group_evals:
                            agent_evals[agent_name] = f"[From {group} group feedback]\n{group_evals[group]}"
                    except (IndexError, ValueError):
                        pass

        # If still no individual parsing, use global response for all
        if len(agent_evals) == 0:
            print(f"Warning: Could not parse credit assignment for individual agents, using global response")
            for agent_name in agent_names:
                agent_evals[agent_name] = response
        elif len(agent_evals) < len(agent_names):
            # Fill missing agents with global response
            print(f"Warning: Only parsed {len(agent_evals)}/{len(agent_names)} agents, filling missing with global response")
            for agent_name in agent_names:
                if agent_name not in agent_evals:
                    agent_evals[agent_name] = f"[Global feedback - specific evaluation not found]\n{response}"

        return agent_evals

    def optimize_step(
        self,
        policy_vars: List[tg.Variable]
    ) -> List[str]:
        """
        Execute optimization step with automatic gradient reduction

        When policy variables have accumulated multiple gradients with reduce_meta,
        textgrad will automatically aggregate them using the specified reduction
        operation (e.g., _reduce_gradients_mean) before updating the parameters.

        Args:
            policy_vars: List of policy variables

        Returns:
            List of optimized policy strings
        """
        from textgrad import logger
        from collections import defaultdict

        # Log gradient reduction information before optimization step
        for idx, policy_var in enumerate(policy_vars):
            # Check if this variable has gradients that need reduction
            if len(policy_var.gradients) > 1 and policy_var._reduce_meta:
                # Group gradients by reduce_group_id
                id_to_gradient_set = defaultdict(set)
                for gradient in policy_var.gradients:
                    for reduce_item in gradient._reduce_meta:
                        id_to_gradient_set[reduce_item["id"]].add(gradient)

                # Log the reduction operation
                logger.info(
                    f"Batch Gradient Reduction - Policy {idx}",
                    extra={
                        "event_type": "batch_gradient_reduction",
                        "policy_index": idx,
                        "policy_role": policy_var.get_role_description(),
                        "total_gradients": len(policy_var.gradients),
                        "num_reduce_groups": len(id_to_gradient_set),
                        "reduce_groups": {
                            str(group_id): len(grads)
                            for group_id, grads in id_to_gradient_set.items()
                        },
                        "gradients_preview": [
                            {
                                "role": g.get_role_description(),
                                "value_preview": g.value,
                                "reduce_meta_count": len(g._reduce_meta)
                            }
                            for g in list(policy_var.gradients)[:3]
                        ]
                    }
                )
            elif len(policy_var.gradients) > 0:
                # Log even when no reduction is needed
                logger.info(
                    f"Gradient Status - Policy {idx}",
                    extra={
                        "event_type": "gradient_status",
                        "policy_index": idx,
                        "policy_role": policy_var.get_role_description(),
                        "total_gradients": len(policy_var.gradients),
                        "requires_reduction": False
                    }
                )

        # Log optimizer step initiation
        logger.info(
            "Optimizer Step - Starting TGD",
            extra={
                "event_type": "optimizer_step_start",
                "num_parameters": len(policy_vars),
                "policy_roles": [var.get_role_description() for var in policy_vars]
            }
        )

        # Execute the optimization step
        optimizer = tg.TGD(parameters=policy_vars)
        optimizer.step()

        # Log optimizer step completion
        logger.info(
            "Optimizer Step - Completed",
            extra={
                "event_type": "optimizer_step_completed",
                "updated_policies_preview": [
                    var.value for var in policy_vars
                ]
            }
        )

        return [var.value for var in policy_vars]

    def clear_gradients(self, policy_vars: List[tg.Variable]):
        """
        Clear accumulated gradients and reduction metadata

        Args:
            policy_vars: List of policy variables
        """
        for var in policy_vars:
            var.gradients = set()
            var.gradients_context = {}
            var._reduce_meta = []
