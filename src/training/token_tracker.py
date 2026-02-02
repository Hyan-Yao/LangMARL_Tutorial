"""Token usage tracking and cost estimation module"""

from typing import Dict


class TokenTracker:
    """Track token usage and estimate costs for LLM API calls"""

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.15,  # $0.15 per 1M input tokens
            "output": 0.60   # $0.60 per 1M output tokens
        },
        "gpt-4o": {
            "input": 5.00,   # $5.00 per 1M input tokens
            "output": 15.00  # $15.00 per 1M output tokens
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        },
        "claude-3-opus": {
            "input": 15.00,
            "output": 75.00
        },
        "claude-3-sonnet": {
            "input": 3.00,
            "output": 15.00
        },
        "claude-3-haiku": {
            "input": 0.25,
            "output": 1.25
        }
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        input_price: float = None,
        output_price: float = None
    ):
        """
        Initialize token tracker

        Args:
            model: Model name for pricing lookup
            input_price: Custom input price per million tokens (overrides lookup)
            output_price: Custom output price per million tokens (overrides lookup)
        """
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

        # Get pricing: custom prices take precedence over lookup
        if input_price is not None and output_price is not None:
            self.pricing = {
                "input": input_price,
                "output": output_price
            }
        elif model in self.PRICING:
            self.pricing = self.PRICING[model]
        else:
            # Default to gpt-4o-mini pricing if model not found
            print(f"Warning: Pricing not found for model '{model}', using gpt-4o-mini pricing")
            self.pricing = self.PRICING["gpt-4o-mini"]

    def add_usage(self, input_tokens: int, output_tokens: int):
        """
        Add token usage

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_stats(self) -> Dict:
        """
        Get usage statistics and cost estimate

        Returns:
            Dictionary with token counts and cost
        """
        total_tokens = self.input_tokens + self.output_tokens

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (self.input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * self.pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "cost_usd": total_cost
        }

    def reset(self):
        """Reset token counts to zero"""
        self.input_tokens = 0
        self.output_tokens = 0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a given number of tokens without adding to tracker

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost

    def get_summary_string(self) -> str:
        """
        Get a formatted summary string

        Returns:
            Formatted string with usage statistics
        """
        stats = self.get_stats()
        return (
            f"Token Usage Summary ({stats['model']}):\n"
            f"  Input tokens:  {stats['input_tokens']:,}\n"
            f"  Output tokens: {stats['output_tokens']:,}\n"
            f"  Total tokens:  {stats['total_tokens']:,}\n"
            f"  Estimated cost: ${stats['cost_usd']:.4f}"
        )
