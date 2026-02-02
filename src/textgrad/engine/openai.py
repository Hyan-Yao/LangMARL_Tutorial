try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    raise ImportError(
        "If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables."
    )

import base64
import json
import os
from typing import List, Union

import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import CachedEngine, EngineLM
from .engine_utils import get_image_type_from_bytes

# Default base URL for OLLAMA
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Check if the user set the OLLAMA_BASE_URL environment variable
if os.getenv("OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


class BaseOpenAIEngine(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        cache_path: str,
        system_prompt: str,
        model_string: str,
        is_multimodal: bool = False,
        use_cache: bool = True,
    ):
        super().__init__(cache_path=cache_path)
        self.system_prompt = system_prompt
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.use_cache = use_cache

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str = None,
        **kwargs,
    ):
        if isinstance(content, str):
            return self._generate_from_single_prompt(
                content, system_prompt=system_prompt, **kwargs
            )

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError(
                    "Multimodal generation is only supported for Claude-3 and beyond."
                )

            return self._generate_from_multiple_input(
                content, system_prompt=system_prompt, **kwargs
            )

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature=0,
        max_tokens=20000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_or_none = self._check_cache(sys_prompt_arg + prompt)
            # Skip empty cached responses (from previous API failures)
            if cache_or_none is not None and cache_or_none.strip():
                return cache_or_none

        # Model-specific API parameter handling:
        # - Reasoning models (o1, o3, gpt-5): no system role, no temperature/top_p, use max_completion_tokens
        # - gpt-5-mini: NOT a reasoning model but uses max_completion_tokens instead of max_tokens
        model_lower = self.model_string.lower()
        is_reasoning_model = (
            "o1" in model_lower or
            "o3" in model_lower or
            (model_lower == "gpt-5" or (model_lower.startswith("gpt-5-") and "mini" not in model_lower))
        )
        uses_max_completion_tokens = is_reasoning_model or "gpt-5-mini" in model_lower

        # Build API parameters
        if is_reasoning_model:
            # Reasoning models don't support system role, merge into user message
            combined_prompt = f"{sys_prompt_arg}\n\n{prompt}"
            api_params = {
                "model": self.model_string,
                "messages": [
                    {"role": "user", "content": combined_prompt},
                ],
                "max_completion_tokens": max_tokens,
            }
        else:
            api_params = {
                "model": self.model_string,
                "messages": [
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
            }
            # gpt-5-mini does not support temperature parameter (only default value 1)
            if "gpt-5-mini" not in model_lower:
                api_params["temperature"] = temperature
                api_params["top_p"] = top_p
            # gpt-5-mini uses max_completion_tokens but is not a reasoning model
            if uses_max_completion_tokens:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**api_params)

        response = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(sys_prompt_arg + prompt, response)

        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API."""
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode("utf-8")
                formatted_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}"
                        },
                    }
                )
            elif isinstance(item, str):
                formatted_content.append({"type": "text", "text": item})
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        if self.use_cache:
            cache_or_none = self._check_cache(cache_key)
            # Skip empty cached responses (from previous API failures)
            if cache_or_none is not None and cache_or_none.strip():
                return cache_or_none

        # Model-specific API parameter handling:
        # - Reasoning models (o1, o3, gpt-5): no system role, no temperature/top_p, use max_completion_tokens
        # - gpt-5-mini: NOT a reasoning model but uses max_completion_tokens instead of max_tokens
        model_lower = self.model_string.lower()
        is_reasoning_model = (
            "o1" in model_lower or
            "o3" in model_lower or
            (model_lower == "gpt-5" or (model_lower.startswith("gpt-5-") and "mini" not in model_lower))
        )
        uses_max_completion_tokens = is_reasoning_model or "gpt-5-mini" in model_lower

        # Build API parameters
        if is_reasoning_model:
            # Reasoning models don't support system role, prepend system prompt as text
            combined_content = [{"type": "text", "text": sys_prompt_arg}] + formatted_content
            api_params = {
                "model": self.model_string,
                "messages": [
                    {"role": "user", "content": combined_content},
                ],
                "max_completion_tokens": max_tokens,
            }
        else:
            api_params = {
                "model": self.model_string,
                "messages": [
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
            }
            # gpt-5-mini does not support temperature parameter (only default value 1)
            if "gpt-5-mini" not in model_lower:
                api_params["temperature"] = temperature
                api_params["top_p"] = top_p
            # gpt-5-mini uses max_completion_tokens but is not a reasoning model
            if uses_max_completion_tokens:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**api_params)

        response_text = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text


class ChatOpenAI(BaseOpenAIEngine):
    def __init__(
        self,
        model_string: str = "gpt-3.5-turbo-0613",
        system_prompt: str = BaseOpenAIEngine.DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: str = None,
        api_key: str = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Initialize ChatOpenAI engine with support for multiple providers.

        All providers use OpenAI-compatible API format.

        :param model_string: Model identifier to pass to the API
        :param system_prompt: Default system prompt
        :param is_multimodal: Whether the model supports multimodal input
        :param base_url: Custom base URL for the API (e.g., for Gemini, Llama, Qwen providers)
        :param api_key: API key (if None, will use environment variable)
        :param api_key_env_var: Environment variable name for API key (default: OPENAI_API_KEY)
        :param use_cache: Whether to use disk cache for responses (default: True)
        """
        root = platformdirs.user_cache_dir("textgrad")
        # Create unique cache path based on base_url to avoid conflicts
        cache_suffix = model_string.replace("/", "_").replace(":", "_")
        if base_url:
            # Extract domain for cache file naming
            from urllib.parse import urlparse
            domain = urlparse(base_url).netloc.replace(".", "_").replace(":", "_")
            cache_suffix = f"{domain}_{cache_suffix}"
        cache_path = os.path.join(root, f"cache_openai_{cache_suffix}.db")

        super().__init__(cache_path, system_prompt, model_string, is_multimodal, use_cache)

        self.base_url = base_url
        self.api_key_env_var = api_key_env_var

        # Resolve API key
        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = os.getenv(api_key_env_var)

        # Handle different scenarios
        if base_url:
            # Custom base URL provided (Gemini, Llama, Qwen, DeepSeek, etc.)
            if base_url == OLLAMA_BASE_URL or "localhost" in base_url or "127.0.0.1" in base_url:
                # Local Ollama instance - use dummy key if not provided
                if resolved_api_key is None:
                    resolved_api_key = "ollama"
            elif resolved_api_key is None:
                raise ValueError(
                    f"API key required for base URL '{base_url}'. "
                    f"Set '{api_key_env_var}' environment variable or provide 'api_key' parameter."
                )
            self.client = OpenAI(base_url=base_url, api_key=resolved_api_key)
        else:
            # Default OpenAI endpoint
            if resolved_api_key is None:
                raise ValueError(
                    f"Please set the {api_key_env_var} environment variable "
                    "or provide 'api_key' parameter if you'd like to use OpenAI models."
                )
            self.client = OpenAI(api_key=resolved_api_key)


class AzureChatOpenAI(BaseOpenAIEngine):
    def __init__(
        self,
        model_string="gpt-35-turbo",
        system_prompt=BaseOpenAIEngine.DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        **kwargs,
    ):
        """
        Initializes an interface for interacting with Azure's OpenAI models.

        This class extends the EngineLM and CachedEngine classes to use Azure's OpenAI API instead of OpenAI's API. It sets up the necessary client with the appropriate API version, API key, and endpoint from environment variables.

        :param model_string: The model identifier for Azure OpenAI. Defaults to 'gpt-35-turbo'.
        :param system_prompt: The default system prompt to use when generating responses. Defaults to the default system prompt.
        :param is_multimodal: Whether this is a multimodal model. Defaults to False.
        :param kwargs: Additional keyword arguments.

        Environment variables:
        - AZURE_OPENAI_API_KEY: The API key for authenticating with Azure OpenAI.
        - AZURE_OPENAI_API_BASE: The base URL for the Azure OpenAI API.
        - AZURE_OPENAI_API_VERSION: The API version to use. Defaults to '2023-07-01-preview' if not set.

        Raises:
            ValueError: If the AZURE_OPENAI_API_KEY environment variable is not set.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(
            root, f"cache_azure_{model_string}.db"
        )  # Changed cache path to differentiate from OpenAI cache

        super().__init__(cache_path, system_prompt, model_string, is_multimodal)

        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        if os.getenv("AZURE_OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, and AZURE_OPENAI_API_VERSION environment variables if you'd like to use Azure OpenAI models."
            )

        self.client = AzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            azure_deployment=model_string,
        )
