from .base import EngineLM, CachedEngine
# from textgrad.engine_experimental.litellm import LiteLLMEngine

__ENGINE_NAME_SHORTCUTS__ = {
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "sonnet-3.5": "claude-3-5-sonnet-20240620",
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
    "vllm-llama-3-8b": "vllm-meta-llama/Meta-Llama-3-8B-Instruct",
}

# Any better way to do this?
__MULTIMODAL_ENGINES__ = ["gpt-4-turbo",
                          "gpt-4o",
                          "gpt-4o-mini",
                          "gpt-5",
                          "o1",
                          "gemini-1.5-pro",
                          "gemini-1.5-flash",
                          "gemini-2.0-flash",
                          "claude-3-5-sonnet-20240620",
                          "claude-3-opus-20240229",
                          "claude-3-sonnet-20240229",
                          "claude-3-haiku-20240307",
                          "gpt-4-turbo-2024-04-09",
                          ]

def _check_if_multimodal(engine_name: str):
    return any([name == engine_name for name in __MULTIMODAL_ENGINES__])

def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}")


def get_engine_from_config(llm_config) -> EngineLM:
    """
    Create an engine from an LLMConfig object.

    This function supports all providers with OpenAI-compatible APIs:
    - OpenAI (gpt-4o, gpt-5, etc.)
    - Google Gemini (via OpenAI-compatible endpoint)
    - Llama (via Together, etc.)
    - Qwen (via Together, etc.)
    - DeepSeek
    - Local Ollama

    Args:
        llm_config: LLMConfig object with model configuration

    Returns:
        EngineLM instance
    """
    from .openai import ChatOpenAI

    return ChatOpenAI(
        model_string=llm_config.model_string,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        api_key_env_var=llm_config.api_key_env_var,
        is_multimodal=llm_config.is_multimodal,
        **llm_config.extra_params
    )


def get_engine(engine_name: str, **kwargs) -> EngineLM:
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    if "seed" in kwargs and "gpt-4" not in engine_name and "gpt-3.5" not in engine_name and "gpt-35" not in engine_name:
        raise ValueError(f"Seed is currently supported only for OpenAI engines, not {engine_name}")

    if "cache" in kwargs and "experimental" not in engine_name:
        raise ValueError(f"Cache is currently supported only for LiteLLM engines, not {engine_name}")

    # If base_url is provided, use ChatOpenAI directly (OpenAI-compatible API)
    if "base_url" in kwargs and kwargs["base_url"]:
        from .openai import ChatOpenAI
        return ChatOpenAI(
            model_string=engine_name,
            is_multimodal=kwargs.pop("is_multimodal", _check_if_multimodal(engine_name)),
            **kwargs
        )

    # check if engine_name starts with "experimental:"
    # if engine_name.startswith("experimental:"):
    #    engine_name = engine_name.split("experimental:")[1]
    #    return LiteLLMEngine(model_string=engine_name, **kwargs)
    if engine_name.startswith("azure"):
        from .openai import AzureChatOpenAI
        # remove engine_name "azure-" prefix
        engine_name = engine_name[6:]
        return AzureChatOpenAI(model_string=engine_name, **kwargs)
    elif (("gpt" in engine_name) or ("o1" in engine_name)):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "claude" in engine_name:
        from .anthropic import ChatAnthropic
        return ChatAnthropic(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "gemini" in engine_name:
        # Use OpenAI-compatible endpoint for Gemini
        from .openai import ChatOpenAI
        return ChatOpenAI(
            model_string=engine_name,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env_var="GOOGLE_API_KEY",
            is_multimodal=_check_if_multimodal(engine_name),
            **kwargs
        )
    elif "together" in engine_name:
        from .together import ChatTogether
        engine_name = engine_name.replace("together-", "")
        return ChatTogether(model_string=engine_name, **kwargs)
    elif engine_name in ["command-r-plus", "command-r", "command", "command-light"]:
        from .cohere import ChatCohere
        return ChatCohere(model_string=engine_name, **kwargs)
    elif engine_name.startswith("ollama"):
        from .openai import ChatOpenAI, OLLAMA_BASE_URL
        model_string = engine_name.replace("ollama-", "")
        return ChatOpenAI(
            model_string=model_string,
            base_url=OLLAMA_BASE_URL,
            **kwargs
        )
    elif "vllm" in engine_name:
        from .vllm import ChatVLLM
        engine_name = engine_name.replace("vllm-", "")
        return ChatVLLM(model_string=engine_name, **kwargs)
    elif "groq" in engine_name:
        from .groq import ChatGroq
        engine_name = engine_name.replace("groq-", "")
        return ChatGroq(model_string=engine_name, **kwargs)
    elif "deepseek" in engine_name:
        # DeepSeek uses OpenAI-compatible API
        from .openai import ChatOpenAI
        return ChatOpenAI(
            model_string=engine_name,
            base_url="https://api.deepseek.com/v1",
            api_key_env_var="DEEPSEEK_API_KEY",
            **kwargs
        )
    elif "llama" in engine_name.lower() or "qwen" in engine_name.lower():
        # Default to Together API for Llama and Qwen models
        from .openai import ChatOpenAI
        return ChatOpenAI(
            model_string=engine_name,
            base_url="https://api.together.xyz/v1",
            api_key_env_var="TOGETHER_API_KEY",
            **kwargs
        )
    else:
        # Fallback: try using ChatOpenAI with the engine_name as model_string
        # This allows custom model names to work with OpenAI-compatible APIs
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=engine_name, **kwargs)
