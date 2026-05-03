import os
from typing import Optional

from .base_client import BaseLLMClient

# Providers that use the OpenAI-compatible chat completions API
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek",
    "qwen", "qwen-cn",
    "glm", "glm-cn",
    "minimax", "minimax-cn",
    "ollama", "openrouter",
)

# z.ai's Anthropic-compatible endpoint (used by Claude Code via cc-switch).
# Same provider, different billing surface than the PaaS/OpenAI-compatible
# endpoint — covered by the Coding Plan subscription instead of pay-as-you-go.
_GLM_ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic"
_GLM_ANTHROPIC_KEY_ENVS = ("ZAI_API_KEY", "ZHIPU_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider modules are imported lazily so that simply importing this
    factory (e.g. during test collection) does not pull in heavy LLM SDKs
    or fail when their API keys are absent.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower in _OPENAI_COMPATIBLE:
        from .openai_client import OpenAIClient
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "glm-anthropic":
        from .anthropic_client import AnthropicClient
        if "api_key" not in kwargs:
            for env_name in _GLM_ANTHROPIC_KEY_ENVS:
                key = os.environ.get(env_name)
                if key:
                    kwargs["api_key"] = key
                    break
        return AnthropicClient(
            model,
            base_url or _GLM_ANTHROPIC_BASE_URL,
            provider="glm-anthropic",
            **kwargs,
        )

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
