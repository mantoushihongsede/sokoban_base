from llm_protocol.deepseek_client import DeepSeekClient
from llm_protocol.openai_client import OpenAIClient
from llm_protocol.gemini_client import GeminiClient
from llm_protocol.claude_client import ClaudeClient


def create_client(provider: str, api_key: str, base_url: str = None):
    provider = provider.lower()

    if provider == "deepseek":
        return DeepSeekClient(api_key=api_key, base_url=base_url)
    elif provider == "openai":
        return OpenAIClient(api_key=api_key, base_url=base_url)
    elif provider == "gemini":
        return GeminiClient(api_key=api_key, base_url=base_url)
    elif provider == "claude":
        return ClaudeClient(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unsupported provider: {provider}")