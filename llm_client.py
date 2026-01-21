"""
LLM Client for the Autograder - supports Ollama, OpenAI, and Anthropic.
"""
import json
import requests
from typing import Optional, List, Dict, Any

from config import Config

# Timeout settings (in seconds)
OLLAMA_TIMEOUT = 300  # 5 minutes for complex evaluations


class LLMClient:
    """Unified client for LLM API interactions."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider ('ollama', 'openai', or 'anthropic').
            model: Model name to use.
            api_key: API key for the provider (not used for ollama).
        """
        self.provider = provider or Config.LLM_PROVIDER
        self.model = model
        self.api_key = api_key
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "ollama":
            if not self.model:
                self.model = Config.OLLAMA_MODEL
            return None
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                if not self.model:
                    self.model = Config.OPENAI_MODEL
                key = self.api_key or Config.OPENAI_API_KEY
                return OpenAI(api_key=key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                if not self.model:
                    self.model = Config.ANTHROPIC_MODEL
                key = self.api_key or Config.ANTHROPIC_API_KEY
                return anthropic.Anthropic(api_key=key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Send a chat completion request."""
        max_tokens = max_tokens or Config.MAX_TOKENS
        temperature = temperature or Config.TEMPERATURE

        if self.provider == "ollama":
            return self._ollama_chat(messages, system_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._openai_chat(messages, system_prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            return self._anthropic_chat(messages, system_prompt, max_tokens, temperature)

    def _ollama_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle Ollama chat completion with streaming."""
        all_messages = []

        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})

        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'message' in chunk and 'content' in chunk['message']:
                            full_response += chunk['message']['content']
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue

            return full_response

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}. "
                "Please make sure Ollama is running (ollama serve)."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {OLLAMA_TIMEOUT} seconds."
            )
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle OpenAI chat completion."""
        all_messages = []

        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})

        all_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle Anthropic chat completion."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=messages,
            temperature=temperature
        )

        return response.content[0].text

    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """Get available models for a provider."""
        if provider == "ollama":
            return Config.get_available_ollama_models()
        elif provider == "openai":
            return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        elif provider == "anthropic":
            return ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-20241022"]
        return []

    @staticmethod
    def test_connection(provider: str, model: str = None, api_key: str = None) -> Dict[str, Any]:
        """Test connection to a provider."""
        try:
            client = LLMClient(provider=provider, model=model, api_key=api_key)
            response = client.chat(
                messages=[{"role": "user", "content": "Say 'ready' in one word."}],
                max_tokens=10
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
