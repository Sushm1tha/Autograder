"""
Configuration settings for the Autograder Agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for the autograder."""

    # LLM Settings - supports: ollama, openai, anthropic
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama Settings (Local)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

    # OpenAI Settings (API)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Anthropic Settings (API)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Agent Settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

    # Grading Settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    SUPPORTED_EXTENSIONS = [".py", ".ipynb", ".zip", ".pptx"]

    # Tool Settings (for ReAct orchestrator)
    TOOL_EXECUTION_TIMEOUT = int(os.getenv("TOOL_EXECUTION_TIMEOUT", "30"))
    TOOL_MAX_OUTPUT_LENGTH = int(os.getenv("TOOL_MAX_OUTPUT_LENGTH", "5000"))
    ORCHESTRATOR_MAX_ITERATIONS = int(os.getenv("ORCHESTRATOR_MAX_ITERATIONS", "5"))

    # Available providers
    PROVIDERS = ["ollama", "openai", "anthropic"]

    # Available tools
    AVAILABLE_TOOLS = ["execute_code", "analyze_code", "run_tests"]

    @classmethod
    def get_model(cls):
        """Get the appropriate model based on provider."""
        if cls.LLM_PROVIDER == "openai":
            return cls.OPENAI_MODEL
        elif cls.LLM_PROVIDER == "anthropic":
            return cls.ANTHROPIC_MODEL
        elif cls.LLM_PROVIDER == "ollama":
            return cls.OLLAMA_MODEL
        return ""

    @classmethod
    def is_ollama_available(cls):
        """Check if Ollama is running."""
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def get_available_ollama_models(cls):
        """Get list of available Ollama models."""
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except Exception:
            pass
        return []
