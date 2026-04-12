import os

from inference import build_openai_compatible_policy
from src.executive_assistant.config import OpenRouterConfig


def test_inference_prefers_openrouter_api_with_openai_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("MODEL_NAME", "wrong-model")
    policy = build_openai_compatible_policy()
    assert policy.config.api_key == "openrouter-key"
    assert policy.config.base_url == "https://openrouter.ai/api/v1"
    assert policy.config.model_name == "google/gemma-4-31b-it"


def test_openrouter_config_accepts_hackathon_env_names(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MODEL_NAME", "google/gemma-4-31b-it")
    config = OpenRouterConfig.from_env()
    assert config.api_key == "test-key"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.model_name == "google/gemma-4-31b-it"


def test_inference_builds_openai_compatible_policy(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MODEL_NAME", "google/gemma-4-31b-it")
    policy = build_openai_compatible_policy()
    assert policy.config.api_key == "test-key"
    assert policy.config.base_url == "https://openrouter.ai/api/v1"
    assert policy.config.model_name == "google/gemma-4-31b-it"
