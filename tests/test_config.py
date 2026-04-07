import os

from src.executive_assistant.config import OpenRouterConfig, load_env_file


def test_load_env_file_sets_openrouter_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env.training"
    env_file.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=test-key",
                "OPENROUTER_MODEL=google/gemma-4-31b-it",
                "OPENROUTER_SITE_URL=http://localhost:8888",
            ]
        )
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)

    loaded = load_env_file(env_file)
    config = OpenRouterConfig.from_env()

    assert loaded is True
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"
    assert config.model_name == "google/gemma-4-31b-it"
