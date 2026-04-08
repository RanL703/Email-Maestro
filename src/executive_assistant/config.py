from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(env_path: str | Path, override: bool = False) -> bool:
    path = Path(env_path)
    if not path.exists():
        return False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
    return True


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    model_name: str = "google/gemma-4-31b-it"
    base_url: str = "https://openrouter.ai/api/v1"
    site_url: str = "http://localhost:7860"
    app_name: str = "Autonomous Executive Assistant Sandbox"
    temperature: float = 0.1
    max_tokens: int = 600

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "OpenRouterConfig":
        if env_file is not None:
            load_env_file(env_file)
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or os.environ.get(
            "OPENAI_API_KEY", ""
        ).strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY or OPENAI_API_KEY is required for model access."
            )
        return cls(
            api_key=api_key,
            model_name=os.environ.get(
                "OPENROUTER_MODEL",
                os.environ.get("MODEL_NAME", "google/gemma-4-31b-it"),
            ),
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL",
                os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1"),
            ),
            site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost:7860"),
            app_name=os.environ.get(
                "OPENROUTER_APP_NAME",
                "Autonomous Executive Assistant Sandbox",
            ),
            temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "600")),
        )

    def extra_headers(self) -> dict[str, str]:
        return {
            "HTTP-Referer": self.site_url,
            "X-OpenRouter-Title": self.app_name,
        }


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    kernel_name: str = "scalerhack2-training"
    kernel_display_name: str = "Python (scalerhack2-training)"
    checkpoint_dir: str = "artifacts/checkpoints"
    trace_dir: str = "artifacts/traces"
    env_file: str = ".env.training"
    default_checkpoint_name: str = "q_policy_notebook.json"


@dataclass(frozen=True)
class AppRuntimeConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    env_file: str = ".env.app"
    checkpoint_dir: str = "artifacts/checkpoints"
    default_checkpoint_name: str = "q_policy_notebook.json"
