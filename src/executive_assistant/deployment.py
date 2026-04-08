from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.training import default_checkpoint_path, train_q_learning


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPACE_TITLE = "Project Epsilon | Executive Assistant Sandbox"
DEFAULT_HF_USERNAMES = [
    "HF_USERNAME_1",
    "HF_USERNAME_2",
    "HF_USERNAME_3",
]
DEFAULT_CHECKPOINT_NAME = "q_policy_notebook.json"
DEFAULT_STAGE_IGNORE_NAMES = {
    ".git",
    ".codex",
    ".pytest_cache",
    ".venv-app",
    ".venv-training",
    ".vscode",
    "__pycache__",
}
DEFAULT_STAGE_IGNORE_SUFFIXES = {
    ".pyc",
}
DEFAULT_STAGE_IGNORE_FILES = {
    ".env",
    ".env.app",
    ".env.hf.space",
    ".env.training",
    "training_env.executed.ipynb",
}


@dataclass(frozen=True)
class HFSpaceDeployConfig:
    repo_id: str
    title: str = DEFAULT_SPACE_TITLE
    team_name: str = "Project Epsilon"
    hf_usernames: tuple[str, ...] = tuple(DEFAULT_HF_USERNAMES)
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME
    app_port: int = 7860
    private: bool = False
    include_checkpoint: bool = True

    @property
    def repo_slug(self) -> str:
        return self.repo_id.split("/", 1)[1]

    @property
    def owner(self) -> str:
        return self.repo_id.split("/", 1)[0]

    @property
    def space_url(self) -> str:
        return f"https://huggingface.co/spaces/{self.repo_id}"

    @property
    def app_url(self) -> str:
        return f"https://{self.owner}-{self.repo_slug}.hf.space"

    @property
    def checkpoint_source_path(self) -> Path:
        return REPO_ROOT / "artifacts" / "checkpoints" / self.checkpoint_name


def parse_hf_usernames(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None or not raw_value.strip():
        return tuple(DEFAULT_HF_USERNAMES)
    usernames = [item.strip().lstrip("@") for item in raw_value.split(",") if item.strip()]
    return tuple(usernames) or tuple(DEFAULT_HF_USERNAMES)


def render_space_readme(config: HFSpaceDeployConfig) -> str:
    usernames = ", ".join(f"`@{username}`" for username in config.hf_usernames)
    checkpoint_note = (
        "A trained RL checkpoint is bundled in `artifacts/checkpoints/` so the `rl` policy "
        "is available immediately in the demo."
        if config.include_checkpoint
        else "The Space can still run the deterministic baseline immediately; add an RL checkpoint "
        "later if you want the `rl` option available in the UI."
    )
    return f"""---
title: {config.title}
emoji: "🧭"
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: {config.app_port}
pinned: false
short_description: OpenEnv executive assistant sandbox demo for judges.
---

# {config.team_name}

Discrete Hugging Face Space for the **Autonomous Executive Assistant Sandbox**, built for the **OpenEnv Scaler x Meta x PyTorch Hack**.

## Team

- Team name: `{config.team_name}`
- Hugging Face usernames: {usernames}
- Space repo: `{config.repo_id}`

Replace the placeholder usernames above once the final team accounts are ready.

## What This Space Shows

- A deterministic OpenEnv-style executive assistant environment backed by an isolated SQLite workspace
- A judge-friendly Gradio interface that replays the shared `EpisodeRunner` loop step by step
- Side-by-side policy execution for `baseline`, `rl`, and optional `openrouter`
- Visible inbox, todo, file-search, and action-log state so evaluators can inspect each mutation

## Hack Context

OpenEnv was announced by Hugging Face and Meta as an open source framework for building agent environments with typed observations, actions, and rewards. The Scaler dashboard for this hack lists the submission round as **March 25, 2026 through April 8, 2026**, with finals on **April 25-26, 2026** in Bengaluru. This Space packages our environment to match that workflow: deterministic tasks, structured actions, visible state transitions, and reproducible judge demos.

## Runtime Notes

- SDK: `docker`
- App port: `{config.app_port}`
- Entry point: `python app.py`
- Optional secret: `OPENROUTER_API_KEY`
- {checkpoint_note}

## Judge Flow

1. Open the Space and choose one of the seeded scenarios.
2. Run the deterministic `baseline` policy for a guaranteed reference trace.
3. Switch to `rl` to replay the bundled learned checkpoint.
4. Add `OPENROUTER_API_KEY` in Space secrets to enable the live model-backed path.

## References

- Hack dashboard: https://www.scaler.com/openenv-hackathon
- OpenEnv launch: https://huggingface.co/blog/openenv
- Space URL: {config.space_url}
"""


def copy_repo_for_space(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for source in REPO_ROOT.iterdir():
        if source.name in DEFAULT_STAGE_IGNORE_NAMES:
            continue
        if source.name in DEFAULT_STAGE_IGNORE_FILES:
            continue
        if source.suffix in DEFAULT_STAGE_IGNORE_SUFFIXES:
            continue
        destination = stage_dir / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".env",
                    ".env.app",
                    ".env.hf.space",
                    ".env.training",
                    "training_env.executed.ipynb",
                ),
            )
        else:
            shutil.copy2(source, destination)


def ensure_checkpoint(config: HFSpaceDeployConfig, stage_dir: Path) -> Path | None:
    if not config.include_checkpoint:
        return None

    destination = stage_dir / "artifacts" / "checkpoints" / config.checkpoint_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    source = config.checkpoint_source_path
    if source.exists():
        shutil.copy2(source, destination)
        return destination

    policy, _ = train_q_learning(episodes=120, epsilon=0.12, teacher=BaselineAgent())
    return policy.save(destination)


def stage_space_bundle(config: HFSpaceDeployConfig, stage_dir: Path) -> Path | None:
    copy_repo_for_space(stage_dir)
    checkpoint_path = ensure_checkpoint(config, stage_dir)
    readme_path = stage_dir / "README.md"
    readme_path.write_text(render_space_readme(config))
    example_env_path = stage_dir / ".env.hf.space.example"
    if example_env_path.exists():
        example_env_path.unlink()
    return checkpoint_path


def default_checkpoint_runtime_path(checkpoint_name: str = DEFAULT_CHECKPOINT_NAME) -> Path:
    return default_checkpoint_path("artifacts/checkpoints", checkpoint_name)
