from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.training import default_checkpoint_path, train_q_learning


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPACE_TITLE = "EmailMaestro | Executive Assistant Sandbox"
DEFAULT_HF_USERNAMES = [
    "flickinshots",
    "ShreyaKhatik",
    "itsayushdey",
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
    team_name: str = "Team Epsilon"
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
    roster_lines: list[str] = []
    if config.hf_usernames:
        roster_lines.append(f"- `@{config.hf_usernames[0]}` — Team lead and primary Space owner")
    for username in config.hf_usernames[1:]:
        roster_lines.append(f"- `@{username}` — Team member")
    roster = "\n".join(roster_lines) if roster_lines else "- Team roster to be added"
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
tags:
  - openenv
  - docker
  - gradio
short_description: OpenEnv executive assistant sandbox demo for judges.
---

# {config.team_name}

Welcome judges of the **OpenEnv Scaler x Meta x PyTorch Hack**. This Space hosts **EmailMaestro**, our deterministic executive assistant environment and policy demo built by **{config.team_name}** for repeatable agent evaluation, visible tool use, and side-by-side policy comparison.

## Team Epsilon Roster

- Team name: `{config.team_name}`
- Hugging Face Space: `{config.repo_id}`
- Live app: `{config.app_url}`
- Public repository view on Hugging Face: `{config.space_url}`

{roster}

## Executive Summary

EmailMaestro is an **Autonomous Executive Assistant Sandbox** designed around the OpenEnv pattern: typed observations, typed actions, deterministic rewards, and a visible environment loop. Instead of depending on a brittle live email provider, the agent operates inside an isolated SQLite-backed workspace that simulates an inbox, a todo manager, and a local document store. That lets judges inspect policy quality through reproducible runs rather than through one-off anecdotal chats.

This Space is intended to show three things clearly:

- The agent can operate as a structured tool user, not just a text generator.
- The same environment loop is shared across notebook experiments, CLI evaluation, tests, and the live Gradio UI.
- Baseline, learned, and model-backed policies can be compared under the same task and reward conditions.

## Why This Fits The Hackathon

OpenEnv was introduced by Hugging Face and Meta as an open framework for building agent environments with typed observations, actions, and rewards. The Scaler hack timeline for this event lists the main build window as **March 25, 2026 through April 8, 2026**, with finals on **April 25-26, 2026** in Bengaluru. Our submission is shaped directly around that evaluation style:

- Deterministic environment setup
- Typed environment contracts
- Observable step-by-step policy execution
- Reproducible seeded tasks
- Judge-friendly visualization of state transitions

## Problem Framing

Most assistant demos stop at text quality. We wanted to show that an agent can manage a workflow end to end:

- read a chaotic inbox
- extract structured work into a task list
- triage low-priority versus high-priority communication
- search a local knowledge source before replying
- produce actions that can be graded deterministically

To make that possible under hackathon constraints, we replaced live services with a controlled mock workspace that still feels operationally realistic.

## Core Architecture

- **Environment state:** in-memory SQLite workspace simulating emails, todos, files, and action history
- **OpenEnv contract:** Pydantic models defining observations, actions, rewards, and policy decisions
- **Execution loop:** shared `EpisodeRunner` used by tests, scripts, notebook experiments, and the Gradio app
- **Policies:** deterministic baseline, tabular RL checkpoint replay, and optional OpenRouter-backed live policy execution
- **UI layer:** Gradio control room plus visible workspace snapshots for judges

## Seeded Judge Tasks

### 1. Easy: Deadline Extraction

The environment injects an academic email containing multiple deadlines. The policy must read the message, create the correct todo entries, and archive the source email.

### 2. Medium: Inbox Triage And Negotiation

The environment mixes newsletters, an urgent complaint, and a meeting reschedule request. The policy must archive low-value mail, escalate the complaint properly, and send a concrete meeting reply.

### 3. Hard: RAG Reply

The environment includes a stakeholder email asking for exact metrics from a local report. The policy must search the file store, recover the relevant values, and draft a grounded reply using the retrieved evidence.

## What Judges Can Inspect In This Space

- Live observation payloads
- Workspace tables for emails, todos, files, and action logs
- Step-by-step trace rows with reasoning, action type, status, score, and done state
- Differences between `baseline`, bundled `rl`, and optional `openrouter` policies

## Runtime And Deployment Notes

- SDK: `docker`
- App port: `{config.app_port}`
- Entry point: `python app.py`
- Optional secret: `OPENROUTER_API_KEY`
- {checkpoint_note}
- The bundled RL artifact lives at `artifacts/checkpoints/{config.checkpoint_name}`
- The Space is deployed from the same repository used for local tests and notebook-backed experiments

## Recommended Judge Flow

1. Open the Space and choose one of the seeded scenarios.
2. Run the deterministic `baseline` policy for a guaranteed reference trace.
3. Switch to `rl` to replay the bundled learned checkpoint.
4. Add `OPENROUTER_API_KEY` in Space secrets to enable the live model-backed path.
5. Compare how the workspace mutates after each step instead of evaluating only the final response.

## Implementation Notes

- The app, scripts, notebook, and tests all rely on the same `EpisodeRunner` workflow loop.
- Live API access stays in the policy layer, so deterministic evaluation remains possible without network access.
- The current RL path is intentionally lightweight and reproducible: a tabular Q-learning prototype trained over seeded action templates.
- The Gradio interface is designed for demonstration and debugging, not just for final-state screenshots.

## What We Want Judges To Notice

- Strong separation between environment state, policy choice, and reward logic
- Clear evidence of agent tool use
- Reproducibility across runs
- A hackathon-friendly deployment that still preserves engineering discipline

## References And Context

- Hack dashboard: https://www.scaler.com/openenv-hackathon
- OpenEnv launch: https://huggingface.co/blog/openenv
- Space page: {config.space_url}
- Live app: {config.app_url}
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
