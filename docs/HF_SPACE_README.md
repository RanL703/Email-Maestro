---
title: Project Epsilon | Executive Assistant Sandbox
emoji: "🧭"
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv executive assistant sandbox demo for judges.
---

# Project Epsilon

Discrete Hugging Face Space README for the **Autonomous Executive Assistant Sandbox**, prepared for the **OpenEnv Scaler x Meta x PyTorch Hack**.

## Team

- Team name: `Project Epsilon`
- Hugging Face usernames: `@HF_USERNAME_1`, `@HF_USERNAME_2`, `@HF_USERNAME_3`
- Space repo: `HF_USERNAME_PLACEHOLDER/project-epsilon-executive-assistant`

Replace the placeholder usernames and repo owner when the final team accounts are ready.

## What This Space Shows

- Deterministic OpenEnv-style tasks over a SQLite-backed executive assistant workspace
- A Gradio judge console that replays the shared `EpisodeRunner` loop step by step
- Policy switching across `baseline`, bundled `rl`, and optional `openrouter`
- Visible inbox, todo, file-search, and action-log state transitions

## Hack Context

OpenEnv was introduced by Hugging Face and Meta as an open source framework for typed agent environments. The Scaler hack dashboard lists the build window as **March 25, 2026 through April 8, 2026**, with finals on **April 25-26, 2026** in Bengaluru. This Space is tuned for that style of evaluation: deterministic tasks, structured actions, reproducible runs, and a judge-friendly visual trace.

## Runtime Notes

- SDK: `docker`
- App port: `7860`
- Entry point: `python app.py`
- Optional secret: `OPENROUTER_API_KEY`
- Bundled RL checkpoint path: `artifacts/checkpoints/q_policy_notebook.json`

## Judge Flow

1. Open the Space and choose one of the seeded scenarios.
2. Run `baseline` first for the reference trace.
3. Switch to `rl` to replay the trained checkpoint bundled with the Space.
4. Add `OPENROUTER_API_KEY` in Space secrets to enable the live model-backed policy.

## References

- Hack dashboard: https://www.scaler.com/openenv-hackathon
- OpenEnv launch: https://huggingface.co/blog/openenv
