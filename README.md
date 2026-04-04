# Autonomous Executive Assistant Sandbox

Deterministic RL-style workspace for an executive-assistant agent operating over a mock inbox, todo list, and local document store.

## Project status

This repository is scaffolded from the product requirements in [PRD.md](./PRD.md). The current setup establishes:

- a Python package layout for the environment, agent, graders, and seed data
- an OpenEnv-oriented contract using Pydantic models
- a lightweight Gradio entrypoint for future visualization work
- a notebook scaffold for iterative development without turning the notebook into the source of truth

## Repository layout

```text
.
├── app.py
├── openenv.yaml
├── requirements.txt
├── training_env.ipynb
├── src/
│   └── executive_assistant/
│       ├── agent.py
│       ├── env.py
│       ├── graders.py
│       ├── models.py
│       ├── seeds.py
│       └── workspace.py
└── tests/
    ├── test_env.py
    ├── test_models.py
    └── test_workspace.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python app.py
```

## Development workflow

1. Keep reusable logic in `src/executive_assistant/`.
2. Use `training_env.ipynb` for exploration, experiments, and prompt iteration only.
3. Promote notebook code into modules once it stabilizes.
4. Validate behavior through unit tests and deterministic scenario checks.
