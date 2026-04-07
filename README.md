# Autonomous Executive Assistant Sandbox

Deterministic RL-style workspace for an executive-assistant agent operating over a mock inbox, todo list, and local document store.

## Project status

This repository is scaffolded from the product requirements in [PRD.md](./PRD.md). The current setup establishes:

- a Python package layout for the environment, agent, graders, and seed data
- an OpenEnv-oriented contract using Pydantic models
- an OpenRouter-backed Gemma policy layer for live model execution
- a tabular RL training pipeline with rollout export and checkpointing
- separate app and training environments, including a registered Jupyter kernel

## Repository layout

```text
.
├── app.py
├── openenv.yaml
├── requirements.txt
├── requirements.app.txt
├── requirements.training.txt
├── training_env.ipynb
├── src/
│   └── executive_assistant/
│       ├── agent.py
│       ├── config.py
│       ├── env.py
│       ├── graders.py
│       ├── llm_service.py
│       ├── models.py
│       ├── prompts.py
│       ├── runner.py
│       ├── seeds.py
│       ├── training.py
│       └── workspace.py
├── scripts/
│   ├── evaluate_policies.py
│   ├── run_policy_episode.py
│   ├── setup_app_env.sh
│   ├── setup_training_env.sh
│   └── train_rl_agent.py
└── tests/
    ├── test_agent.py
    ├── test_env.py
    ├── test_llm_service.py
    ├── test_models.py
    ├── test_runner.py
    ├── test_training.py
    └── test_workspace.py
```

## Environment setup

```bash
bash scripts/setup_app_env.sh
bash scripts/setup_training_env.sh
```

The training setup registers the Jupyter kernel `scalerhack2-training`.

## Validation and runners

Run the deterministic baseline across all seeded tasks:

```bash
.venv-training/bin/python scripts/evaluate_policies.py --provider baseline
```

Run a single episode and print the full trace:

```bash
.venv-training/bin/python scripts/run_policy_episode.py --task hard_rag_reply --provider baseline
```

Run the RL training smoke pipeline and save a checkpoint:

```bash
.venv-training/bin/python scripts/train_rl_agent.py --episodes 300
```

Start the deployed app runtime:

```bash
.venv-app/bin/python app.py
```

To use the OpenRouter-backed Gemma policy, set `OPENROUTER_API_KEY` first and then switch `--provider openrouter`.

## Development workflow

1. Keep reusable logic in `src/executive_assistant/`.
2. Use `training_env.ipynb` with the `scalerhack2-training` kernel for rollouts, prompt iteration, and RL experiments.
3. Promote notebook code into modules once it stabilizes.
4. Validate behavior through unit tests, deterministic scenario checks, RL checkpoint smoke runs, and exported episode traces.
