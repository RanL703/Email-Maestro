# Autonomous Executive Assistant Sandbox

Deterministic RL-style workspace for an executive-assistant agent operating over a mock inbox, todo list, and local document store.

This project is being packaged for deployment to Hugging Face Spaces as a judge-facing demo for the **OpenEnv Scaler x Meta x PyTorch Hack**. The hack dashboard currently lists the main build round as **March 25, 2026 through April 8, 2026**, with finals on **April 25-26, 2026** in Bengaluru.

## Project status

This repository is scaffolded from the product requirements in [PRD.md](./PRD.md). The current setup establishes:

- a Python package layout for the environment, agent, graders, and seed data
- an OpenEnv-oriented contract using Pydantic models
- an OpenRouter-backed Gemma inference path through the OpenAI client for validator-facing model execution
- a tabular RL training pipeline with rollout export and checkpointing
- separate app and training environments, including a registered Jupyter kernel

## Environment Description

This environment models a real knowledge-work loop that humans perform every day:

- triaging email
- extracting structured tasks from unstructured communication
- escalating urgent requests
- negotiating meeting times
- searching local documents before replying

The environment is intentionally deterministic so an agent can be graded on workflow quality rather than on lucky wording. Instead of relying on Gmail, a live calendar, or a live file server, the system uses an isolated SQLite-backed workspace that simulates:

- an inbox with seeded emails
- a todo list with deadlines and context
- a file store for retrieval-style tasks
- an action log for deterministic grading and reward shaping

## OpenEnv Interface

The environment entrypoint is [src/executive_assistant/env.py](/home/ranl/code/scalerhack2/src/executive_assistant/env.py).

- `reset()` seeds a task-specific workspace and returns the initial typed observation.
- `step(action)` executes a typed action and returns `(observation, reward, done, info)`.
- `state()` returns the current environment state, including the observation snapshot and full workspace tables.
- [openenv.yaml](./openenv.yaml) binds the environment class and typed models together.

### Observation Space

`WorkspaceObservation` includes:

- `current_time`
- `unread_emails`
- `active_todos`
- `last_action_status`
- `current_email`
- `search_results`
- `action_history`

### Action Space

`AssistantAction.action_type` supports:

- `read_email`
- `reply`
- `forward`
- `add_todo`
- `archive`
- `search_files`

### Reward Space

`TaskReward` includes:

- `step_reward`
- `total_score`
- `is_done`
- `reasoning`

Rewards are dense and shaped over the full trajectory. Partial progress is rewarded, invalid or undesirable behavior lowers score indirectly through missed milestones and penalties, and episodes terminate at a fixed step budget.

## Tasks And Difficulty

### Easy: `easy_deadline_extraction`

Read a seeded academic email, extract three exact deadlines into todos, and archive the source message.

### Medium: `medium_triage_and_negotiation`

Archive newsletters, escalate a client complaint to the manager, and reply with a concrete time to a meeting reschedule request.

### Hard: `hard_rag_reply`

Read a stakeholder request, search the local report store, retrieve the relevant metrics, and reply with the correct grounded values.

All three tasks are deterministically graded in [src/executive_assistant/graders.py](/home/ranl/code/scalerhack2/src/executive_assistant/graders.py) with scores clamped to `0.0–1.0`.

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

Current deterministic baseline scores:

- `easy_deadline_extraction`: `1.0`
- `medium_triage_and_negotiation`: `1.0`
- `hard_rag_reply`: `1.0`

Run the required root-level inference script through the OpenRouter API using the OpenAI client compatibility layer. The canonical setup is:

```bash
OPENROUTER_API_KEY=... \
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=google/gemma-4-31b-it \
.venv-training/bin/python inference.py
```

If a validator requires the `OPENAI_API_KEY` variable name, set it to the same OpenRouter key:

```bash
OPENAI_API_KEY=... \
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=google/gemma-4-31b-it \
.venv-training/bin/python inference.py
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

The live Gradio app intentionally exposes only `baseline` and `rl`; the `rl` policy always loads the trained JSON checkpoint. OpenRouter-backed Gemma inference remains available through the root-level [inference.py](/home/ranl/code/scalerhack2/inference.py) script and CLI tooling for validation.

The repository intentionally uses the `openai` Python client with `base_url=https://openrouter.ai/api/v1` and `MODEL_NAME=google/gemma-4-31b-it`. It accepts the hackathon-compatible aliases `OPENAI_API_KEY`, `API_BASE_URL`, and `MODEL_NAME`, but the provider remains OpenRouter.

## OpenEnv Validation And Submission Checklist

Submission-sensitive files:

- environment metadata: [openenv.yaml](/home/ranl/code/scalerhack2/openenv.yaml)
- environment runtime: [src/executive_assistant/env.py](/home/ranl/code/scalerhack2/src/executive_assistant/env.py)
- typed models: [src/executive_assistant/models.py](/home/ranl/code/scalerhack2/src/executive_assistant/models.py)
- root inference script: [inference.py](/home/ranl/code/scalerhack2/inference.py)
- Docker build target: [Dockerfile](/home/ranl/code/scalerhack2/Dockerfile)

Recommended pre-submission checks:

```bash
.venv-training/bin/pytest -q
.venv-training/bin/python scripts/evaluate_policies.py --provider baseline
.venv-training/bin/python inference.py
docker build -t email-maestro .
docker run -p 7860:7860 email-maestro
```

If you have the OpenEnv validator installed locally, also run:

```bash
openenv validate
```

## Hugging Face Spaces deployment

The repository now includes a one-command Hugging Face Spaces deployment path that stages a Space-friendly bundle, injects a discrete HF `README.md`, carries over the RL checkpoint, creates or updates the Space, uploads the app, and sets runtime metadata variables.

1. Create the training environment if you have not already:

```bash
bash scripts/setup_training_env.sh
```

2. Prepare deployment variables:

```bash
cp .env.hf.space.example .env.hf.space
```

3. Fill in at least:

- `HF_TOKEN`
- `HF_SPACE_REPO`
- `HF_SPACE_TEAM_USERNAMES`

4. Deploy in one command:

```bash
bash scripts/deploy_hf_space.sh
```

What the deploy pipeline does:

- creates the target Space with `sdk=docker`
- stages a clean bundle without local `.env` files, virtualenvs, caches, or git metadata
- writes a discrete HF Space `README.md` addressed to **Team Epsilon**
- bundles `artifacts/checkpoints/q_policy_notebook.json` for the `rl` policy, or trains a fresh checkpoint if one is missing
- uploads the Space contents and sets `OPENROUTER_APP_NAME` and `OPENROUTER_SITE_URL`
- optionally sets `OPENROUTER_API_KEY` on the Space when it is present locally

Supporting deployment docs:

- HF README example: [docs/HF_SPACE_README.md](./docs/HF_SPACE_README.md)
- Deployment env template: [.env.hf.space.example](./.env.hf.space.example)
- Deployment script: [scripts/deploy_hf_space.py](./scripts/deploy_hf_space.py)

## Development workflow

1. Keep reusable logic in `src/executive_assistant/`.
2. Use `training_env.ipynb` with the `scalerhack2-training` kernel for rollouts, prompt iteration, and RL experiments.
3. Promote notebook code into modules once it stabilizes.
4. Validate behavior through unit tests, deterministic scenario checks, RL checkpoint smoke runs, and exported episode traces.
