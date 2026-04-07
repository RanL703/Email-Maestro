# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/executive_assistant/`. Keep environment logic in `env.py`, SQLite workspace behavior in `workspace.py`, reward logic in `graders.py`, typed contracts in `models.py`, provider configuration in `config.py`, prompt construction in `prompts.py`, OpenRouter calls in `llm_service.py`, shared episode execution in `runner.py`, policies in `agent.py`, and RL logic in `training.py`. Tests live in `tests/` and should mirror the module they validate. Operational scripts live in `scripts/`. Use `training_env.ipynb` with the `scalerhack2-training` kernel for experiments and rollout export only; move stable logic back into `src/`. Top-level runtime files include `app.py`, `openenv.yaml`, `requirements*.txt`, and `PRD.md`.

## Build, Test, and Development Commands
Set up the separate app and training environments with:

```bash
bash scripts/setup_app_env.sh
bash scripts/setup_training_env.sh
```

Run the test suite with `.venv-training/bin/pytest -q`. Start the local Gradio entrypoint with `.venv-app/bin/python app.py`. Evaluate the deterministic baseline across all seeded tasks with `.venv-training/bin/python scripts/evaluate_policies.py --provider baseline`. Run one full episode trace with `.venv-training/bin/python scripts/run_policy_episode.py --task hard_rag_reply --provider baseline`. Train the tabular RL policy with `.venv-training/bin/python scripts/train_rl_agent.py --episodes 300`. To exercise the Gemma model through OpenRouter, set `OPENROUTER_API_KEY` first, then switch `--provider openrouter` or set `POLICY_PROVIDER = "openrouter"` in the notebook.

```bash
.venv-training/bin/python scripts/evaluate_policies.py --provider baseline
```

## Coding Style & Naming Conventions
Target Python 3.11+ and use 4-space indentation. Prefer explicit types and small, single-purpose functions. Follow existing naming patterns: `snake_case` for functions, variables, and modules; `PascalCase` for Pydantic models and environment classes; uppercase for constants such as `TASK_SEEDS`. Keep comments brief and only where behavior is not obvious. There is no formatter configured yet, so match the existing style and keep imports tidy.

## Testing Guidelines
Tests use `pytest`. Add or update tests with every behavioral change, especially for environment transitions, reward shaping, seeded task completion, runner traces, OpenRouter service behavior, and RL training smoke paths. Name test files `test_*.py` and test functions `test_*`. Prefer deterministic assertions against observations, snapshots, action logs, checkpoints, and scores over loose text checks. If you change notebook-driven workflows, validate the underlying module or script rather than testing notebook JSON behavior only.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects such as `Initial RL agent sandbox scaffold` and `Add PRD progress checkpoint note`. Continue that style: concise subject line, capitalized first word, no trailing period. Pull requests should include a brief summary, note any changed scenarios or rewards, list validation steps run (`pytest -q`, smoke tests), and attach screenshots only when UI behavior in `app.py` changes.

## Agent-Specific Notes
Preserve determinism in the environment, graders, and baseline policy. Live API access belongs in policy layers such as `OpenRouterPolicy`, not in the workspace or reward path. Keep `EpisodeRunner` as the shared execution path for scripts, tests, Gradio, and notebook workflows. Treat OpenRouter calls as optional runtime behavior: tests and RL smoke runs must stay runnable without network access. If notebook experiments uncover a useful change, codify it in `src/` and cover it with tests before treating it as part of the baseline.

## Agent Workflow Loop
All execution surfaces in this repository should follow the same loop:

1. Load environment state
2. Generate observation
3. Send to LLM or policy
4. Receive structured action
5. Execute action in workspace
6. Update state
7. Repeat until task complete

In code, keep this flow inside `EpisodeRunner`. Use `initialize()` for steps 1-2, `choose_action()` for steps 3-4, and `advance()` plus `env.step()` for steps 5-6. Do not duplicate bespoke episode loops in notebooks, scripts, or UI handlers.
