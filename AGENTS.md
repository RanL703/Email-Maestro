# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/executive_assistant/`. Keep environment logic in `env.py`, SQLite workspace behavior in `workspace.py`, reward logic in `graders.py`, typed contracts in `models.py`, seeded scenarios in `seeds.py`, shared episode execution in `runner.py`, and policies in `agent.py`. Tests live in `tests/` and should mirror the module they validate, for example `tests/test_env.py` for `env.py`. Operational scripts live in `scripts/`. Use `training_env.ipynb` for experiments and rollout export only; move stable logic back into `src/`. Top-level runtime files include `app.py`, `openenv.yaml`, `requirements.txt`, and `PRD.md`.

## Build, Test, and Development Commands
Set up a local environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite with `pytest -q`. Start the local Gradio entrypoint with `python app.py`. Evaluate the deterministic baseline across all seeded tasks with `python scripts/evaluate_policies.py --provider baseline`. Run one full episode trace with `python scripts/run_policy_episode.py --task hard_rag_reply --provider baseline`. To exercise the OpenAI policy, set `OPENAI_API_KEY` first, then switch `--provider openai` or set `POLICY_PROVIDER = "openai"` in the notebook.

```bash
python scripts/evaluate_policies.py --provider baseline
```

## Coding Style & Naming Conventions
Target Python 3.11+ and use 4-space indentation. Prefer explicit types and small, single-purpose functions. Follow existing naming patterns: `snake_case` for functions, variables, and modules; `PascalCase` for Pydantic models and environment classes; uppercase for constants such as `TASK_SEEDS`. Keep comments brief and only where behavior is not obvious. There is no formatter configured yet, so match the existing style and keep imports tidy.

## Testing Guidelines
Tests use `pytest`. Add or update tests with every behavioral change, especially for environment transitions, reward shaping, seeded task completion, runner traces, and policy fallbacks. Name test files `test_*.py` and test functions `test_*`. Prefer deterministic assertions against observations, snapshots, action logs, and scores over loose text checks. If you change notebook-driven workflows, validate the underlying module or script rather than testing notebook JSON behavior only.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects such as `Initial RL agent sandbox scaffold` and `Add PRD progress checkpoint note`. Continue that style: concise subject line, capitalized first word, no trailing period. Pull requests should include a brief summary, note any changed scenarios or rewards, list validation steps run (`pytest -q`, smoke tests), and attach screenshots only when UI behavior in `app.py` changes.

## Agent-Specific Notes
Preserve determinism in the environment, graders, and baseline policy. Live API access belongs in policy layers such as `OpenAIResponsesPolicy`, not in the workspace or reward path. Keep `EpisodeRunner` as the shared execution path for scripts, tests, Gradio, and notebook workflows. If notebook experiments uncover a useful change, codify it in `src/` and cover it with tests before treating it as part of the baseline.

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
