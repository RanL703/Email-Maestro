# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/executive_assistant/`. Keep environment logic in `env.py`, SQLite workspace behavior in `workspace.py`, reward logic in `graders.py`, typed contracts in `models.py`, seeded scenarios in `seeds.py`, and agent loops in `agent.py`. Tests live in `tests/` and should mirror the module they validate, for example `tests/test_env.py` for `env.py`. Use `training_env.ipynb` for experiments only; move stable logic back into `src/`. Top-level runtime files include `app.py`, `openenv.yaml`, `requirements.txt`, and `PRD.md`.

## Build, Test, and Development Commands
Set up a local environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite with `pytest -q`. Start the local Gradio entrypoint with `python app.py`. For a deterministic agent smoke run without training, use:

```bash
python - <<'PY'
from src.executive_assistant.agent import smoke_test_training_pipeline
print(smoke_test_training_pipeline())
PY
```

## Coding Style & Naming Conventions
Target Python 3.11+ and use 4-space indentation. Prefer explicit types and small, single-purpose functions. Follow existing naming patterns: `snake_case` for functions, variables, and modules; `PascalCase` for Pydantic models and environment classes; uppercase for constants such as `TASK_SEEDS`. Keep comments brief and only where behavior is not obvious. There is no formatter configured yet, so match the existing style and keep imports tidy.

## Testing Guidelines
Tests use `pytest`. Add or update tests with every behavioral change, especially for environment transitions, reward shaping, and seeded task completion. Name test files `test_*.py` and test functions `test_*`. Prefer deterministic assertions against observations, action logs, and scores over loose text checks.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects such as `Initial RL agent sandbox scaffold` and `Add PRD progress checkpoint note`. Continue that style: concise subject line, capitalized first word, no trailing period. Pull requests should include a brief summary, note any changed scenarios or rewards, list validation steps run (`pytest -q`, smoke tests), and attach screenshots only when UI behavior in `app.py` changes.

## Agent-Specific Notes
Preserve determinism. Do not introduce live external dependencies into the environment loop. If notebook experiments uncover a useful change, codify it in `src/` and cover it with tests before treating it as part of the baseline.
