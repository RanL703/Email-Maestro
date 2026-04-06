from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_assistant.agent import BaselineAgent, OpenAIResponsesPolicy
from src.executive_assistant.runner import EpisodeRunner


def build_policy(provider: str, model_name: str) -> object:
    if provider == "baseline":
        return BaselineAgent()
    if provider == "openai":
        return OpenAIResponsesPolicy(model_name=model_name)
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single policy episode.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--provider", choices=["baseline", "openai"], default="baseline")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    runner = EpisodeRunner(policy=build_policy(args.provider, args.model), max_steps=args.max_steps)
    trace = runner.run(args.task)
    print(json.dumps(trace.to_dict(), indent=2))


if __name__ == "__main__":
    main()
