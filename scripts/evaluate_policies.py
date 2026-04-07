from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_assistant.agent import BaselineAgent, OpenRouterPolicy
from src.executive_assistant.config import OpenRouterConfig, TrainingRuntimeConfig, load_env_file
from src.executive_assistant.runner import export_traces_jsonl, run_policy_suite


TASKS = [
    "easy_deadline_extraction",
    "medium_triage_and_negotiation",
    "hard_rag_reply",
]


def build_policy(provider: str, model_name: str) -> object:
    if provider == "baseline":
        return BaselineAgent()
    if provider == "openrouter":
        load_env_file(TrainingRuntimeConfig().env_file)
        config = OpenRouterConfig.from_env()
        config = OpenRouterConfig(
            api_key=config.api_key,
            model_name=model_name,
            base_url=config.base_url,
            site_url=config.site_url,
            app_name=config.app_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return OpenRouterPolicy(config=config)
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    load_env_file(TrainingRuntimeConfig().env_file)
    parser = argparse.ArgumentParser(description="Evaluate a policy over all seeded tasks.")
    parser.add_argument("--provider", choices=["baseline", "openrouter"], default="baseline")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    traces = run_policy_suite(
        policy=build_policy(args.provider, args.model),
        task_names=TASKS,
        max_steps=args.max_steps,
    )
    summary = {
        task_name: {
            "completed": trace.completed,
            "final_score": trace.final_score,
            "steps": len(trace.steps),
            "termination_reason": trace.termination_reason,
        }
        for task_name, trace in traces.items()
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        export_traces_jsonl(list(traces.values()), args.output)
        print(f"Saved traces to {args.output}")


if __name__ == "__main__":
    main()
