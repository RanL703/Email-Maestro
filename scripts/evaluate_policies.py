from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_assistant.agent import BaselineAgent, OpenAIResponsesPolicy
from src.executive_assistant.runner import export_traces_jsonl, run_policy_suite


TASKS = [
    "easy_deadline_extraction",
    "medium_triage_and_negotiation",
    "hard_rag_reply",
]


def build_policy(provider: str, model_name: str) -> object:
    if provider == "baseline":
        return BaselineAgent()
    if provider == "openai":
        return OpenAIResponsesPolicy(model_name=model_name)
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy over all seeded tasks.")
    parser.add_argument("--provider", choices=["baseline", "openai"], default="baseline")
    parser.add_argument("--model", default="gpt-4.1-mini")
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
