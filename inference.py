from __future__ import annotations

import argparse
import json
import os

from src.executive_assistant.agent import OpenRouterPolicy
from src.executive_assistant.config import OpenRouterConfig
from src.executive_assistant.runner import run_policy_suite


TASKS = [
    "easy_deadline_extraction",
    "medium_triage_and_negotiation",
    "hard_rag_reply",
]


def build_openai_compatible_policy() -> OpenRouterPolicy:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip()
    base_url = (
        os.environ.get("OPENROUTER_BASE_URL", "").strip()
        or os.environ.get("API_BASE_URL", "").strip()
        or "https://openrouter.ai/api/v1"
    )
    model_name = (
        os.environ.get("OPENROUTER_MODEL", "").strip()
        or os.environ.get("MODEL_NAME", "").strip()
        or "google/gemma-4-31b-it"
    )
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY is required.")
    if not base_url:
        raise RuntimeError("API_BASE_URL or OPENROUTER_BASE_URL is required.")
    if not model_name:
        raise RuntimeError("MODEL_NAME or OPENROUTER_MODEL is required.")
    config = OpenRouterConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost:7860"),
        app_name=os.environ.get(
            "OPENROUTER_APP_NAME",
            "EmailMaestro | Executive Assistant Sandbox",
        ),
        temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "600")),
    )
    return OpenRouterPolicy(config=config)


def summarize_traces(traces) -> dict[str, dict[str, object]]:
    return {
        task_name: {
            "completed": trace.completed,
            "final_score": trace.final_score,
            "steps": len(trace.steps),
            "termination_reason": trace.termination_reason,
        }
        for task_name, trace in traces.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the required OpenAI-client inference baseline against all seeded tasks."
    )
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    print("START")
    traces = run_policy_suite(
        policy=build_openai_compatible_policy(),
        task_names=TASKS,
        max_steps=args.max_steps,
    )
    for task_name, trace in traces.items():
        print(
            "STEP "
            f"task={task_name} "
            f"score={trace.final_score:.4f} "
            f"completed={trace.completed}"
        )
    print(json.dumps(summarize_traces(traces), indent=2))
    print("END")


if __name__ == "__main__":
    main()
