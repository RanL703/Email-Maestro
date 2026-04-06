from __future__ import annotations

import json

import gradio as gr

from src.executive_assistant.agent import BaselineAgent, OpenAIResponsesPolicy
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.runner import EpisodeRunner


def build_snapshot(task_name: str) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    env = ExecutiveAssistantEnv(task_name=task_name)
    observation = env.reset()
    snapshot = env.workspace.snapshot()
    return (
        json.dumps(observation.model_dump(), indent=2),
        snapshot["emails"],
        snapshot["todos"],
        snapshot["files"],
        snapshot["action_log"],
    )


def _build_policy(provider: str, model_name: str, api_key: str) -> object:
    if provider == "baseline":
        return BaselineAgent()
    return OpenAIResponsesPolicy(api_key=api_key or None, model_name=model_name)


def _trace_to_rows(trace: object) -> list[dict]:
    return [
        {
            "step": step.step_index,
            "reasoning": step.reasoning,
            "action_type": step.action["action_type"],
            "status": step.status,
            "score": step.reward["total_score"],
            "done": step.reward["is_done"],
        }
        for step in trace.steps
    ]


def run_live_episode(
    task_name: str,
    provider: str,
    model_name: str,
    api_key: str,
    max_steps: int,
) -> tuple[str, list[dict], list[dict], list[dict], list[dict], list[dict], str]:
    runner = EpisodeRunner(
        policy=_build_policy(provider=provider, model_name=model_name, api_key=api_key),
        max_steps=max_steps,
    )
    trace = runner.run(task_name)
    final_step = trace.steps[-1] if trace.steps else None
    final_observation = final_step.observation if final_step else {}
    final_snapshot = final_step.snapshot if final_step else {
        "emails": [],
        "todos": [],
        "files": [],
        "action_log": [],
    }
    return (
        json.dumps(final_observation, indent=2),
        final_snapshot["emails"],
        final_snapshot["todos"],
        final_snapshot["files"],
        final_snapshot["action_log"],
        _trace_to_rows(trace),
        json.dumps(
            {
                "task_name": trace.task_name,
                "policy_name": trace.policy_name,
                "final_score": trace.final_score,
                "completed": trace.completed,
                "termination_reason": trace.termination_reason,
            },
            indent=2,
        ),
    )


with gr.Blocks(title="Autonomous Executive Assistant Sandbox") as demo:
    gr.Markdown("# Autonomous Executive Assistant Sandbox")
    gr.Markdown(
        "Deterministic executive-assistant sandbox with live episode execution for baseline and OpenAI policies."
    )

    with gr.Row():
        task = gr.Dropdown(
            choices=[
                "easy_deadline_extraction",
                "medium_triage_and_negotiation",
                "hard_rag_reply",
            ],
            value="easy_deadline_extraction",
            label="Scenario",
        )
        provider = gr.Dropdown(
            choices=["baseline", "openai"],
            value="baseline",
            label="Policy",
        )
        model_name = gr.Textbox(value="gpt-4.1-mini", label="OpenAI Model")
        max_steps = gr.Number(value=12, precision=0, label="Max Steps")
    api_key = gr.Textbox(type="password", label="OPENAI_API_KEY (optional for baseline)")

    with gr.Row():
        reset = gr.Button("Reset Scenario")
        run_episode_btn = gr.Button("Run Episode")

    observation = gr.Code(label="Observation", language="json")
    emails = gr.Dataframe(label="Unread Emails")
    todos = gr.Dataframe(label="Todos")
    files = gr.Dataframe(label="Search Results")
    action_log = gr.Dataframe(label="Action Log")
    trace_table = gr.Dataframe(label="Episode Trace")
    summary = gr.Code(label="Run Summary", language="json")

    reset.click(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )
    run_episode_btn.click(
        fn=run_live_episode,
        inputs=[task, provider, model_name, api_key, max_steps],
        outputs=[observation, emails, todos, files, action_log, trace_table, summary],
    )

    demo.load(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )


if __name__ == "__main__":
    demo.launch()
