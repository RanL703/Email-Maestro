from __future__ import annotations

import json

import gradio as gr

from src.executive_assistant.env import ExecutiveAssistantEnv


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


with gr.Blocks(title="Autonomous Executive Assistant Sandbox") as demo:
    gr.Markdown("# Autonomous Executive Assistant Sandbox")
    gr.Markdown(
        "Initial scaffold for a deterministic executive-assistant environment. "
        "The live agent loop and streaming reasoning UI can be layered on top of this baseline."
    )

    task = gr.Dropdown(
        choices=[
            "easy_deadline_extraction",
            "medium_triage_and_negotiation",
            "hard_rag_reply",
        ],
        value="easy_deadline_extraction",
        label="Scenario",
    )
    reset = gr.Button("Reset Scenario")
    observation = gr.Code(label="Observation", language="json")
    emails = gr.Dataframe(label="Emails")
    todos = gr.Dataframe(label="Todos")
    files = gr.Dataframe(label="Files")
    action_log = gr.Dataframe(label="Action Log")

    reset.click(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )

    demo.load(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )


if __name__ == "__main__":
    demo.launch()
