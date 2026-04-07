from __future__ import annotations

import json
import os
import time
import uuid

import gradio as gr

from src.executive_assistant.agent import BaselineAgent, OpenRouterPolicy
from src.executive_assistant.config import AppRuntimeConfig, OpenRouterConfig, load_env_file
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.runner import EpisodeRunner
from src.executive_assistant.training import QLearningPolicy, default_checkpoint_path

load_env_file(AppRuntimeConfig().env_file)
APP_RUNTIME = AppRuntimeConfig()
EMAIL_COLUMNS = ["id", "sender", "recipient", "subject", "body", "timestamp", "is_read", "is_archived"]
TODO_COLUMNS = ["id", "task_name", "deadline_date", "context"]
FILE_COLUMNS = ["id", "filename", "content_text"]
ACTION_LOG_COLUMNS = ["id", "action_type", "target_id", "payload", "secondary_payload", "status"]
TRACE_COLUMNS = ["step", "reasoning", "action_type", "status", "score", "done"]


def _records_to_rows(records: list[dict], columns: list[str]) -> list[list[object]]:
    return [[record.get(column) for column in columns] for record in records]


def build_snapshot(task_name: str) -> tuple[str, list[list[object]], list[list[object]], list[list[object]], list[list[object]]]:
    env = ExecutiveAssistantEnv(task_name=task_name)
    observation = env.reset()
    snapshot = env.workspace.snapshot()
    return (
        json.dumps(observation.model_dump(), indent=2),
        _records_to_rows(snapshot["emails"], EMAIL_COLUMNS),
        _records_to_rows(snapshot["todos"], TODO_COLUMNS),
        _records_to_rows(snapshot["files"], FILE_COLUMNS),
        _records_to_rows(snapshot["action_log"], ACTION_LOG_COLUMNS),
    )


def _default_rl_checkpoint() -> str:
    return str(
        default_checkpoint_path(
            APP_RUNTIME.checkpoint_dir,
            APP_RUNTIME.default_checkpoint_name,
        )
    )


def _build_policy(
    provider: str,
    model_name: str,
    api_key: str,
    checkpoint_path: str,
) -> object:
    if provider == "baseline":
        return BaselineAgent()
    if provider == "rl":
        return QLearningPolicy.load(checkpoint_path or _default_rl_checkpoint())
    env_api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    config = OpenRouterConfig(
        api_key=env_api_key,
        model_name=model_name,
        site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost:7860"),
        app_name=os.environ.get(
            "OPENROUTER_APP_NAME",
            "Autonomous Executive Assistant Sandbox",
        ),
    )
    return OpenRouterPolicy(config=config)


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


def _summary_payload(
    *,
    run_id: str,
    task_name: str,
    provider: str,
    policy_name: str,
    model_name: str,
    checkpoint_path: str,
    status: str,
    final_score: float,
    completed: bool,
    termination_reason: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "task_name": task_name,
        "requested_provider": provider,
        "policy_name": policy_name,
        "model_name": model_name if provider == "openrouter" else None,
        "checkpoint_path": checkpoint_path if provider == "rl" else None,
        "status": status,
        "final_score": final_score,
        "completed": completed,
        "termination_reason": termination_reason,
    }


def _step_payload(
    observation_payload: dict,
    snapshot_payload: dict,
    trace_rows: list[dict],
    summary_payload: dict,
) -> tuple[str, list[list[object]], list[list[object]], list[list[object]], list[list[object]], list[list[object]], str]:
    return (
        json.dumps(observation_payload, indent=2),
        _records_to_rows(snapshot_payload["emails"], EMAIL_COLUMNS),
        _records_to_rows(snapshot_payload["todos"], TODO_COLUMNS),
        _records_to_rows(snapshot_payload["files"], FILE_COLUMNS),
        _records_to_rows(snapshot_payload["action_log"], ACTION_LOG_COLUMNS),
        _records_to_rows(trace_rows, TRACE_COLUMNS),
        json.dumps(summary_payload, indent=2),
    )


def configure_provider_inputs(provider: str) -> tuple[dict, dict, dict]:
    is_openrouter = provider == "openrouter"
    is_rl = provider == "rl"
    return (
        gr.update(visible=is_openrouter, interactive=is_openrouter),
        gr.update(visible=is_openrouter, interactive=is_openrouter),
        gr.update(visible=is_rl, interactive=is_rl),
    )


def run_live_episode(
    task_name: str,
    provider: str,
    model_name: str,
    api_key: str,
    max_steps: int,
    checkpoint_path: str,
):
    run_id = uuid.uuid4().hex[:8]
    runner = EpisodeRunner(
        policy=_build_policy(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            checkpoint_path=checkpoint_path,
        ),
        max_steps=max_steps,
    )
    env, observation = runner.initialize(task_name)
    trace_rows: list[dict] = []

    initial_snapshot = env.workspace.snapshot()
    yield _step_payload(
        observation_payload=observation.model_dump(),
        snapshot_payload=initial_snapshot,
        trace_rows=trace_rows,
        summary_payload=_summary_payload(
            run_id=run_id,
            task_name=task_name,
            provider=provider,
            policy_name=type(runner.policy).__name__,
            model_name=model_name,
            checkpoint_path=checkpoint_path or _default_rl_checkpoint(),
            status="initialized",
            final_score=0.0,
            completed=False,
            termination_reason="episode not started",
        ),
    )

    while True:
        _, observation, reward, record = runner.advance(task_name, env, observation)
        trace_rows.append(
            {
                "step": record.step_index,
                "reasoning": record.reasoning,
                "action_type": record.action["action_type"],
                "status": record.status,
                "score": record.reward["total_score"],
                "done": record.reward["is_done"],
            }
        )
        yield _step_payload(
            observation_payload=record.observation,
            snapshot_payload=record.snapshot,
            trace_rows=trace_rows,
            summary_payload=_summary_payload(
                run_id=run_id,
                task_name=task_name,
                provider=provider,
                policy_name=type(runner.policy).__name__,
                model_name=model_name,
                checkpoint_path=checkpoint_path or _default_rl_checkpoint(),
                status="running" if not reward.is_done else "completed",
                final_score=reward.total_score,
                completed=reward.total_score >= 1.0,
                termination_reason=reward.reasoning,
            ),
        )
        if reward.is_done:
            return
        time.sleep(0.15)


with gr.Blocks(title="Autonomous Executive Assistant Sandbox") as demo:
    gr.Markdown("# Autonomous Executive Assistant Sandbox")
    gr.Markdown(
        "Deterministic executive-assistant sandbox with live episode execution for the baseline policy and OpenRouter Gemma."
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
            choices=["baseline", "openrouter", "rl"],
            value="baseline",
            label="Policy",
        )
        model_name = gr.Textbox(value="google/gemma-4-31b-it", label="OpenRouter Model")
        max_steps = gr.Number(value=12, precision=0, label="Max Steps")
    checkpoint_path = gr.Textbox(
        value=_default_rl_checkpoint(),
        label="RL Checkpoint Path",
    )
    api_key = gr.Textbox(type="password", label="OPENROUTER_API_KEY (optional for baseline)")

    with gr.Row():
        reset = gr.Button("Reset Scenario")
        run_episode_btn = gr.Button("Run Episode")

    observation = gr.Code(label="Observation", language="json")
    emails = gr.Dataframe(headers=EMAIL_COLUMNS, label="Unread Emails")
    todos = gr.Dataframe(headers=TODO_COLUMNS, label="Todos")
    files = gr.Dataframe(headers=FILE_COLUMNS, label="Search Results")
    action_log = gr.Dataframe(headers=ACTION_LOG_COLUMNS, label="Action Log")
    trace_table = gr.Dataframe(headers=TRACE_COLUMNS, label="Episode Trace")
    summary = gr.Code(label="Run Summary", language="json")

    reset.click(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )
    provider.change(
        fn=configure_provider_inputs,
        inputs=[provider],
        outputs=[model_name, api_key, checkpoint_path],
    )
    run_episode_btn.click(
        fn=run_live_episode,
        inputs=[task, provider, model_name, api_key, max_steps, checkpoint_path],
        outputs=[observation, emails, todos, files, action_log, trace_table, summary],
    )

    demo.load(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )
    demo.load(
        fn=configure_provider_inputs,
        inputs=[provider],
        outputs=[model_name, api_key, checkpoint_path],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=APP_RUNTIME.host,
        server_port=APP_RUNTIME.port,
        show_error=True,
    )
