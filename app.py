from __future__ import annotations

import json
import os
import time
import uuid
from html import escape

import gradio as gr

from src.executive_assistant.agent import BaselineAgent, OpenRouterPolicy
from src.executive_assistant.config import AppRuntimeConfig, OpenRouterConfig, load_env_file
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import PolicyDecision, WorkspaceObservation
from src.executive_assistant.runner import EpisodeRunner
from src.executive_assistant.training import QLearningPolicy, default_checkpoint_path, train_q_learning

load_env_file(AppRuntimeConfig().env_file)
APP_RUNTIME = AppRuntimeConfig()
EMAIL_COLUMNS = ["id", "sender", "recipient", "subject", "body", "timestamp", "is_read", "is_archived"]
TODO_COLUMNS = ["id", "task_name", "deadline_date", "context"]
FILE_COLUMNS = ["id", "filename", "content_text"]
ACTION_LOG_COLUMNS = ["id", "action_type", "target_id", "payload", "secondary_payload", "status"]
TRACE_COLUMNS = ["step", "reasoning", "action_type", "status", "score", "done"]
APP_CSS = """
:root {
  color-scheme: dark;
  --ea-bg: #120f0c;
  --ea-bg-soft: #1a1511;
  --ea-panel: rgba(28, 22, 18, 0.88);
  --ea-panel-strong: #241c17;
  --ea-ink: #f5ede2;
  --ea-muted: #b7a796;
  --ea-border: rgba(236, 214, 188, 0.12);
  --ea-border-strong: rgba(236, 214, 188, 0.24);
  --ea-accent: #c97943;
  --ea-accent-deep: #e1a16f;
  --ea-highlight: #3a2a1f;
  --ea-success: #72c79a;
  --ea-danger: #ef8d76;
  --ea-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
}

.gradio-container {
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(124, 73, 39, 0.22), transparent 24%),
    radial-gradient(circle at 85% 10%, rgba(201, 121, 67, 0.16), transparent 22%),
    linear-gradient(180deg, #17120f 0%, #0f0c0a 100%);
  color: var(--ea-ink);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container .prose,
.gradio-container .gr-markdown,
.gradio-container .gr-button,
.gradio-container .gr-input,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel {
  color: var(--ea-ink);
}

.app-shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 18px 18px 28px;
}

.hero {
  background:
    linear-gradient(140deg, rgba(33, 25, 20, 0.96), rgba(21, 17, 14, 0.96)),
    linear-gradient(90deg, rgba(201, 121, 67, 0.12), transparent);
  border: 1px solid var(--ea-border);
  border-radius: 32px;
  padding: 34px;
  box-shadow: var(--ea-shadow);
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}

.hero::after {
  content: "";
  position: absolute;
  inset: auto -10% -44% 34%;
  height: 220px;
  background: radial-gradient(circle, rgba(201, 121, 67, 0.18), transparent 62%);
  pointer-events: none;
}

.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.95fr);
  gap: 22px;
  align-items: end;
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(201, 121, 67, 0.10);
  border: 1px solid rgba(201, 121, 67, 0.18);
  color: var(--ea-accent-deep);
  font-size: 0.76rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 16px;
}

.hero-copy {
  position: relative;
  z-index: 1;
}

.hero h1 {
  margin: 0 0 12px;
  font-family: "Baskerville", "Times New Roman", serif;
  font-size: clamp(2.6rem, 5vw, 4.5rem);
  line-height: 1.05;
  letter-spacing: -0.05em;
  max-width: 10ch;
}

.hero p {
  margin: 0;
  max-width: 760px;
  color: var(--ea-muted);
  font-size: 1.02rem;
  line-height: 1.65;
}

.hero-strip {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 22px;
}

.hero-pill {
  background: rgba(255, 255, 255, 0.05);
  color: var(--ea-ink);
  border: 1px solid rgba(236, 214, 188, 0.08);
  border-radius: 999px;
  padding: 10px 14px;
  font-size: 0.84rem;
  backdrop-filter: blur(12px);
}

.hero-aside {
  position: relative;
  z-index: 1;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(236, 214, 188, 0.08);
  border-radius: 24px;
  padding: 20px;
  backdrop-filter: blur(12px);
}

.hero-aside-label {
  margin: 0 0 10px;
  color: var(--ea-accent-deep);
  font-size: 0.8rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.hero-aside-value {
  margin: 0 0 14px;
  font-family: "Baskerville", "Times New Roman", serif;
  font-size: 1.6rem;
  line-height: 1.05;
}

.hero-aside-copy {
  margin: 0;
  color: var(--ea-muted);
  line-height: 1.6;
}

.panel-card,
.status-card {
  background: var(--ea-panel);
  border: 1px solid var(--ea-border);
  border-radius: 24px;
  box-shadow: var(--ea-shadow);
  backdrop-filter: blur(10px);
}

.panel-card {
  padding: 18px;
  overflow: visible;
}

.status-card {
  padding: 22px 22px 18px;
  overflow: hidden;
}

.panel-title {
  margin: 0 0 6px;
  font-family: "Baskerville", "Times New Roman", serif;
  font-size: 1.5rem;
  letter-spacing: -0.03em;
}

.panel-copy {
  margin: 0 0 16px;
  color: var(--ea-muted);
  line-height: 1.55;
}

.control-room {
  background:
    linear-gradient(180deg, rgba(45, 40, 49, 0.82), rgba(34, 28, 24, 0.94) 18%, rgba(28, 23, 19, 0.96) 100%);
  border: 1px solid rgba(236, 214, 188, 0.10);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.04),
    0 24px 70px rgba(0, 0, 0, 0.28);
}

.control-room .panel-title {
  margin-bottom: 8px;
}

.control-room .panel-copy {
  max-width: 56ch;
  color: #cdbca9;
}

.surface-card {
  background: rgba(23, 18, 14, 0.84);
  border: 1px solid var(--ea-border);
  border-radius: 24px;
  box-shadow: var(--ea-shadow);
  overflow: hidden;
}

.surface-card .gr-tab-nav {
  background: rgba(255, 255, 255, 0.03);
  padding: 10px 10px 0;
  border-bottom: 1px solid var(--ea-border);
}

.surface-card .gr-tab-nav button {
  border-radius: 16px 16px 0 0;
  border: 1px solid transparent;
  color: var(--ea-muted);
  font-weight: 600;
}

.surface-card .gr-tab-nav button.selected {
  background: var(--ea-panel-strong);
  color: var(--ea-ink);
  border-color: var(--ea-border);
}

.surface-card .gr-tabitem {
  padding: 18px;
}

.status-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 12px;
}

.status-title {
  font-family: "Baskerville", "Times New Roman", serif;
  font-size: 1.7rem;
  letter-spacing: -0.04em;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 8px 13px;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  border: 1px solid transparent;
  background: rgba(201, 121, 67, 0.10);
}

.status-badge.running,
.status-badge.initialized {
  border-color: rgba(180, 95, 45, 0.18);
  color: var(--ea-accent-deep);
}

.status-badge.completed.success {
  background: rgba(45, 122, 88, 0.10);
  border-color: rgba(45, 122, 88, 0.18);
  color: var(--ea-success);
}

.status-badge.completed.failure {
  background: rgba(178, 76, 56, 0.10);
  border-color: rgba(178, 76, 56, 0.16);
  color: var(--ea-danger);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.metric {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(236, 214, 188, 0.08);
  border-radius: 18px;
  padding: 14px;
  min-width: 0;
}

.metric-label {
  color: var(--ea-muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.11em;
  margin-bottom: 7px;
}

.metric-value {
  font-size: 1rem;
  line-height: 1.25;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.control-room .control-field {
  margin-bottom: 12px;
}

.control-room .control-field:last-of-type {
  margin-bottom: 0;
}

.control-room input,
.control-room textarea,
.control-room select,
.control-room button {
  min-height: 48px;
}

.control-room .gr-block-label,
.control-room label,
.control-room .gr-form > label {
  color: #efe2d1;
  font-size: 0.78rem;
  letter-spacing: 0.12em;
}

.control-room [role="radiogroup"] {
  display: grid !important;
  gap: 10px !important;
  margin-top: 10px;
}

.control-room .control-field:first-of-type [role="radiogroup"] {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.control-room .control-field:nth-of-type(2) [role="radiogroup"] {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.control-room [role="radio"] {
  position: relative;
  display: flex !important;
  align-items: center;
  justify-content: flex-start;
  gap: 12px;
  min-height: 72px;
  border: 1px solid rgba(236, 214, 188, 0.14) !important;
  border-radius: 18px !important;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02)) !important;
  padding: 16px 18px !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.03),
    0 10px 24px rgba(0, 0, 0, 0.16);
  transition:
    transform 160ms ease,
    border-color 160ms ease,
    background 160ms ease,
    box-shadow 160ms ease;
}

.control-room [role="radio"]:hover {
  transform: translateY(-1px);
  border-color: rgba(225, 161, 111, 0.34) !important;
  background:
    linear-gradient(180deg, rgba(225, 161, 111, 0.08), rgba(255, 255, 255, 0.03)) !important;
}

.control-room [role="radio"][aria-checked="true"] {
  border-color: rgba(201, 121, 67, 0.58) !important;
  background:
    linear-gradient(180deg, rgba(201, 121, 67, 0.18), rgba(201, 121, 67, 0.08)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 16px 34px rgba(108, 54, 24, 0.22);
}

.control-room [role="radio"]::before {
  content: "";
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  border-radius: 999px;
  border: 1px solid rgba(236, 214, 188, 0.26);
  background: rgba(255, 255, 255, 0.02);
  box-shadow: inset 0 0 0 3px rgba(25, 21, 18, 0.96);
}

.control-room [role="radio"][aria-checked="true"]::before {
  background: var(--ea-accent-deep);
  border-color: rgba(225, 161, 111, 0.82);
  box-shadow:
    inset 0 0 0 3px rgba(36, 28, 23, 0.92),
    0 0 0 4px rgba(201, 121, 67, 0.14);
}

.control-room [role="radio"] span {
  color: #f3e7d9 !important;
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  line-height: 1.35;
  white-space: normal !important;
  word-break: break-word;
}

.control-room [role="radio"] svg,
.control-room [role="radio"] input {
  display: none !important;
}

.control-room [data-testid="number"] {
  margin-top: 2px;
}

.control-room [data-testid="number"] input {
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.06) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.control-room .gr-accordion {
  margin-top: 4px;
  border: 1px solid rgba(236, 214, 188, 0.12) !important;
  border-radius: 18px !important;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.03) !important;
}

.control-room .gr-accordion summary,
.control-room .gr-accordion button {
  font-weight: 600;
}

.control-room .gr-accordion .label-wrap,
.control-room .gr-accordion label {
  color: #efe2d1 !important;
}

.control-room .gr-accordion .hide-container {
  background: rgba(18, 15, 12, 0.18);
}

.control-room .gr-button {
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  font-size: 1.02rem;
}

.control-room .gr-button.secondary {
  background: linear-gradient(180deg, rgba(90, 88, 94, 0.9), rgba(72, 70, 76, 0.95));
}

.control-room .gr-button.primary {
  background: linear-gradient(135deg, #cf8450 0%, #df9a67 100%);
}

.control-room .footnote {
  margin-top: 18px;
  color: #c0b09f;
}

.status-reason {
  background: rgba(201, 121, 67, 0.08);
  border: 1px solid rgba(236, 214, 188, 0.08);
  border-radius: 18px;
  padding: 14px 15px;
  color: var(--ea-muted);
  line-height: 1.55;
}

.scenario-brief {
  background: linear-gradient(180deg, rgba(32, 25, 20, 0.92), rgba(22, 18, 14, 0.94));
  border: 1px solid var(--ea-border);
  border-radius: 24px;
  padding: 22px;
  color: var(--ea-ink);
  box-shadow: var(--ea-shadow);
}

.scenario-brief h3 {
  margin: 0 0 10px;
  font-family: "Baskerville", "Times New Roman", serif;
  font-size: 1.5rem;
  letter-spacing: -0.03em;
}

.scenario-brief p {
  margin: 0 0 14px;
  color: var(--ea-muted);
  line-height: 1.6;
}

.scenario-brief ul {
  margin: 0;
  padding-left: 18px;
  color: var(--ea-ink);
}

.scenario-brief li {
  margin-bottom: 8px;
  line-height: 1.5;
}

.panel-card .gr-form,
.panel-card .gr-box,
.panel-card .gr-group {
  border: 0;
  background: transparent;
  box-shadow: none;
}

.panel-card .gr-button,
.gradio-container .gr-button {
  min-height: 48px;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.gradio-container button.primary {
  background: linear-gradient(135deg, var(--ea-accent) 0%, var(--ea-accent-deep) 100%);
  border: 0;
  box-shadow: 0 14px 30px rgba(138, 62, 23, 0.18);
}

.gradio-container button.secondary {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--ea-border-strong);
  color: var(--ea-ink);
}

.gradio-container label,
.gradio-container .gr-block-label,
.gradio-container .gr-form > label {
  color: var(--ea-muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(236, 214, 188, 0.12) !important;
  border-radius: 16px !important;
  color: var(--ea-ink) !important;
}

.gradio-container .gr-accordion,
.gradio-container .gr-panel,
.gradio-container .gr-box,
.gradio-container .block {
  border-color: var(--ea-border) !important;
}

.workspace-grid .gr-dataframe,
.workspace-grid .gr-code,
.workspace-grid .gr-box,
.workspace-grid .gr-panel {
  border-radius: 20px !important;
  overflow: hidden;
}

.workspace-grid .gr-code,
.workspace-grid .gr-dataframe {
  box-shadow: inset 0 0 0 1px rgba(58, 43, 28, 0.06);
}

.workspace-grid table {
  font-size: 0.92rem;
}

.footnote {
  margin-top: 14px;
  color: var(--ea-muted);
  font-size: 0.85rem;
  line-height: 1.6;
}

@media (max-width: 1120px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 980px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .control-room .control-field:first-of-type [role="radiogroup"],
  .control-room .control-field:nth-of-type(2) [role="radiogroup"] {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .hero {
    padding: 24px 18px;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }

  .app-shell {
    padding: 12px 12px 20px;
  }
}
"""
SCENARIO_GUIDANCE = {
    "easy_deadline_extraction": {
        "title": "Deadline Extraction",
        "description": "Read the professor email, capture the three exact milestones as todos, then archive the source email once the list is complete.",
        "checks": [
            "Read the source email before creating todos.",
            "Create exactly three canonical todos with ISO dates.",
            "Archive the email only after all deadlines are captured.",
        ],
    },
    "medium_triage_and_negotiation": {
        "title": "Inbox Triage And Negotiation",
        "description": "Clear low-value newsletters, escalate the client complaint to the manager, and send a concrete meeting time to the teammate without archiving unresolved important mail too early.",
        "checks": [
            "Archive all three newsletters.",
            "Forward the client complaint to manager@company.com.",
            "Reply to the teammate with a specific meeting time.",
        ],
    },
    "hard_rag_reply": {
        "title": "RAG Reply",
        "description": "Read the stakeholder request, search the local report store, and reply with the exact Q3 metrics from the matching file.",
        "checks": [
            "Read the VIP email first.",
            "Search for the Q3 architecture report before replying.",
            "Reply with 99.95%, 182ms, and 14% plus a greeting and signoff.",
        ],
    },
}


def _records_to_rows(records: list[dict], columns: list[str]) -> list[list[object]]:
    return [[record.get(column) for column in columns] for record in records]


def render_scenario_brief(task_name: str) -> str:
    guidance = SCENARIO_GUIDANCE[task_name]
    checks = "".join(f"<li>{escape(item)}</li>" for item in guidance["checks"])
    return (
        '<div class="scenario-brief">'
        f"<h3>{escape(guidance['title'])}</h3>"
        f"<p>{escape(guidance['description'])}</p>"
        f"<ul>{checks}</ul>"
        "</div>"
    )


def render_status_card(summary_payload: dict) -> str:
    status = str(summary_payload["status"])
    completed = bool(summary_payload["completed"])
    badge_class = f"status-badge {status} {'success' if completed else 'failure'}".strip()
    return (
        '<div class="status-card">'
        '<div class="status-topline">'
        f'<div class="status-title">Run {escape(str(summary_payload["run_id"]))}</div>'
        f'<div class="{badge_class}">{escape(status)}</div>'
        "</div>"
        '<div class="metric-grid">'
        f'<div class="metric"><div class="metric-label">Requested Provider</div><div class="metric-value">{escape(str(summary_payload["requested_provider"]))}</div></div>'
        f'<div class="metric"><div class="metric-label">Effective Policy</div><div class="metric-value">{escape(str(summary_payload["policy_name"]))}</div></div>'
        f'<div class="metric"><div class="metric-label">Scenario</div><div class="metric-value">{escape(str(summary_payload["task_name"]))}</div></div>'
        f'<div class="metric"><div class="metric-label">Final Score</div><div class="metric-value">{summary_payload["final_score"]:.2f}</div></div>'
        "</div>"
        '<div class="metric-grid">'
        f'<div class="metric"><div class="metric-label">Checkpoint</div><div class="metric-value">{escape(str(summary_payload["checkpoint_path"] or "n/a"))}</div></div>'
        f'<div class="metric"><div class="metric-label">Completed</div><div class="metric-value">{escape(str(completed))}</div></div>'
        f'<div class="metric"><div class="metric-label">Status</div><div class="metric-value">{escape(status)}</div></div>'
        "</div>"
        f'<div class="status-reason">{escape(str(summary_payload["termination_reason"]))}</div>'
        "</div>"
    )


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


def _ensure_rl_checkpoint(checkpoint_path: str) -> str:
    path = default_checkpoint_path(
        APP_RUNTIME.checkpoint_dir,
        APP_RUNTIME.default_checkpoint_name,
    )
    if checkpoint_path:
        path = default_checkpoint_path("", checkpoint_path)
    if path.exists():
        return str(path)

    policy, _ = train_q_learning(
        episodes=300,
        epsilon=0.15,
        teacher=BaselineAgent(),
    )
    saved_path = policy.save(path)
    return str(saved_path)


class OpenRouterGuidedCheckpointPolicy:
    def __init__(
        self,
        checkpoint_policy: QLearningPolicy,
        model_policy: OpenRouterPolicy | None,
    ) -> None:
        self.checkpoint_policy = checkpoint_policy
        self.model_policy = model_policy

    def choose_action(self, task_name: str, observation: WorkspaceObservation) -> PolicyDecision:
        checkpoint_decision = self.checkpoint_policy.choose_action(task_name, observation)
        if self.model_policy is None:
            return PolicyDecision(
                reasoning=(
                    "OpenRouter model is not configured; using the trained RL checkpoint action. "
                    f"{checkpoint_decision.reasoning}"
                ),
                action=checkpoint_decision.action,
            )
        guided_observation = observation.model_copy(
            update={
                "action_history": observation.action_history
                + [
                    (
                        "Trained RL checkpoint recommendation: "
                        f"reasoning={checkpoint_decision.reasoning}; "
                        f"action={checkpoint_decision.action.model_dump()}"
                    )
                ]
            }
        )
        try:
            model_decision = self.model_policy.choose_action(task_name, guided_observation)
        except Exception as exc:
            return PolicyDecision(
                reasoning=(
                    f"OpenRouter model call failed ({exc}); using the trained RL checkpoint action. "
                    f"{checkpoint_decision.reasoning}"
                ),
                action=checkpoint_decision.action,
            )
        return PolicyDecision(
            reasoning=(
                "OpenRouter Gemma generated this action using the trained RL checkpoint recommendation. "
                f"Model reasoning: {model_decision.reasoning} | Checkpoint recommendation: "
                f"{checkpoint_decision.reasoning}"
            ),
            action=model_decision.action,
        )


def _build_openrouter_policy() -> OpenRouterPolicy | None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or os.environ.get(
        "OPENAI_API_KEY",
        "",
    ).strip()
    if not api_key:
        return None
    config = OpenRouterConfig(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model_name=os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it"),
        site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost:7860"),
        app_name=os.environ.get(
            "OPENROUTER_APP_NAME",
            "EmailMaestro | Executive Assistant Sandbox",
        ),
        temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.1")),
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "600")),
    )
    return OpenRouterPolicy(config=config)


def _build_policy(
    provider: str,
    checkpoint_path: str,
) -> object:
    if provider == "baseline":
        return BaselineAgent()
    if provider == "rl":
        checkpoint_policy = QLearningPolicy.load(
            _ensure_rl_checkpoint(checkpoint_path or _default_rl_checkpoint())
        )
        return OpenRouterGuidedCheckpointPolicy(
            checkpoint_policy=checkpoint_policy,
            model_policy=_build_openrouter_policy(),
        )
    raise ValueError(f"Unsupported app policy provider: {provider}")


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
) -> tuple[str, str, list[list[object]], list[list[object]], list[list[object]], list[list[object]], list[list[object]], str]:
    return (
        json.dumps(observation_payload, indent=2),
        render_status_card(summary_payload),
        _records_to_rows(snapshot_payload["emails"], EMAIL_COLUMNS),
        _records_to_rows(snapshot_payload["todos"], TODO_COLUMNS),
        _records_to_rows(snapshot_payload["files"], FILE_COLUMNS),
        _records_to_rows(snapshot_payload["action_log"], ACTION_LOG_COLUMNS),
        _records_to_rows(trace_rows, TRACE_COLUMNS),
        json.dumps(summary_payload, indent=2),
    )


def configure_checkpoint_input(provider: str) -> dict:
    is_rl = provider == "rl"
    return gr.update(visible=is_rl, interactive=is_rl)


def build_initial_status(task_name: str, provider: str, checkpoint_path: str) -> str:
    return render_status_card(
        _summary_payload(
            run_id="pending",
            task_name=task_name,
            provider=provider,
            policy_name="not started",
            checkpoint_path=checkpoint_path or _default_rl_checkpoint(),
            status="initialized",
            final_score=0.0,
            completed=False,
            termination_reason="Choose a policy and start an episode.",
        )
    )


def run_live_episode(
    task_name: str,
    provider: str,
    max_steps: int,
    checkpoint_path: str,
):
    run_id = uuid.uuid4().hex[:8]
    runner = EpisodeRunner(
        policy=_build_policy(
            provider=provider,
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
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <section class="hero">
              <div class="hero-grid">
                <div class="hero-copy">
                  <div class="hero-kicker">Deterministic Eval Console</div>
                  <h1>Executive Assistant Sandbox</h1>
                  <p>
                    Run the exact same episode loop used in training, inspect each workspace mutation in real time,
                    and compare the deterministic baseline against the OpenRouter-guided RL checkpoint without losing the structure of the task.
                  </p>
                  <div class="hero-strip">
                    <div class="hero-pill">Shared EpisodeRunner path</div>
                    <div class="hero-pill">Seeded scenarios with visible state</div>
                    <div class="hero-pill">Policy debugging without notebook sprawl</div>
                  </div>
                </div>
                <aside class="hero-aside">
                  <p class="hero-aside-label">What This UI Optimizes For</p>
                  <p class="hero-aside-value">Fast policy comparison with readable state.</p>
                  <p class="hero-aside-copy">
                    The interface is intentionally light, structured, and editorial rather than “chat app” themed.
                    Controls stay compact while the workspace and trace remain the visual priority.
                  </p>
                </aside>
              </div>
            </section>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Group(elem_classes=["panel-card", "control-room"]):
                    gr.HTML(
                        """
                        <h2 class="panel-title">Control Room</h2>
                        <p class="panel-copy">
                          Pick a scenario, choose baseline or the OpenRouter-guided trained RL JSON checkpoint, and run a stepwise episode against the same environment used by training and evaluation.
                        </p>
                        """
                    )
                    task = gr.Radio(
                        choices=[
                            "easy_deadline_extraction",
                            "medium_triage_and_negotiation",
                            "hard_rag_reply",
                        ],
                        value="easy_deadline_extraction",
                        label="Scenario",
                        elem_classes=["control-field"],
                    )
                    provider = gr.Radio(
                        choices=["baseline", "rl"],
                        value="baseline",
                        label="Policy",
                        elem_classes=["control-field"],
                    )
                    max_steps = gr.Number(
                        value=12,
                        precision=0,
                        label="Max Steps",
                        elem_classes=["control-field"],
                    )
                    with gr.Accordion("Provider Settings", open=False):
                        checkpoint_path = gr.Textbox(
                            value=_default_rl_checkpoint(),
                            label="RL Checkpoint Path",
                            elem_classes=["control-field"],
                        )
                    with gr.Row():
                        reset = gr.Button("Reset Scenario", variant="secondary")
                        run_episode_btn = gr.Button("Run Episode", variant="primary")
                    gr.HTML(
                        """
                        <p class="footnote">
                          The RL policy loads the trained JSON checkpoint as guidance, then asks OpenRouter Gemma through the OpenAI client to generate the runtime action. If the model call fails, it falls back to the checkpoint action.
                        </p>
                        """
                    )
            with gr.Column(scale=5):
                scenario_brief = gr.HTML(render_scenario_brief("easy_deadline_extraction"))
                status_card = gr.HTML(
                    build_initial_status(
                        "easy_deadline_extraction",
                        "baseline",
                        _default_rl_checkpoint(),
                    )
                )

        with gr.Group(elem_classes=["surface-card", "workspace-grid"]):
            with gr.Tabs():
                with gr.Tab("Live Workspace"):
                    with gr.Row():
                        observation = gr.Code(label="Observation", language="json")
                        summary = gr.Code(label="Run Summary", language="json")
                    with gr.Row():
                        emails = gr.Dataframe(headers=EMAIL_COLUMNS, label="Unread Emails")
                        todos = gr.Dataframe(headers=TODO_COLUMNS, label="Todos")
                    with gr.Row():
                        files = gr.Dataframe(headers=FILE_COLUMNS, label="Search Results")
                        action_log = gr.Dataframe(headers=ACTION_LOG_COLUMNS, label="Action Log")
                with gr.Tab("Episode Trace"):
                    trace_table = gr.Dataframe(headers=TRACE_COLUMNS, label="Episode Trace")

    reset.click(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )
    reset.click(
        fn=render_scenario_brief,
        inputs=[task],
        outputs=[scenario_brief],
    )
    reset.click(
        fn=build_initial_status,
        inputs=[task, provider, checkpoint_path],
        outputs=[status_card],
    )
    provider.change(
        fn=configure_checkpoint_input,
        inputs=[provider],
        outputs=[checkpoint_path],
    )
    provider.change(
        fn=build_initial_status,
        inputs=[task, provider, checkpoint_path],
        outputs=[status_card],
    )
    task.change(
        fn=render_scenario_brief,
        inputs=[task],
        outputs=[scenario_brief],
    )
    task.change(
        fn=build_initial_status,
        inputs=[task, provider, checkpoint_path],
        outputs=[status_card],
    )
    run_episode_btn.click(
        fn=run_live_episode,
        inputs=[task, provider, max_steps, checkpoint_path],
        outputs=[observation, status_card, emails, todos, files, action_log, trace_table, summary],
    )

    demo.load(
        fn=build_snapshot,
        inputs=[task],
        outputs=[observation, emails, todos, files, action_log],
    )
    demo.load(
        fn=configure_checkpoint_input,
        inputs=[provider],
        outputs=[checkpoint_path],
    )
    demo.load(
        fn=render_scenario_brief,
        inputs=[task],
        outputs=[scenario_brief],
    )
    demo.load(
        fn=build_initial_status,
        inputs=[task, provider, checkpoint_path],
        outputs=[status_card],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=APP_RUNTIME.host,
        server_port=APP_RUNTIME.port,
        show_error=True,
        css=APP_CSS,
    )
