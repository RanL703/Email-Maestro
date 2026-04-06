from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.executive_assistant.models import AssistantAction, PolicyDecision, WorkspaceObservation
from src.executive_assistant.runner import EpisodeRunner, EpisodeTrace, run_policy_suite


class ActionCatalog:
    """Finite action templates for smoke-testing and future policy indexing."""

    @staticmethod
    def enumerate_actions(observation: WorkspaceObservation) -> list[AssistantAction]:
        actions: list[AssistantAction] = []
        for email in observation.unread_emails:
            actions.append(AssistantAction(action_type="read_email", target_id=email.id))
            actions.append(AssistantAction(action_type="archive", target_id=email.id))
            actions.append(
                AssistantAction(
                    action_type="forward",
                    target_id=email.id,
                    secondary_payload="manager@company.com",
                    payload="Escalating this for review.",
                )
            )
        if observation.current_email is not None:
            actions.append(
                AssistantAction(
                    action_type="reply",
                    target_id=observation.current_email.id,
                    payload="Hello, I will follow up shortly.\nRegards, Executive Assistant",
                )
            )
        actions.extend(
            [
                AssistantAction(action_type="search_files", payload="Q3 Architecture"),
                AssistantAction(action_type="search_files", payload="architecture metrics"),
            ]
        )
        return actions


class BaselineAgent:
    """Deterministic baseline policy for seeded scenarios and training-pipeline smoke tests."""

    def __init__(self, model_name: str = "deterministic-baseline-v1") -> None:
        self.model_name = model_name

    def choose_action(self, task_name: str, observation: WorkspaceObservation) -> PolicyDecision:
        if task_name == "easy_deadline_extraction":
            return self._choose_easy_action(observation)
        if task_name == "medium_triage_and_negotiation":
            return self._choose_medium_action(observation)
        if task_name == "hard_rag_reply":
            return self._choose_hard_action(observation)
        raise ValueError(f"Unsupported task: {task_name}")

    def _choose_easy_action(self, observation: WorkspaceObservation) -> PolicyDecision:
        if observation.current_email is None:
            email = observation.unread_emails[0]
            return PolicyDecision(
                reasoning="Read the seeded deadline email before extracting any tasks.",
                action=AssistantAction(action_type="read_email", target_id=email.id),
            )

        deadlines = self._extract_deadlines(observation.current_email.body)
        existing = {todo.strip().lower() for todo in observation.active_todos}
        for task_name, deadline_date in deadlines:
            if task_name.lower() not in existing:
                return PolicyDecision(
                    reasoning=f"Add the missing todo '{task_name}' with deadline {deadline_date}.",
                    action=AssistantAction(
                        action_type="add_todo",
                        payload=task_name,
                        secondary_payload=deadline_date,
                    ),
                )
        return PolicyDecision(
            reasoning="All deadlines are captured, so archive the source email.",
            action=AssistantAction(action_type="archive", target_id=observation.current_email.id),
        )

    def _choose_medium_action(self, observation: WorkspaceObservation) -> PolicyDecision:
        newsletters = {
            "news@updates.example",
            "promotions@vendor.example",
            "events@community.example",
        }
        action_history = " ".join(observation.action_history).lower()
        for email in observation.unread_emails:
            if email.sender in newsletters:
                return PolicyDecision(
                    reasoning=f"Archive non-actionable newsletter from {email.sender}.",
                    action=AssistantAction(action_type="archive", target_id=email.id),
                )

        client_email = next(
            (email for email in observation.unread_emails if email.sender == "client@company.com"),
            None,
        )
        if client_email is not None and "forward: forwarded to manager@company.com" not in action_history:
            return PolicyDecision(
                reasoning="Escalate the urgent client complaint to the manager.",
                action=AssistantAction(
                    action_type="forward",
                    target_id=client_email.id,
                    secondary_payload="manager@company.com",
                    payload="Urgent client complaint. Please take over immediately.",
                ),
            )

        teammate_email = next(
            (email for email in observation.unread_emails if email.sender == "teammate@company.com"),
            None,
        )
        if teammate_email is not None and "reply: reply drafted" not in action_history:
            return PolicyDecision(
                reasoning="Reply to the reschedule request with a concrete proposed time.",
                action=AssistantAction(
                    action_type="reply",
                    target_id=teammate_email.id,
                    payload="Hello, 3:30 PM IST works for me. Regards, Executive Assistant",
                ),
            )

        if observation.current_email is not None:
            return PolicyDecision(
                reasoning="Archive the currently open message to reduce inbox clutter.",
                action=AssistantAction(action_type="archive", target_id=observation.current_email.id),
            )
        raise RuntimeError("No valid medium-task action available")

    def _choose_hard_action(self, observation: WorkspaceObservation) -> PolicyDecision:
        if observation.current_email is None:
            email = observation.unread_emails[0]
            return PolicyDecision(
                reasoning="Read the stakeholder email to ground the response request.",
                action=AssistantAction(action_type="read_email", target_id=email.id),
            )

        if not observation.search_results:
            return PolicyDecision(
                reasoning="Search the local report store for the Q3 architecture document.",
                action=AssistantAction(action_type="search_files", payload="Q3 Architecture"),
            )

        metrics = self._extract_report_metrics(observation.search_results[0].snippet)
        payload = (
            "Hello,\n"
            f"Here are the requested Q3 architecture metrics: availability {metrics['availability']}, "
            f"mean API latency {metrics['latency']}, and infrastructure cost reduction {metrics['cost_reduction']}.\n"
            "Regards,\nExecutive Assistant"
        )
        return PolicyDecision(
            reasoning="Reply with the three requested metrics pulled from the report search results.",
            action=AssistantAction(
                action_type="reply",
                target_id=observation.current_email.id,
                payload=payload,
            ),
        )

    @staticmethod
    def _extract_deadlines(email_body: str) -> list[tuple[str, str]]:
        pattern = re.compile(r"([a-z ]+ due)\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
        cleaned: list[tuple[str, str]] = []
        for task, date in pattern.findall(email_body):
            normalized_task = re.sub(r"^(and\s+)", "", task.strip(), flags=re.IGNORECASE)
            cleaned.append((normalized_task.title(), date))
        return cleaned

    @staticmethod
    def _extract_report_metrics(snippet: str) -> dict[str, str]:
        metrics = {
            "availability": re.search(r"(\d+\.\d+%)", snippet),
            "latency": re.search(r"(\d+ms)", snippet),
            "cost_reduction": re.search(r"(\d+%)", snippet.split("Infrastructure cost reduction:")[-1]),
        }
        return {
            "availability": metrics["availability"].group(1) if metrics["availability"] else "unknown",
            "latency": metrics["latency"].group(1) if metrics["latency"] else "unknown",
            "cost_reduction": (
                metrics["cost_reduction"].group(1) if metrics["cost_reduction"] else "unknown"
            ),
        }


class OpenAIResponsesPolicy:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1/responses",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url

    def choose_action(self, task_name: str, observation: WorkspaceObservation) -> PolicyDecision:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required to use OpenAIResponsesPolicy.")

        payload = {
            "model": self.model_name,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._build_system_prompt(task_name),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(observation.model_dump(), indent=2),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "assistant_policy_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "reasoning": {"type": "string"},
                            "action": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action_type": {
                                        "type": "string",
                                        "enum": [
                                            "read_email",
                                            "reply",
                                            "forward",
                                            "add_todo",
                                            "archive",
                                            "search_files",
                                        ],
                                    },
                                    "target_id": {"type": ["integer", "null"]},
                                    "payload": {"type": ["string", "null"]},
                                    "secondary_payload": {"type": ["string", "null"]},
                                },
                                "required": [
                                    "action_type",
                                    "target_id",
                                    "payload",
                                    "secondary_payload",
                                ],
                            },
                        },
                        "required": ["reasoning", "action"],
                    },
                }
            },
        }
        response = self._post(payload)
        parsed = self._extract_structured_output(response)
        return PolicyDecision.model_validate(parsed)

    def _build_system_prompt(self, task_name: str) -> str:
        return (
            "You are controlling a deterministic executive assistant environment. "
            "Return exactly one action and short reasoning. "
            "Prefer progress toward the active task over passive behavior. "
            f"Active scenario: {task_name}. "
            "Use only IDs visible in the observation. "
            "For deadline extraction, read first, then create exact todos with exact dates, then archive. "
            "For triage, archive newsletters, forward the urgent client complaint to manager@company.com, "
            "and reply to the meeting request with a concrete time string. "
            "For RAG, read the email, search files for the report, then reply with the exact metrics from the result."
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    @staticmethod
    def _extract_structured_output(response: dict[str, Any]) -> dict[str, Any]:
        if isinstance(response.get("output_parsed"), dict):
            return response["output_parsed"]

        for item in response.get("output", []):
            for content in item.get("content", []):
                text_value = content.get("text")
                if isinstance(text_value, str):
                    return json.loads(text_value)
                if isinstance(content.get("json"), dict):
                    return content["json"]
        raise RuntimeError("Unable to extract structured action from OpenAI response.")


def run_episode(task_name: str, max_steps: int = 12) -> EpisodeTrace:
    runner = EpisodeRunner(policy=BaselineAgent(), max_steps=max_steps)
    return runner.run(task_name)


def smoke_test_training_pipeline() -> dict[str, EpisodeTrace]:
    return run_policy_suite(
        policy=BaselineAgent(),
        task_names=[
            "easy_deadline_extraction",
            "medium_triage_and_negotiation",
            "hard_rag_reply",
        ],
    )
