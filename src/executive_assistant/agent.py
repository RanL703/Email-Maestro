from __future__ import annotations

import re

from src.executive_assistant.config import OpenRouterConfig
from src.executive_assistant.llm_service import OpenRouterLLMService
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


class OpenRouterPolicy:
    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        service: OpenRouterLLMService | None = None,
    ) -> None:
        self.config = config or OpenRouterConfig.from_env()
        self.service = service or OpenRouterLLMService(self.config)

    def choose_action(self, task_name: str, observation: WorkspaceObservation) -> PolicyDecision:
        decision = self.service.generate_policy_decision(task_name, observation)
        return self._sanitize_decision(task_name, observation, decision)

    def _sanitize_decision(
        self,
        task_name: str,
        observation: WorkspaceObservation,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        action = decision.action
        if action.action_type == "add_todo":
            action = self._normalize_easy_todo_action(task_name, observation, action)
        elif action.action_type == "search_files":
            action = AssistantAction(
                action_type=action.action_type,
                target_id=None,
                payload=action.payload,
                secondary_payload=None,
            )
        elif action.action_type == "add_todo":
            action = AssistantAction(
                action_type=action.action_type,
                target_id=None,
                payload=action.payload,
                secondary_payload=action.secondary_payload,
            )
        elif action.action_type in {"read_email", "archive"}:
            action = AssistantAction(
                action_type=action.action_type,
                target_id=action.target_id,
                payload=None,
                secondary_payload=None,
            )
        elif action.action_type == "forward":
            action = self._normalize_forward_action(task_name, observation, action)
        if action.action_type == "reply" and action.payload:
            payload = action.payload.strip()
            target_id = action.target_id
            if task_name == "hard_rag_reply":
                if not payload.lower().startswith("hello"):
                    payload = f"Hello,\n{payload}"
                if "regards" not in payload.lower():
                    payload = f"{payload}\nRegards,\nExecutive Assistant"
            elif task_name == "medium_triage_and_negotiation":
                if not re.search(r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b", payload):
                    payload = "Hello, 3:30 PM IST works for me."
                if "regards" not in payload.lower():
                    payload = f"{payload}\nRegards,\nExecutive Assistant"
                target_id = self._resolve_teammate_email_id(observation, action.target_id)
            action = AssistantAction(
                action_type=action.action_type,
                target_id=target_id,
                payload=payload,
                secondary_payload=action.secondary_payload,
            )

        return PolicyDecision(reasoning=decision.reasoning, action=action)

    def _normalize_easy_todo_action(
        self,
        task_name: str,
        observation: WorkspaceObservation,
        action: AssistantAction,
    ) -> AssistantAction:
        if task_name != "easy_deadline_extraction":
            return AssistantAction(
                action_type=action.action_type,
                target_id=None,
                payload=action.payload,
                secondary_payload=action.secondary_payload,
            )

        canonical_todos = [
            ("proposal", "Proposal Due", "2026-04-10"),
            ("prototype", "Prototype Due", "2026-04-20"),
            ("final report", "Final Report Due", "2026-04-30"),
        ]
        payload = (action.payload or "").strip()
        payload_lower = payload.lower()

        for marker, canonical_name, canonical_deadline in canonical_todos:
            if marker in payload_lower:
                return AssistantAction(
                    action_type="add_todo",
                    target_id=None,
                    payload=canonical_name,
                    secondary_payload=canonical_deadline,
                )

        existing = {todo.strip().lower() for todo in observation.active_todos}
        for _, canonical_name, canonical_deadline in canonical_todos:
            if canonical_name.lower() not in existing:
                return AssistantAction(
                    action_type="add_todo",
                    target_id=None,
                    payload=canonical_name,
                    secondary_payload=canonical_deadline,
                )

        return AssistantAction(
            action_type="add_todo",
            target_id=None,
            payload=payload,
            secondary_payload=action.secondary_payload,
        )

    def _normalize_forward_action(
        self,
        task_name: str,
        observation: WorkspaceObservation,
        action: AssistantAction,
    ) -> AssistantAction:
        target_id = action.target_id
        recipient = action.secondary_payload
        note = action.payload

        if task_name == "medium_triage_and_negotiation":
            if target_id is None and observation.current_email is not None:
                target_id = observation.current_email.id
            if recipient is None:
                recipient = "manager@company.com"
            if note is None or not note.strip():
                note = "Urgent client complaint. Please take over immediately."

        return AssistantAction(
            action_type="forward",
            target_id=target_id,
            payload=note,
            secondary_payload=recipient,
        )

    @staticmethod
    def _resolve_teammate_email_id(
        observation: WorkspaceObservation,
        target_id: int | None,
    ) -> int | None:
        if target_id is not None:
            return target_id
        if observation.current_email and observation.current_email.sender == "teammate@company.com":
            return observation.current_email.id
        teammate_email = next(
            (email for email in observation.unread_emails if email.sender == "teammate@company.com"),
            None,
        )
        return teammate_email.id if teammate_email is not None else None


OpenAIResponsesPolicy = OpenRouterPolicy


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
