from __future__ import annotations

from src.executive_assistant.graders import grade_easy, grade_hard, grade_medium
from src.executive_assistant.models import AssistantAction, EmailSummary, TaskReward, WorkspaceObservation
from src.executive_assistant.seeds import TASK_SEEDS
from src.executive_assistant.workspace import MockWorkspace


class ExecutiveAssistantEnv:
    def __init__(self, task_name: str = "easy_deadline_extraction") -> None:
        self.task_name = task_name
        self.workspace = MockWorkspace()
        self.last_action_status = "environment initialized"

    def reset(self) -> WorkspaceObservation:
        self.workspace = MockWorkspace()
        seed = TASK_SEEDS[self.task_name]
        self.workspace.seed(seed.get("emails", []), seed.get("files", []))
        self.last_action_status = f"scenario reset: {self.task_name}"
        return self.observe()

    def observe(self) -> WorkspaceObservation:
        unread = [
            EmailSummary(
                id=row["id"],
                sender=row["sender"],
                subject=row["subject"],
                snippet=row["snippet"],
            )
            for row in self.workspace.get_unread_emails()
        ]
        todos = [row["task_name"] for row in self.workspace.list_todos()]
        return WorkspaceObservation(
            current_time="2026-04-04T10:00:00Z",
            unread_emails=unread,
            active_todos=todos,
            last_action_status=self.last_action_status,
        )

    def step(self, action: AssistantAction) -> tuple[WorkspaceObservation, TaskReward]:
        if action.action_type == "read_email" and action.target_id is not None:
            row = self.workspace.read_email(action.target_id)
            self.last_action_status = "email read" if row else "email not found"
        elif action.action_type == "reply" and action.target_id is not None and action.payload:
            self.last_action_status = self.workspace.send_reply(action.target_id, action.payload)
        elif (
            action.action_type == "forward"
            and action.target_id is not None
            and action.secondary_payload
        ):
            self.last_action_status = self.workspace.forward_email(
                action.target_id,
                action.secondary_payload,
                action.payload,
            )
        elif action.action_type == "add_todo" and action.payload:
            self.last_action_status = self.workspace.create_todo(
                task_name=action.payload,
                deadline_date=action.secondary_payload,
                context=f"Created from task {self.task_name}",
            )
        elif action.action_type == "archive" and action.target_id is not None:
            self.last_action_status = self.workspace.archive_email(action.target_id)
        elif action.action_type == "search_files" and action.payload:
            results = self.workspace.search_documents(action.payload)
            self.last_action_status = f"search returned {len(results)} file(s)"
        else:
            self.last_action_status = "invalid action payload"

        observation = self.observe()
        reward = self.grade()
        return observation, reward

    def grade(self) -> TaskReward:
        if self.task_name == "easy_deadline_extraction":
            return grade_easy(self.workspace)
        if self.task_name == "medium_triage_and_negotiation":
            return grade_medium(self.workspace)
        if self.task_name == "hard_rag_reply":
            return grade_hard(self.workspace)
        return TaskReward(reasoning="No grader configured")
