from __future__ import annotations

from src.executive_assistant.graders import grade_easy, grade_hard, grade_medium
from src.executive_assistant.models import (
    AssistantAction,
    EmailDetail,
    EmailSummary,
    FileSearchResult,
    TaskReward,
    WorkspaceObservation,
)
from src.executive_assistant.seeds import TASK_SEEDS
from src.executive_assistant.workspace import MockWorkspace


class ExecutiveAssistantEnv:
    def __init__(self, task_name: str = "easy_deadline_extraction") -> None:
        self.task_name = task_name
        self.workspace = MockWorkspace()
        self.last_action_status = "environment initialized"
        self.current_email: EmailDetail | None = None
        self.search_results: list[FileSearchResult] = []
        self.step_count = 0
        self.max_steps = 12

    def reset(self) -> WorkspaceObservation:
        self.workspace = MockWorkspace()
        seed = TASK_SEEDS[self.task_name]
        self.workspace.seed(seed.get("emails", []), seed.get("files", []))
        self.last_action_status = f"scenario reset: {self.task_name}"
        self.current_email = None
        self.search_results = []
        self.step_count = 0
        return self.observe()

    def state(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "last_action_status": self.last_action_status,
            "current_email": self.current_email.model_dump() if self.current_email else None,
            "search_results": [result.model_dump() for result in self.search_results],
            "observation": self.observe().model_dump(),
            "workspace": self.workspace.snapshot(),
        }

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
        recent_actions = [
            f"{row['action_type']}: {row['status']}"
            for row in reversed(self.workspace.list_recent_actions(limit=6))
        ]
        return WorkspaceObservation(
            current_time="2026-04-04T10:00:00Z",
            unread_emails=unread,
            active_todos=todos,
            last_action_status=self.last_action_status,
            current_email=self.current_email,
            search_results=self.search_results,
            action_history=recent_actions,
        )

    def step(self, action: AssistantAction) -> tuple[WorkspaceObservation, TaskReward, bool, dict[str, object]]:
        self.step_count += 1
        if action.action_type == "read_email" and action.target_id is not None:
            row = self.workspace.read_email(action.target_id)
            self.current_email = EmailDetail(**dict(row)) if row else None
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
                context=(
                    f"Created from email {self.current_email.id}: {self.current_email.subject}"
                    if self.current_email
                    else f"Created from task {self.task_name}"
                ),
            )
        elif action.action_type == "archive" and action.target_id is not None:
            self.last_action_status = self.workspace.archive_email(action.target_id)
        elif action.action_type == "search_files" and action.payload:
            results = self.workspace.search_documents(action.payload)
            self.search_results = [
                FileSearchResult(
                    id=row["id"],
                    filename=row["filename"],
                    snippet=row["content_text"][:160],
                )
                for row in results
            ]
            self.last_action_status = f"search returned {len(results)} file(s)"
        else:
            self.last_action_status = "invalid action payload"

        observation = self.observe()
        reward = self.grade()
        if self.step_count >= self.max_steps and not reward.is_done:
            reward = TaskReward(
                step_reward=reward.step_reward,
                total_score=reward.total_score,
                is_done=True,
                reasoning=f"{reward.reasoning}; terminated at step budget",
            )
        done = reward.is_done
        info = {
            "task_name": self.task_name,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "status": self.last_action_status,
            "reasoning": reward.reasoning,
            "total_score": reward.total_score,
            "state": self.state(),
        }
        return observation, reward, done, info

    def grade(self) -> TaskReward:
        if self.task_name == "easy_deadline_extraction":
            return grade_easy(self.workspace)
        if self.task_name == "medium_triage_and_negotiation":
            return grade_medium(self.workspace)
        if self.task_name == "hard_rag_reply":
            return grade_hard(self.workspace)
        return TaskReward(reasoning="No grader configured")
