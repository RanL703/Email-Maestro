from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmailSummary(BaseModel):
    id: int
    sender: str
    subject: str
    snippet: str


class WorkspaceObservation(BaseModel):
    current_time: str
    unread_emails: list[EmailSummary]
    active_todos: list[str]
    last_action_status: str


class AssistantAction(BaseModel):
    action_type: Literal[
        "read_email",
        "reply",
        "forward",
        "add_todo",
        "archive",
        "search_files",
    ]
    target_id: int | None = None
    payload: str | None = None
    secondary_payload: str | None = None


class TaskReward(BaseModel):
    step_reward: float = Field(default=0.0)
    total_score: float = Field(default=0.0)
    is_done: bool = Field(default=False)
    reasoning: str = Field(default="")
