from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmailSummary(BaseModel):
    id: int
    sender: str
    subject: str
    snippet: str


class EmailDetail(BaseModel):
    id: int
    sender: str
    recipient: str
    subject: str
    body: str
    timestamp: str


class FileSearchResult(BaseModel):
    id: int
    filename: str
    snippet: str


class WorkspaceObservation(BaseModel):
    current_time: str
    unread_emails: list[EmailSummary]
    active_todos: list[str]
    last_action_status: str
    current_email: EmailDetail | None = None
    search_results: list[FileSearchResult] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)


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


class PolicyDecision(BaseModel):
    reasoning: str = Field(default="")
    action: AssistantAction
