from __future__ import annotations

import re

from src.executive_assistant.models import TaskReward
from src.executive_assistant.workspace import MockWorkspace


def grade_easy(workspace: MockWorkspace) -> TaskReward:
    count = workspace.connection.execute(
        "SELECT COUNT(*) FROM Todos WHERE deadline_date IS NOT NULL"
    ).fetchone()[0]
    archived = workspace.connection.execute(
        "SELECT COUNT(*) FROM Emails WHERE is_archived = 1"
    ).fetchone()[0]
    done = count == 3 and archived >= 1
    return TaskReward(
        step_reward=1.0 if done else 0.0,
        total_score=1.0 if done else 0.0,
        is_done=done,
        reasoning="Three deadline todos and email archived" if done else "Task incomplete",
    )


def grade_medium(workspace: MockWorkspace) -> TaskReward:
    newsletters_archived = workspace.connection.execute(
        """
        SELECT COUNT(*) FROM Emails
        WHERE sender IN ('news@updates.example', 'promotions@vendor.example', 'events@community.example')
          AND is_archived = 1
        """
    ).fetchone()[0]
    forwarded = workspace.connection.execute(
        """
        SELECT COUNT(*) FROM ActionLog
        WHERE action_type = 'forward' AND secondary_payload = 'manager@company.com'
        """
    ).fetchone()[0]
    reply = workspace.connection.execute(
        """
        SELECT payload FROM ActionLog
        WHERE action_type = 'reply'
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()

    score = 0.0
    if newsletters_archived == 3:
        score += 0.3
    if forwarded >= 1:
        score += 0.4
    if reply and re.search(r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b", reply[0]):
        score += 0.3

    return TaskReward(
        step_reward=score,
        total_score=score,
        is_done=score >= 1.0,
        reasoning="Medium-task grading completed",
    )


def grade_hard(workspace: MockWorkspace) -> TaskReward:
    search_called = workspace.connection.execute(
        "SELECT COUNT(*) FROM ActionLog WHERE action_type = 'search_files'"
    ).fetchone()[0]
    reply = workspace.connection.execute(
        """
        SELECT payload FROM ActionLog
        WHERE action_type = 'reply'
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()

    score = 0.3 if search_called >= 1 else 0.0
    if reply and "99.95%" in reply[0] and "182ms" in reply[0] and "14%" in reply[0]:
        score += 0.7

    return TaskReward(
        step_reward=score,
        total_score=score,
        is_done=score >= 1.0,
        reasoning="Hard-task grading completed",
    )
