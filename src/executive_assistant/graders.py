from __future__ import annotations

import re

from src.executive_assistant.models import TaskReward
from src.executive_assistant.workspace import MockWorkspace


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def grade_easy(workspace: MockWorkspace) -> TaskReward:
    expected = {
        ("proposal due", "2026-04-10"),
        ("prototype due", "2026-04-20"),
        ("final report due", "2026-04-30"),
    }
    todos = workspace.connection.execute(
        "SELECT task_name, deadline_date FROM Todos"
    ).fetchall()
    normalized = {
        (row["task_name"].strip().lower(), (row["deadline_date"] or "").strip()) for row in todos
    }
    matched = len(expected & normalized)
    incorrect = len(normalized - expected)
    read_source = workspace.connection.execute(
        "SELECT COUNT(*) FROM ActionLog WHERE action_type = 'read_email' AND target_id = 1"
    ).fetchone()[0]
    archived = workspace.connection.execute(
        "SELECT COUNT(*) FROM Emails WHERE id = 1 AND is_archived = 1"
    ).fetchone()[0]

    score = 0.15 if read_source else 0.0
    score += matched * 0.25
    score += 0.10 if archived else 0.0
    score -= incorrect * 0.10
    total_score = _clamp_score(score)
    done = matched == 3 and archived == 1 and incorrect == 0
    return TaskReward(
        step_reward=total_score,
        total_score=total_score,
        is_done=done,
        reasoning=(
            "Extracted all three deadlines and archived the source email"
            if done
            else f"Matched {matched}/3 deadlines, archived={bool(archived)}, incorrect_todos={incorrect}"
        ),
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
    correct_forward = workspace.connection.execute(
        """
        SELECT COUNT(*) FROM ActionLog
        WHERE action_type = 'forward'
          AND secondary_payload = 'manager@company.com'
          AND target_id = (
              SELECT id FROM Emails WHERE sender = 'client@company.com' LIMIT 1
          )
        """
    ).fetchone()[0]
    reply = workspace.connection.execute(
        """
        SELECT payload, target_id FROM ActionLog
        WHERE action_type = 'reply'
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    important_archived = workspace.connection.execute(
        """
        SELECT COUNT(*) FROM Emails
        WHERE sender IN ('client@company.com', 'teammate@company.com')
          AND is_archived = 1
        """
    ).fetchone()[0]

    score = 0.0
    score += min(newsletters_archived, 3) * 0.1
    if correct_forward >= 1:
        score += 0.4
    elif forwarded >= 1:
        score += 0.1

    teammate_id = workspace.connection.execute(
        "SELECT id FROM Emails WHERE sender = 'teammate@company.com' LIMIT 1"
    ).fetchone()[0]
    if (
        reply
        and reply["target_id"] == teammate_id
        and re.search(r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b", reply["payload"] or "")
    ):
        score += 0.3
    elif reply and re.search(r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b", reply["payload"] or ""):
        score += 0.1

    score -= important_archived * 0.15
    total_score = _clamp_score(score)

    return TaskReward(
        step_reward=total_score,
        total_score=total_score,
        is_done=newsletters_archived == 3 and correct_forward >= 1 and total_score >= 1.0,
        reasoning=(
            "Archived newsletters, escalated client complaint, and proposed a meeting time"
            if newsletters_archived == 3 and correct_forward >= 1 and total_score >= 1.0
            else (
                f"newsletters_archived={newsletters_archived}/3, "
                f"correct_forward={correct_forward}, important_archived={important_archived}"
            )
        ),
    )


def grade_hard(workspace: MockWorkspace) -> TaskReward:
    search_called = workspace.connection.execute(
        "SELECT COUNT(*) FROM ActionLog WHERE action_type = 'search_files'"
    ).fetchone()[0]
    targeted_search = workspace.connection.execute(
        """
        SELECT COUNT(*) FROM ActionLog
        WHERE action_type = 'search_files'
          AND LOWER(COALESCE(payload, '')) LIKE '%q3%'
          AND LOWER(COALESCE(payload, '')) LIKE '%architecture%'
        """
    ).fetchone()[0]
    reply = workspace.connection.execute(
        """
        SELECT payload, target_id FROM ActionLog
        WHERE action_type = 'reply'
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    vip_id = workspace.connection.execute(
        "SELECT id FROM Emails WHERE sender = 'vip.stakeholder@company.com' LIMIT 1"
    ).fetchone()[0]

    score = 0.1 if search_called >= 1 else 0.0
    score += 0.2 if targeted_search >= 1 else 0.0
    if reply and reply["target_id"] == vip_id:
        payload = reply["payload"] or ""
        metrics_found = sum(
            metric in payload for metric in ("99.95%", "182ms", "14%")
        )
        score += metrics_found * 0.2
        if payload.lower().startswith("hello") or "regards" in payload.lower():
            score += 0.1
    total_score = _clamp_score(score)

    return TaskReward(
        step_reward=total_score,
        total_score=total_score,
        is_done=total_score >= 1.0,
        reasoning=(
            "Searched the report and replied with the required metrics"
            if total_score >= 1.0
            else f"search_called={search_called}, targeted_search={targeted_search}, score={total_score}"
        ),
    )
