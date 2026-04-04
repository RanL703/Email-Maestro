from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import AssistantAction


def test_easy_env_reset_exposes_seeded_email() -> None:
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    assert len(observation.unread_emails) == 1


def test_easy_env_can_add_todo() -> None:
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    env.reset()
    observation, reward = env.step(
        AssistantAction(
            action_type="add_todo",
            payload="Proposal due",
            secondary_payload="2026-04-10",
        )
    )
    assert "Proposal due" in observation.active_todos
    assert reward.total_score >= 0.0
