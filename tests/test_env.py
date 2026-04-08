from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import AssistantAction


def test_easy_env_reset_exposes_seeded_email() -> None:
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    assert len(observation.unread_emails) == 1


def test_easy_env_can_add_todo() -> None:
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    env.reset()
    observation, reward, done, info = env.step(
        AssistantAction(
            action_type="add_todo",
            payload="Proposal due",
            secondary_payload="2026-04-10",
        )
    )
    assert "Proposal due" in observation.active_todos
    assert reward.total_score >= 0.0
    assert done is False
    assert info["task_name"] == "easy_deadline_extraction"


def test_read_email_populates_current_email() -> None:
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    observation, _, _, _ = env.step(
        AssistantAction(action_type="read_email", target_id=observation.unread_emails[0].id)
    )
    assert observation.current_email is not None
    assert "proposal due" in observation.current_email.body.lower()


def test_search_files_populates_results() -> None:
    env = ExecutiveAssistantEnv(task_name="hard_rag_reply")
    env.reset()
    observation, _, _, _ = env.step(AssistantAction(action_type="search_files", payload="Q3 Architecture"))
    assert observation.search_results
    assert observation.search_results[0].filename == "Q3_Architecture_Report.txt"


def test_state_returns_workspace_snapshot() -> None:
    env = ExecutiveAssistantEnv(task_name="medium_triage_and_negotiation")
    env.reset()
    state = env.state()
    assert state["task_name"] == "medium_triage_and_negotiation"
    assert "workspace" in state
    assert "emails" in state["workspace"]
