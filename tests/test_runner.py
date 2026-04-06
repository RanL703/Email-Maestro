from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.runner import run_policy_suite


def test_run_policy_suite_returns_all_requested_tasks() -> None:
    traces = run_policy_suite(
        policy=BaselineAgent(),
        task_names=["easy_deadline_extraction", "hard_rag_reply"],
    )
    assert set(traces) == {"easy_deadline_extraction", "hard_rag_reply"}
