from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.runner import run_policy_suite


def test_run_policy_suite_returns_all_requested_tasks() -> None:
    traces = run_policy_suite(
        policy=BaselineAgent(),
        task_names=["easy_deadline_extraction", "hard_rag_reply"],
    )
    assert set(traces) == {"easy_deadline_extraction", "hard_rag_reply"}


def test_episode_runner_exposes_explicit_workflow_steps() -> None:
    from src.executive_assistant.runner import EpisodeRunner

    runner = EpisodeRunner(policy=BaselineAgent(), max_steps=12)
    env, observation = runner.initialize("easy_deadline_extraction")
    _, next_observation, reward, record = runner.advance(
        "easy_deadline_extraction",
        env,
        observation,
    )
    assert record.step_index == 1
    assert next_observation.last_action_status == "email read"
    assert reward.is_done is False
