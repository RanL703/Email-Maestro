import pytest

from src.executive_assistant.agent import (
    ActionCatalog,
    BaselineAgent,
    OpenAIResponsesPolicy,
    smoke_test_training_pipeline,
)
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.runner import EpisodeRunner, export_traces_jsonl


def test_action_catalog_exposes_candidate_actions() -> None:
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    actions = ActionCatalog.enumerate_actions(observation)
    assert any(action.action_type == "read_email" for action in actions)


def test_baseline_pipeline_solves_seeded_tasks() -> None:
    traces = smoke_test_training_pipeline()
    assert traces["easy_deadline_extraction"].completed is True
    assert traces["medium_triage_and_negotiation"].completed is True
    assert traces["hard_rag_reply"].completed is True


def test_episode_runner_produces_trace_records() -> None:
    trace = EpisodeRunner(policy=BaselineAgent()).run("easy_deadline_extraction")
    assert trace.steps
    assert trace.steps[-1].reward["is_done"] is True


def test_export_traces_jsonl_writes_output(tmp_path) -> None:
    trace = EpisodeRunner(policy=BaselineAgent()).run("hard_rag_reply")
    output_path = export_traces_jsonl([trace], tmp_path / "traces.jsonl")
    assert output_path.exists()
    assert output_path.read_text().strip()


def test_openai_policy_requires_api_key() -> None:
    policy = OpenAIResponsesPolicy(api_key=None)
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        policy.choose_action("easy_deadline_extraction", observation)
