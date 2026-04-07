import pytest

from src.executive_assistant.agent import (
    ActionCatalog,
    BaselineAgent,
    OpenRouterPolicy,
    smoke_test_training_pipeline,
)
from src.executive_assistant.config import OpenRouterConfig
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import AssistantAction, PolicyDecision
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


def test_openrouter_policy_uses_service() -> None:
    class StubService:
        def generate_policy_decision(self, task_name, observation):
            return BaselineAgent().choose_action(task_name, observation)

    policy = OpenRouterPolicy(
        config=OpenRouterConfig(api_key="test-key"),
        service=StubService(),
    )
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    decision = policy.choose_action("easy_deadline_extraction", observation)
    assert decision.action.action_type == "read_email"


def test_openrouter_policy_sanitizes_hard_reply_payload() -> None:
    class StubService:
        def generate_policy_decision(self, task_name, observation):
            return PolicyDecision(
                reasoning="Reply with metrics.",
                action=AssistantAction(
                    action_type="reply",
                    target_id=1,
                    payload="System availability: 99.95%, Mean API latency: 182ms, Infrastructure cost reduction: 14%.",
                    secondary_payload=None,
                ),
            )

    policy = OpenRouterPolicy(
        config=OpenRouterConfig(api_key="test-key"),
        service=StubService(),
    )
    env = ExecutiveAssistantEnv(task_name="hard_rag_reply")
    observation = env.reset()
    observation, _ = env.step(AssistantAction(action_type="read_email", target_id=1))
    observation, _ = env.step(AssistantAction(action_type="search_files", payload="Q3 Architecture"))
    decision = policy.choose_action("hard_rag_reply", observation)
    assert decision.action.payload is not None
    assert decision.action.payload.lower().startswith("hello")
    assert "regards" in decision.action.payload.lower()


def test_openrouter_policy_clears_unused_search_fields() -> None:
    class StubService:
        def generate_policy_decision(self, task_name, observation):
            return PolicyDecision(
                reasoning="Search for the report.",
                action=AssistantAction(
                    action_type="search_files",
                    target_id=99,
                    payload="Q3 architecture report",
                    secondary_payload="unused",
                ),
            )

    policy = OpenRouterPolicy(
        config=OpenRouterConfig(api_key="test-key"),
        service=StubService(),
    )
    env = ExecutiveAssistantEnv(task_name="hard_rag_reply")
    observation = env.reset()
    decision = policy.choose_action("hard_rag_reply", observation)
    assert decision.action.target_id is None
    assert decision.action.secondary_payload is None


def test_openrouter_policy_normalizes_easy_todo_payload() -> None:
    class StubService:
        def generate_policy_decision(self, task_name, observation):
            return PolicyDecision(
                reasoning="Track the proposal deadline.",
                action=AssistantAction(
                    action_type="add_todo",
                    payload="proposal",
                    secondary_payload=None,
                ),
            )

    policy = OpenRouterPolicy(
        config=OpenRouterConfig(api_key="test-key"),
        service=StubService(),
    )
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    observation, _ = env.step(AssistantAction(action_type="read_email", target_id=1))
    decision = policy.choose_action("easy_deadline_extraction", observation)
    assert decision.action.payload == "Proposal Due"
    assert decision.action.secondary_payload == "2026-04-10"


def test_openrouter_policy_repairs_medium_forward_fields() -> None:
    class StubService:
        def generate_policy_decision(self, task_name, observation):
            return PolicyDecision(
                reasoning="Forward the complaint.",
                action=AssistantAction(
                    action_type="forward",
                    target_id=None,
                    payload=None,
                    secondary_payload=None,
                ),
            )

    policy = OpenRouterPolicy(
        config=OpenRouterConfig(api_key="test-key"),
        service=StubService(),
    )
    env = ExecutiveAssistantEnv(task_name="medium_triage_and_negotiation")
    observation = env.reset()
    observation, _ = env.step(AssistantAction(action_type="archive", target_id=1))
    observation, _ = env.step(AssistantAction(action_type="archive", target_id=2))
    observation, _ = env.step(AssistantAction(action_type="archive", target_id=3))
    observation, _ = env.step(AssistantAction(action_type="read_email", target_id=4))
    decision = policy.choose_action("medium_triage_and_negotiation", observation)
    assert decision.action.target_id == 4
    assert decision.action.secondary_payload == "manager@company.com"
    assert "Urgent client complaint" in (decision.action.payload or "")
