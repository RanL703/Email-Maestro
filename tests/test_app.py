from pathlib import Path

from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.models import AssistantAction, PolicyDecision
from src.executive_assistant.training import train_q_learning


def test_app_builds_rl_policy_from_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app import _build_policy

    policy, _ = train_q_learning(episodes=12, epsilon=0.1, teacher=BaselineAgent())
    checkpoint = policy.save(tmp_path / "q_policy.json")
    loaded_policy = _build_policy(
        provider="rl",
        checkpoint_path=str(checkpoint),
    )
    assert loaded_policy.checkpoint_policy.epsilon == 0.0


def test_app_builds_missing_rl_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app import _build_policy

    checkpoint = tmp_path / "missing" / "q_policy.json"
    loaded_policy = _build_policy(
        provider="rl",
        checkpoint_path=str(checkpoint),
    )
    assert loaded_policy.checkpoint_policy.epsilon == 0.0
    assert checkpoint.exists()


def test_rl_policy_uses_openrouter_model_with_checkpoint_guidance() -> None:
    from app import OpenRouterGuidedCheckpointPolicy
    from src.executive_assistant.env import ExecutiveAssistantEnv

    class StubModelPolicy:
        def __init__(self) -> None:
            self.observation = None

        def choose_action(self, task_name, observation):
            self.observation = observation
            return PolicyDecision(
                reasoning="Followed the checkpoint hint.",
                action=AssistantAction(action_type="read_email", target_id=1),
            )

    q_policy, _ = train_q_learning(episodes=12, epsilon=0.1, teacher=BaselineAgent())
    model_policy = StubModelPolicy()
    policy = OpenRouterGuidedCheckpointPolicy(q_policy, model_policy)
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    decision = policy.choose_action("easy_deadline_extraction", env.reset())
    assert decision.action.action_type == "read_email"
    assert "OpenRouter Gemma generated" in decision.reasoning
    assert model_policy.observation is not None
    assert any(
        "Trained RL checkpoint recommendation" in entry
        for entry in model_policy.observation.action_history
    )


def test_app_stepwise_episode_generator_yields_updates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app import run_live_episode

    policy, _ = train_q_learning(episodes=12, epsilon=0.1, teacher=BaselineAgent())
    checkpoint = policy.save(tmp_path / "q_policy.json")
    generator = run_live_episode(
        task_name="hard_rag_reply",
        provider="rl",
        max_steps=12,
        checkpoint_path=str(checkpoint),
    )
    first_frame = next(generator)
    assert "scenario reset" in first_frame[0]
    assert "requested_provider" in first_frame[-1]
    assert "Run pending" in first_frame[1] or "Run " in first_frame[1]
    later_frame = None
    for later_frame in generator:
        pass
    assert later_frame is not None
    assert "reply drafted" in later_frame[0] or "search returned" in later_frame[0]
