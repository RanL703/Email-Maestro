from pathlib import Path

from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.training import train_q_learning


def test_app_builds_rl_policy_from_checkpoint(tmp_path) -> None:
    from app import _build_policy

    policy, _ = train_q_learning(episodes=12, epsilon=0.1, teacher=BaselineAgent())
    checkpoint = policy.save(tmp_path / "q_policy.json")
    loaded_policy = _build_policy(
        provider="rl",
        model_name="google/gemma-4-31b-it",
        api_key="",
        checkpoint_path=str(checkpoint),
    )
    assert loaded_policy.epsilon == 0.0


def test_app_stepwise_episode_generator_yields_updates(tmp_path) -> None:
    from app import run_live_episode

    policy, _ = train_q_learning(episodes=12, epsilon=0.1, teacher=BaselineAgent())
    checkpoint = policy.save(tmp_path / "q_policy.json")
    generator = run_live_episode(
        task_name="hard_rag_reply",
        provider="rl",
        model_name="google/gemma-4-31b-it",
        api_key="",
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
