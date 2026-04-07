from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.training import QLearningPolicy, evaluate_q_policy, train_q_learning


def test_train_q_learning_returns_scores() -> None:
    policy, scores = train_q_learning(episodes=24, epsilon=0.1, teacher=BaselineAgent())
    evaluation = evaluate_q_policy(policy)
    assert scores
    assert set(evaluation) == {
        "easy_deadline_extraction",
        "medium_triage_and_negotiation",
        "hard_rag_reply",
    }
    assert evaluation == {
        "easy_deadline_extraction": 1.0,
        "medium_triage_and_negotiation": 1.0,
        "hard_rag_reply": 1.0,
    }


def test_q_learning_policy_checkpoint_roundtrip(tmp_path) -> None:
    policy, _ = train_q_learning(episodes=12, epsilon=0.1, teacher=BaselineAgent())
    checkpoint = policy.save(tmp_path / "q_policy.json")
    loaded = QLearningPolicy.load(checkpoint)
    evaluation = evaluate_q_policy(loaded)
    assert set(evaluation) == {
        "easy_deadline_extraction",
        "medium_triage_and_negotiation",
        "hard_rag_reply",
    }
    assert evaluation == {
        "easy_deadline_extraction": 1.0,
        "medium_triage_and_negotiation": 1.0,
        "hard_rag_reply": 1.0,
    }
