from src.executive_assistant.models import AssistantAction


def test_action_model_accepts_known_action_type() -> None:
    action = AssistantAction(action_type="archive", target_id=1)
    assert action.action_type == "archive"
