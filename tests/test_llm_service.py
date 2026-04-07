from src.executive_assistant.config import OpenRouterConfig
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.llm_service import OpenRouterLLMService


def test_openrouter_service_parses_policy_decision() -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            class Message:
                content = (
                    '{"reasoning":"Read first","action":{"action_type":"read_email","target_id":1,'
                    '"payload":null,"secondary_payload":null}}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    service = OpenRouterLLMService(
        config=OpenRouterConfig(api_key="test-key"),
        client=FakeClient(),
    )
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    decision = service.generate_policy_decision("easy_deadline_extraction", observation)
    assert decision.action.action_type == "read_email"


def test_openrouter_service_repairs_invalid_json() -> None:
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1

            class Message:
                content = "not valid json" if self.calls == 1 else (
                    '{"reasoning":"Recovered","action":{"action_type":"read_email","target_id":1,'
                    '"payload":null,"secondary_payload":null}}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    fake_completions = FakeCompletions()

    class FakeClient:
        class chat:
            completions = fake_completions

    service = OpenRouterLLMService(
        config=OpenRouterConfig(api_key="test-key"),
        client=FakeClient(),
    )
    env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")
    observation = env.reset()
    decision = service.generate_policy_decision("easy_deadline_extraction", observation)
    assert decision.action.action_type == "read_email"
