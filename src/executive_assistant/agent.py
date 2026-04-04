from __future__ import annotations

from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import AssistantAction


class BaselineAgent:
    """Stub baseline agent for notebook and local integration work."""

    def __init__(self, model_name: str = "openai-structured-output-placeholder") -> None:
        self.model_name = model_name

    def choose_action(self, observation_text: str) -> AssistantAction:
        raise NotImplementedError("LLM-backed policy will be implemented after environment validation.")


def run_episode(task_name: str) -> None:
    env = ExecutiveAssistantEnv(task_name=task_name)
    observation = env.reset()
    print(observation.model_dump_json(indent=2))
