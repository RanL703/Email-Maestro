from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import AssistantAction, PolicyDecision, TaskReward, WorkspaceObservation


class AssistantPolicy(Protocol):
    def choose_action(self, task_name: str, observation: WorkspaceObservation) -> PolicyDecision:
        ...


@dataclass(frozen=True)
class EpisodeStepRecord:
    step_index: int
    reasoning: str
    action: dict[str, object]
    observation: dict[str, object]
    snapshot: dict[str, object]
    reward: dict[str, object]
    status: str


@dataclass(frozen=True)
class EpisodeTrace:
    task_name: str
    policy_name: str
    steps: list[EpisodeStepRecord]
    final_score: float
    completed: bool
    termination_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "policy_name": self.policy_name,
            "steps": [asdict(step) for step in self.steps],
            "final_score": self.final_score,
            "completed": self.completed,
            "termination_reason": self.termination_reason,
        }


class EpisodeRunner:
    def __init__(self, policy: AssistantPolicy, max_steps: int = 12) -> None:
        self.policy = policy
        self.max_steps = max_steps

    def initialize(self, task_name: str) -> tuple[ExecutiveAssistantEnv, WorkspaceObservation]:
        """Load environment state and generate the initial observation."""
        env = ExecutiveAssistantEnv(task_name=task_name)
        env.max_steps = self.max_steps
        observation = env.reset()
        return env, observation

    def advance(
        self,
        task_name: str,
        env: ExecutiveAssistantEnv,
        observation: WorkspaceObservation,
    ) -> tuple[PolicyDecision, WorkspaceObservation, TaskReward, EpisodeStepRecord]:
        """
        Execute one full agent workflow step:
        1. Send observation to policy
        2. Receive structured action
        3. Execute action in workspace
        4. Update state and capture the resulting trace record
        """
        decision = self.policy.choose_action(task_name, observation)
        next_observation, reward = env.step(decision.action)
        record = EpisodeStepRecord(
            step_index=env.step_count,
            reasoning=decision.reasoning,
            action=decision.action.model_dump(),
            observation=next_observation.model_dump(),
            snapshot=env.workspace.snapshot(),
            reward=reward.model_dump(),
            status=next_observation.last_action_status,
        )
        return decision, next_observation, reward, record

    def run(self, task_name: str) -> EpisodeTrace:
        """
        Agent workflow loop:
        1. Load environment state
        2. Generate observation
        3. Send to policy/LLM
        4. Receive structured action
        5. Execute action in workspace
        6. Update state
        7. Repeat until task complete
        """
        env, observation = self.initialize(task_name)
        steps: list[EpisodeStepRecord] = []

        while True:
            _, observation, reward, record = self.advance(task_name, env, observation)
            steps.append(record)
            if reward.is_done:
                return EpisodeTrace(
                    task_name=task_name,
                    policy_name=type(self.policy).__name__,
                    steps=steps,
                    final_score=reward.total_score,
                    completed=reward.total_score >= 1.0,
                    termination_reason=reward.reasoning,
                )


def run_policy_suite(
    policy: AssistantPolicy,
    task_names: list[str],
    max_steps: int = 12,
) -> dict[str, EpisodeTrace]:
    runner = EpisodeRunner(policy=policy, max_steps=max_steps)
    return {task_name: runner.run(task_name) for task_name in task_names}


def export_traces_jsonl(traces: list[EpisodeTrace], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(trace.to_dict()) for trace in traces]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path
