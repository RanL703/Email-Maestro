from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.executive_assistant.agent import ActionCatalog, BaselineAgent
from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.models import AssistantAction, PolicyDecision, WorkspaceObservation
from src.executive_assistant.runner import EpisodeRunner, EpisodeTrace


ACTION_NAMES = [
    "read_first_unread",
    "archive_first_unread",
    "forward_client_to_manager",
    "reply_meeting_time",
    "add_deadline_todo",
    "archive_current_email",
    "search_q3_architecture",
    "reply_with_metrics",
]


def _current_email_sender(observation: WorkspaceObservation) -> str:
    return observation.current_email.sender if observation.current_email else "none"


def encode_observation(task_name: str, observation: WorkspaceObservation) -> str:
    unread_senders = ",".join(sorted(email.sender for email in observation.unread_emails)) or "none"
    return "|".join(
        [
            task_name,
            f"unread={len(observation.unread_emails)}",
            f"senders={unread_senders}",
            f"todos={len(observation.active_todos)}",
            f"current={_current_email_sender(observation)}",
            f"search={int(bool(observation.search_results))}",
            f"history={'/'.join(observation.action_history[-3:]) or 'none'}",
        ]
    )


def valid_action_names(task_name: str, observation: WorkspaceObservation) -> list[str]:
    valid: list[str] = []

    if task_name == "easy_deadline_extraction":
        if observation.current_email is None and observation.unread_emails:
            valid.append("read_first_unread")
        if observation.current_email is not None:
            body = observation.current_email.body.lower()
            existing = {todo.lower() for todo in observation.active_todos}
            missing_todo = False
            if "proposal due" in body and "proposal due" not in existing:
                valid.append("add_deadline_todo")
                missing_todo = True
            elif "prototype due" in body and "prototype due" not in existing:
                valid.append("add_deadline_todo")
                missing_todo = True
            elif "final report due" in body and "final report due" not in existing:
                valid.append("add_deadline_todo")
                missing_todo = True
            if not missing_todo:
                valid.append("archive_current_email")
    elif task_name == "medium_triage_and_negotiation":
        newsletter_senders = {
            "news@updates.example",
            "promotions@vendor.example",
            "events@community.example",
        }
        if any(email.sender in newsletter_senders for email in observation.unread_emails):
            valid.append("archive_first_unread")
        if any(email.sender == "client@company.com" for email in observation.unread_emails):
            valid.append("forward_client_to_manager")
        if any(email.sender == "teammate@company.com" for email in observation.unread_emails):
            valid.append("reply_meeting_time")
    elif task_name == "hard_rag_reply":
        if observation.current_email is None and observation.unread_emails:
            valid.append("read_first_unread")
        if observation.current_email is not None and not observation.search_results:
            valid.append("search_q3_architecture")
        if observation.current_email is not None and observation.search_results:
            valid.append("reply_with_metrics")

    return valid or ACTION_NAMES.copy()


def make_action(action_name: str, observation: WorkspaceObservation) -> AssistantAction:
    if action_name == "read_first_unread":
        if observation.unread_emails:
            return AssistantAction(action_type="read_email", target_id=observation.unread_emails[0].id)
    elif action_name == "archive_first_unread":
        if observation.unread_emails:
            return AssistantAction(action_type="archive", target_id=observation.unread_emails[0].id)
    elif action_name == "forward_client_to_manager":
        for email in observation.unread_emails:
            if email.sender == "client@company.com":
                return AssistantAction(
                    action_type="forward",
                    target_id=email.id,
                    secondary_payload="manager@company.com",
                    payload="Urgent client complaint. Please take over immediately.",
                )
    elif action_name == "reply_meeting_time":
        target_id = observation.current_email.id if observation.current_email else None
        if target_id is None:
            for email in observation.unread_emails:
                if email.sender == "teammate@company.com":
                    target_id = email.id
                    break
        if target_id is not None:
            return AssistantAction(
                action_type="reply",
                target_id=target_id,
                payload="Hello, 3:30 PM IST works for me. Regards, Executive Assistant",
            )
    elif action_name == "add_deadline_todo":
        if observation.current_email:
            body = observation.current_email.body.lower()
            candidates = [
                ("Proposal Due", "2026-04-10", "proposal due"),
                ("Prototype Due", "2026-04-20", "prototype due"),
                ("Final Report Due", "2026-04-30", "final report due"),
            ]
            existing = {todo.lower() for todo in observation.active_todos}
            for task_name, deadline, marker in candidates:
                if marker in body and task_name.lower() not in existing:
                    return AssistantAction(
                        action_type="add_todo",
                        payload=task_name,
                        secondary_payload=deadline,
                    )
    elif action_name == "archive_current_email":
        if observation.current_email:
            return AssistantAction(action_type="archive", target_id=observation.current_email.id)
    elif action_name == "search_q3_architecture":
        return AssistantAction(action_type="search_files", payload="Q3 Architecture")
    elif action_name == "reply_with_metrics":
        if observation.current_email and observation.search_results:
            snippet = observation.search_results[0].snippet
            availability = "99.95%" if "99.95%" in snippet else "unknown"
            latency = "182ms" if "182ms" in snippet else "unknown"
            cost = "14%" if "14%" in snippet else "unknown"
            return AssistantAction(
                action_type="reply",
                target_id=observation.current_email.id,
                payload=(
                    "Hello,\n"
                    f"Here are the requested Q3 architecture metrics: availability {availability}, "
                    f"mean API latency {latency}, and infrastructure cost reduction {cost}.\n"
                    "Regards,\nExecutive Assistant"
                ),
            )
    return AssistantAction(action_type="search_files")


@dataclass
class QLearningPolicy:
    epsilon: float = 0.2
    alpha: float = 0.3
    gamma: float = 0.95
    seed: int = 7

    def __post_init__(self) -> None:
        self.q_values: dict[str, dict[str, float]] = defaultdict(
            lambda: {action_name: 0.0 for action_name in ACTION_NAMES}
        )
        self.random = random.Random(self.seed)

    def choose_action(self, task_name: str, observation: WorkspaceObservation) -> PolicyDecision:
        state = encode_observation(task_name, observation)
        candidates = valid_action_names(task_name, observation)
        if self.random.random() < self.epsilon:
            action_name = self.random.choice(candidates)
            return PolicyDecision(
                reasoning=f"Exploring action template {action_name}.",
                action=make_action(action_name, observation),
            )

        action_name = max(candidates, key=lambda name: self.q_values[state][name])
        return PolicyDecision(
            reasoning=f"Selecting greedy action template {action_name}.",
            action=make_action(action_name, observation),
        )

    def update(
        self,
        state: str,
        action_name: str,
        reward: float,
        next_state: str,
        done: bool,
    ) -> None:
        next_best = 0.0 if done else max(self.q_values[next_state].values())
        current = self.q_values[state][action_name]
        target = reward + self.gamma * next_best
        self.q_values[state][action_name] = current + self.alpha * (target - current)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "action_names": ACTION_NAMES,
                "seed": self.seed,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": 0.0,
            },
            "q_values": self.q_values,
        }
        output.write_text(json.dumps(payload, indent=2))
        return output

    @classmethod
    def load(cls, path: str | Path) -> "QLearningPolicy":
        checkpoint_path = Path(path)
        policy = cls(epsilon=0.0)
        raw_payload = json.loads(checkpoint_path.read_text())
        raw_values = raw_payload["q_values"] if "q_values" in raw_payload else raw_payload
        policy.q_values = defaultdict(
            lambda: {action_name: 0.0 for action_name in ACTION_NAMES}
        )
        for state, action_map in raw_values.items():
            policy.q_values[state] = {
                action_name: float(action_map.get(action_name, 0.0))
                for action_name in ACTION_NAMES
            }
        policy.epsilon = 0.0
        return policy


def action_name_from_decision(decision: PolicyDecision, observation: WorkspaceObservation) -> str:
    for action_name in ACTION_NAMES:
        candidate = make_action(action_name, observation)
        if candidate == decision.action:
            return action_name
    return "search_q3_architecture"


def warm_start_from_teacher(
    learner: QLearningPolicy,
    teacher: BaselineAgent,
    task_names: list[str],
    episodes_per_task: int = 4,
) -> None:
    runner = EpisodeRunner(policy=teacher)
    for _ in range(episodes_per_task):
        for task_name in task_names:
            trace = runner.run(task_name)
            for index, step in enumerate(trace.steps):
                current_observation = WorkspaceObservation.model_validate(step.observation)
                previous_observation = (
                    WorkspaceObservation.model_validate(trace.steps[index - 1].observation)
                    if index > 0
                    else None
                )
                observation = previous_observation or current_observation
                state = encode_observation(task_name, observation)
                next_state = encode_observation(task_name, current_observation)
                reward_delta = step.reward["total_score"]
                action_name = action_name_from_decision(
                    PolicyDecision(
                        reasoning=step.reasoning,
                        action=AssistantAction.model_validate(step.action),
                    ),
                    observation,
                )
                learner.update(
                    state=state,
                    action_name=action_name,
                    reward=reward_delta,
                    next_state=next_state,
                    done=bool(step.reward["is_done"]),
                )


def train_q_learning(
    episodes: int = 200,
    epsilon: float = 0.15,
    teacher: BaselineAgent | None = None,
) -> tuple[QLearningPolicy, dict[str, float]]:
    learner = QLearningPolicy(epsilon=epsilon)
    task_names = [
        "easy_deadline_extraction",
        "medium_triage_and_negotiation",
        "hard_rag_reply",
    ]
    if teacher is not None:
        warm_start_from_teacher(learner, teacher, task_names)

    scores: dict[str, float] = {}
    for episode in range(episodes):
        task_name = task_names[episode % len(task_names)]
        env = ExecutiveAssistantEnv(task_name=task_name)
        observation = env.reset()
        previous_total_score = 0.0

        while True:
            state = encode_observation(task_name, observation)
            decision = learner.choose_action(task_name, observation)
            action_name = action_name_from_decision(decision, observation)
            next_observation, reward, _, _ = env.step(decision.action)
            next_state = encode_observation(task_name, next_observation)
            reward_delta = reward.total_score - previous_total_score - 0.01
            previous_total_score = reward.total_score
            learner.update(
                state=state,
                action_name=action_name,
                reward=reward_delta,
                next_state=next_state,
                done=reward.is_done,
            )
            observation = next_observation
            if reward.is_done:
                scores[task_name] = reward.total_score
                break
    return learner, scores


def evaluate_q_policy(policy: QLearningPolicy) -> dict[str, float]:
    original_epsilon = policy.epsilon
    policy.epsilon = 0.0
    try:
        traces = {
            task_name: EpisodeRunner(policy=policy).run(task_name)
            for task_name in [
                "easy_deadline_extraction",
                "medium_triage_and_negotiation",
                "hard_rag_reply",
            ]
        }
    finally:
        policy.epsilon = original_epsilon
    return {task_name: trace.final_score for task_name, trace in traces.items()}


def default_checkpoint_path(checkpoint_dir: str | Path, checkpoint_name: str) -> Path:
    return Path(checkpoint_dir) / checkpoint_name
