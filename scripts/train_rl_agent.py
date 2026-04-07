from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_assistant.agent import BaselineAgent
from src.executive_assistant.config import TrainingRuntimeConfig, load_env_file
from src.executive_assistant.training import evaluate_q_policy, train_q_learning


def main() -> None:
    load_env_file(TrainingRuntimeConfig().env_file)
    parser = argparse.ArgumentParser(description="Train a tabular RL policy for seeded tasks.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/q_policy.json")
    parser.add_argument("--no-teacher", action="store_true")
    args = parser.parse_args()

    teacher = None if args.no_teacher else BaselineAgent()
    policy, training_scores = train_q_learning(
        episodes=args.episodes,
        epsilon=args.epsilon,
        teacher=teacher,
    )
    checkpoint_path = policy.save(args.checkpoint)
    evaluation = evaluate_q_policy(policy)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "training_scores": training_scores,
                "evaluation": evaluation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
