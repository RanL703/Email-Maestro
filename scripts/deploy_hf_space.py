from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_assistant.deployment import (
    DEFAULT_SPACE_TITLE,
    HFSpaceDeployConfig,
    parse_hf_usernames,
    stage_space_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or update a Hugging Face Space from this repository in one command."
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_SPACE_REPO", "").strip(),
        help="Target Space repo in owner/name form. Defaults to HF_SPACE_REPO.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", "").strip(),
        help="Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--title",
        default=os.environ.get("HF_SPACE_TITLE", DEFAULT_SPACE_TITLE),
        help="Space title used in the generated HF README.",
    )
    parser.add_argument(
        "--team-name",
        default=os.environ.get("HF_SPACE_TEAM_NAME", "Team Epsilon"),
        help="Team name shown in the generated HF README.",
    )
    parser.add_argument(
        "--hf-usernames",
        default=os.environ.get(
            "HF_SPACE_TEAM_USERNAMES",
            "flickinshots,ShreyaKhatik,itsayushdey",
        ),
        help="Comma-separated HF usernames for the HF README placeholders.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=os.environ.get("HF_SPACE_CHECKPOINT_NAME", "q_policy_notebook.json"),
        help="Checkpoint filename staged into artifacts/checkpoints/ for RL replay.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        help="Optional secret to set on the Space during deployment.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        help="OpenAI-compatible API base URL for inference.py. Defaults to OpenRouter.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "google/gemma-4-31b-it").strip(),
        help="OpenRouter model id for inference.py. Defaults to Gemma 4.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=os.environ.get("HF_SPACE_PRIVATE", "").strip().lower() == "true",
        help="Create or keep the Space private.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip bundling the RL checkpoint.",
    )
    parser.add_argument(
        "--keep-stage-dir",
        default="",
        help="Optional local folder where the prepared Space bundle should be copied after upload.",
    )
    return parser


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for deployment. Install the training environment "
            "or run `python -m pip install huggingface_hub` first."
        ) from exc
    return HfApi


def maybe_set_space_secret(api, repo_id: str, key: str, value: str) -> str:
    if not value.strip():
        return f"Skipped secret {key} because no value was provided."
    add_secret = getattr(api, "add_space_secret", None)
    if add_secret is None:
        return f"Upload succeeded, but this huggingface_hub version cannot set {key} automatically."
    add_secret(repo_id=repo_id, key=key, value=value)
    return f"Set Space secret {key}."


def maybe_set_space_variable(api, repo_id: str, key: str, value: str) -> str:
    add_variable = getattr(api, "add_space_variable", None)
    if add_variable is None:
        return f"Upload succeeded, but this huggingface_hub version cannot set variable {key} automatically."
    add_variable(repo_id=repo_id, key=key, value=value)
    return f"Set Space variable {key}={value}."


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("A Space repo id is required. Pass --repo-id or set HF_SPACE_REPO.")
    if "/" not in args.repo_id:
        parser.error("Space repo id must be in owner/name form.")
    if not args.token:
        parser.error("A Hugging Face token is required. Pass --token or set HF_TOKEN.")

    config = HFSpaceDeployConfig(
        repo_id=args.repo_id,
        title=args.title,
        team_name=args.team_name,
        hf_usernames=parse_hf_usernames(args.hf_usernames),
        checkpoint_name=args.checkpoint_name,
        private=args.private,
        include_checkpoint=not args.skip_checkpoint,
    )

    HfApi = require_huggingface_hub()
    api = HfApi(token=args.token)

    with tempfile.TemporaryDirectory(prefix="hf-space-stage-") as tmp_dir:
        stage_dir = Path(tmp_dir)
        checkpoint_path = stage_space_bundle(config, stage_dir)

        api.create_repo(
            repo_id=config.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=config.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=config.repo_id,
            repo_type="space",
            commit_message="Deploy Project Epsilon Space bundle",
            delete_patterns=["*", "**/*"],
        )

        messages = [
            f"Uploaded Space bundle to {config.space_url}",
            f"App URL: {config.app_url}",
        ]
        if checkpoint_path is not None:
            messages.append(f"Bundled RL checkpoint: {checkpoint_path.relative_to(stage_dir)}")
        messages.append(maybe_set_space_secret(api, config.repo_id, "OPENROUTER_API_KEY", args.openrouter_api_key))
        messages.append(maybe_set_space_secret(api, config.repo_id, "OPENAI_API_KEY", args.openrouter_api_key))
        messages.append(maybe_set_space_variable(api, config.repo_id, "API_BASE_URL", args.api_base_url))
        messages.append(maybe_set_space_variable(api, config.repo_id, "MODEL_NAME", args.model_name))
        messages.append(maybe_set_space_variable(api, config.repo_id, "OPENROUTER_APP_NAME", config.title))
        messages.append(maybe_set_space_variable(api, config.repo_id, "OPENROUTER_SITE_URL", config.app_url))

        if args.keep_stage_dir:
            target_dir = Path(args.keep_stage_dir).resolve()
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(stage_dir, target_dir)
            messages.append(f"Saved staged bundle to {target_dir}")

    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
