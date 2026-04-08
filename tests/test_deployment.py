from pathlib import Path

from src.executive_assistant.deployment import (
    HFSpaceDeployConfig,
    parse_hf_usernames,
    render_space_readme,
    stage_space_bundle,
)


def test_parse_hf_usernames_strips_at_signs() -> None:
    usernames = parse_hf_usernames("@alice, bob , ,@carol")
    assert usernames == ("alice", "bob", "carol")


def test_render_space_readme_includes_team_epsilon_roster() -> None:
    config = HFSpaceDeployConfig(
        repo_id="Flickinshots/EmailMaestro",
        team_name="Team Epsilon",
        hf_usernames=("flickinshots", "ShreyaKhatik", "itsayushdey"),
    )
    rendered = render_space_readme(config)
    assert "Team Epsilon" in rendered
    assert "@flickinshots" in rendered
    assert "Team lead and primary Space owner" in rendered
    assert "sdk: docker" in rendered
    assert "OpenEnv Scaler x Meta x PyTorch Hack" in rendered


def test_stage_space_bundle_writes_hf_readme_and_checkpoint(tmp_path: Path) -> None:
    config = HFSpaceDeployConfig(
        repo_id="placeholder/project-epsilon-executive-assistant",
        hf_usernames=("HF_USERNAME_1",),
    )
    checkpoint_path = stage_space_bundle(config, tmp_path)
    assert checkpoint_path is not None
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "app.py").exists()
    assert (tmp_path / "src" / "executive_assistant" / "env.py").exists()
    assert (tmp_path / "artifacts" / "checkpoints" / config.checkpoint_name).exists()
    assert not (tmp_path / ".env.app").exists()
    assert not (tmp_path / ".env.training").exists()
    assert not (tmp_path / ".env.hf.space.example").exists()
