from src.executive_assistant.workspace import MockWorkspace


def test_workspace_seed_and_snapshot() -> None:
    workspace = MockWorkspace()
    workspace.seed(
        emails=[
            {
                "sender": "a@example.com",
                "recipient": "b@example.com",
                "subject": "Test",
                "body": "Hello",
                "timestamp": "2026-04-04T00:00:00Z",
            }
        ],
        files=[{"filename": "doc.txt", "content_text": "hello world"}],
    )

    snapshot = workspace.snapshot()
    assert len(snapshot["emails"]) == 1
    assert len(snapshot["files"]) == 1
