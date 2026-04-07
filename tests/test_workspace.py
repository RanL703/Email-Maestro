import threading

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


def test_read_email_is_logged() -> None:
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
        files=[],
    )

    row = workspace.read_email(1)
    assert row is not None
    snapshot = workspace.snapshot()
    assert snapshot["action_log"][0]["action_type"] == "read_email"


def test_workspace_can_be_used_from_worker_thread() -> None:
    workspace = MockWorkspace()
    workspace.seed(
        emails=[
            {
                "sender": "a@example.com",
                "recipient": "b@example.com",
                "subject": "Thread Test",
                "body": "Hello",
                "timestamp": "2026-04-04T00:00:00Z",
            }
        ],
        files=[],
    )
    errors: list[Exception] = []

    def _read_email() -> None:
        try:
            row = workspace.read_email(1)
            assert row is not None
        except Exception as exc:  # pragma: no cover - assertion path is the test failure
            errors.append(exc)

    worker = threading.Thread(target=_read_email)
    worker.start()
    worker.join()

    assert errors == []
