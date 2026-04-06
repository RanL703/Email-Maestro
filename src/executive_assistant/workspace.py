from __future__ import annotations

import sqlite3
from typing import Any


class MockWorkspace:
    """Deterministic SQLite-backed workspace for inbox, todos, and local files."""

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE Emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                is_archived INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE Todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                deadline_date TEXT,
                context TEXT NOT NULL
            );

            CREATE TABLE Files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content_text TEXT NOT NULL
            );

            CREATE TABLE ActionLog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                target_id INTEGER,
                payload TEXT,
                secondary_payload TEXT,
                status TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def seed(self, emails: list[dict[str, Any]], files: list[dict[str, Any]]) -> None:
        self.connection.executemany(
            """
            INSERT INTO Emails (sender, recipient, subject, body, timestamp)
            VALUES (:sender, :recipient, :subject, :body, :timestamp)
            """,
            emails,
        )
        self.connection.executemany(
            """
            INSERT INTO Files (filename, content_text)
            VALUES (:filename, :content_text)
            """,
            files,
        )
        self.connection.commit()

    def get_unread_emails(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT id, sender, subject, substr(body, 1, 80) AS snippet
            FROM Emails
            WHERE is_read = 0 AND is_archived = 0
            ORDER BY timestamp ASC
            """
        ).fetchall()

    def read_email(self, email_id: int) -> sqlite3.Row | None:
        self.connection.execute("UPDATE Emails SET is_read = 1 WHERE id = ?", (email_id,))
        self.connection.commit()
        row = self.connection.execute("SELECT * FROM Emails WHERE id = ?", (email_id,)).fetchone()
        status = "email read" if row else "email not found"
        self.log_action("read_email", email_id, None, None, status)
        return row

    def send_reply(self, email_id: int, text: str) -> str:
        row = self.connection.execute("SELECT id FROM Emails WHERE id = ?", (email_id,)).fetchone()
        if row is None:
            self.log_action("reply", email_id, text, None, "reply failed: email not found")
            return "reply failed: email not found"
        self.log_action("reply", email_id, text, None, "reply drafted")
        return "reply drafted"

    def forward_email(self, email_id: int, recipient: str, note: str | None = None) -> str:
        row = self.connection.execute("SELECT id FROM Emails WHERE id = ?", (email_id,)).fetchone()
        if row is None:
            self.log_action(
                "forward", email_id, note, recipient, "forward failed: email not found"
            )
            return "forward failed: email not found"
        self.log_action("forward", email_id, note, recipient, f"forwarded to {recipient}")
        return f"forwarded to {recipient}"

    def create_todo(self, task_name: str, deadline_date: str | None, context: str) -> str:
        self.connection.execute(
            "INSERT INTO Todos (task_name, deadline_date, context) VALUES (?, ?, ?)",
            (task_name, deadline_date, context),
        )
        self.connection.commit()
        self.log_action("add_todo", None, task_name, deadline_date, "todo created")
        return "todo created"

    def archive_email(self, email_id: int) -> str:
        row = self.connection.execute("SELECT id FROM Emails WHERE id = ?", (email_id,)).fetchone()
        if row is None:
            self.log_action("archive", email_id, None, None, "archive failed: email not found")
            return "archive failed: email not found"
        self.connection.execute("UPDATE Emails SET is_archived = 1 WHERE id = ?", (email_id,))
        self.connection.commit()
        self.log_action("archive", email_id, None, None, "email archived")
        return "email archived"

    def search_documents(self, query: str) -> list[sqlite3.Row]:
        results = self.connection.execute(
            """
            SELECT * FROM Files
            WHERE filename LIKE ? OR content_text LIKE ?
            ORDER BY id ASC
            """,
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
        self.log_action("search_files", None, query, None, f"{len(results)} file(s) matched")
        return results

    def list_todos(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT id, task_name, deadline_date, context FROM Todos ORDER BY id ASC"
        ).fetchall()

    def list_recent_actions(self, limit: int = 8) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT id, action_type, target_id, payload, secondary_payload, status
            FROM ActionLog
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def log_action(
        self,
        action_type: str,
        target_id: int | None,
        payload: str | None,
        secondary_payload: str | None,
        status: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO ActionLog (action_type, target_id, payload, secondary_payload, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (action_type, target_id, payload, secondary_payload, status),
        )
        self.connection.commit()

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "emails": [dict(row) for row in self.connection.execute("SELECT * FROM Emails ORDER BY id ASC")],
            "todos": [dict(row) for row in self.connection.execute("SELECT * FROM Todos ORDER BY id ASC")],
            "files": [dict(row) for row in self.connection.execute("SELECT * FROM Files ORDER BY id ASC")],
            "action_log": [
                dict(row) for row in self.connection.execute("SELECT * FROM ActionLog ORDER BY id ASC")
            ],
        }
