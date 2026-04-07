from __future__ import annotations

import sqlite3
from typing import Any


class MockWorkspace:
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

    def get_unread_emails(self):
        return self.connection.execute(
            """
            SELECT id, sender, subject, substr(body, 1, 80) AS snippet
            FROM Emails
            WHERE is_read = 0 AND is_archived = 0
            ORDER BY timestamp ASC
            """
        ).fetchall()

    def read_email(self, email_id):
        self.connection.execute("UPDATE Emails SET is_read = 1 WHERE id = ?", (email_id,))
        self.connection.commit()
        return self.connection.execute("SELECT * FROM Emails WHERE id = ?", (email_id,)).fetchone()

    def send_reply(self, email_id, text):
        return "reply drafted"

    def forward_email(self, email_id, recipient, note=None):
        return f"forwarded to {recipient}"

    def create_todo(self, task_name, deadline_date, context):
        self.connection.execute(
            "INSERT INTO Todos (task_name, deadline_date, context) VALUES (?, ?, ?)",
            (task_name, deadline_date, context),
        )
        self.connection.commit()
        return "todo created"

    def archive_email(self, email_id):
        self.connection.execute("UPDATE Emails SET is_archived = 1 WHERE id = ?", (email_id,))
        self.connection.commit()
        return "email archived"

    def search_documents(self, query):
        return self.connection.execute(
            "SELECT * FROM Files WHERE content_text LIKE ?",
            (f"%{query}%",),
        ).fetchall()

    def list_todos(self):
        return self.connection.execute(
            "SELECT task_name FROM Todos"
        ).fetchall()

    def list_recent_actions(self, limit=6):
        return []