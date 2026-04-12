from __future__ import annotations

import uvicorn

from app import app
from src.executive_assistant.config import AppRuntimeConfig


def main() -> None:
    runtime = AppRuntimeConfig()
    uvicorn.run(app, host=runtime.host, port=runtime.port)
