from __future__ import annotations

import json
from typing import Any

from src.executive_assistant.config import OpenRouterConfig
from src.executive_assistant.models import PolicyDecision, WorkspaceObservation
from src.executive_assistant.prompts import build_repair_prompt, build_system_prompt, build_user_prompt


class LLMServiceError(RuntimeError):
    """Raised when the configured LLM provider cannot produce a valid policy decision."""


class OpenRouterLLMService:
    def __init__(self, config: OpenRouterConfig, client: Any | None = None) -> None:
        self.config = config
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMServiceError(
                "openai package is required for OpenRouter access. Install requirements first."
            ) from exc
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def generate_policy_decision(
        self,
        task_name: str,
        observation: WorkspaceObservation,
    ) -> PolicyDecision:
        raw_message = self._request_json(
            system_prompt=build_system_prompt(task_name),
            user_prompt=build_user_prompt(task_name, observation),
        )
        try:
            payload = json.loads(raw_message)
            return PolicyDecision.model_validate(payload)
        except Exception:
            repaired_message = self._request_json(
                system_prompt="You are a strict JSON repair assistant.",
                user_prompt=build_repair_prompt(raw_message),
            )
            try:
                repaired_payload = json.loads(repaired_message)
                return PolicyDecision.model_validate(repaired_payload)
            except Exception as exc:
                raise LLMServiceError(
                    f"Provider response did not match policy schema after repair: {repaired_message}"
                ) from exc

    def _request_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_headers=self.config.extra_headers(),
            )
        except Exception as exc:  # pragma: no cover - network/provider dependent
            raise LLMServiceError(f"OpenRouter request failed: {exc}") from exc

        message = completion.choices[0].message.content or ""
        if not message.strip():
            raise LLMServiceError("OpenRouter returned an empty response.")
        return message
