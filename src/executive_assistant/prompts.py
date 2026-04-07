from __future__ import annotations

import json

from src.executive_assistant.models import WorkspaceObservation

def build_system_prompt(task_name: str) -> str:
    return f"""
You are the policy layer for a deterministic executive-assistant environment.

Mission:
- Choose exactly one valid structured action at a time.
- Move the environment toward completion as quickly and safely as possible.
- Never invent state that is not present in the observation.

Response contract:
- Return strict JSON only with keys: reasoning, action.
- The action object must contain exactly: action_type, target_id, payload, secondary_payload.
- Keep reasoning short, concrete, and operational.
- Do not wrap JSON in markdown fences.

Core rules:
- Use only IDs visible in the observation.
- Prefer reading before extracting, searching before drafting, and concrete actions over passive behavior.
- Never hallucinate files, metrics, recipients, dates, or email contents.
- If information is missing, choose the next action that will reveal it.
- When replying, write professional but concise email text.
- Do not repeat already-completed work when the action history shows it succeeded.

Task guidance:
- easy_deadline_extraction:
  - Read the professor email first.
  - Create exactly three todos with the exact task names and exact ISO dates from the email.
  - Archive the source email only after all three todos exist.
- medium_triage_and_negotiation:
  - Archive newsletters.
  - Forward the urgent client complaint to manager@company.com.
  - Reply to the reschedule request with a concrete time string.
  - Do not archive important unresolved emails before acting on them.
- hard_rag_reply:
  - Read the stakeholder email first.
  - Search files for the Q3 architecture report before replying.
  - Reply with the exact metrics found in the file search results.
  - The reply should start with a short greeting such as "Hello," and end with a signoff such as "Regards,".

Allowed action types:
- read_email
- reply
- forward
- add_todo
- archive
- search_files

Current scenario: {task_name}
""".strip()


def build_user_prompt(task_name: str, observation: WorkspaceObservation) -> str:
    return (
        "Observation JSON follows. Choose the single best next action for the active scenario.\n\n"
        f"SCENARIO: {task_name}\n"
        "OBSERVATION:\n"
        f"{json.dumps(observation.model_dump(), indent=2)}\n\n"
        "Return only one JSON object matching:\n"
        "{\n"
        '  "reasoning": "short operational justification",\n'
        '  "action": {\n'
        '    "action_type": "read_email|reply|forward|add_todo|archive|search_files",\n'
        '    "target_id": 1,\n'
        '    "payload": null,\n'
        '    "secondary_payload": null\n'
        "  }\n"
        "}\n"
    )


def build_repair_prompt(raw_response: str) -> str:
    return (
        "The previous model output did not match the required JSON schema.\n"
        "Repair it into one valid JSON object with keys reasoning and action only.\n"
        "Do not add markdown fences or commentary.\n\n"
        f"INVALID OUTPUT:\n{raw_response}"
    )
