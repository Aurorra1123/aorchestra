"""Prompt helpers for parallel delegate mode."""
from __future__ import annotations

from typing import List, Optional


def _parallel_suffix(sub_models: Optional[List[str]] = None) -> str:
    model_choices = sub_models or ["<model_name>"]
    return f"""==== PARALLEL DELEGATION MODE (OPTIONAL) ====
You may return MULTIPLE actions in one response when work can be split.
Only use this when the subtasks are independent.

Single-action format (always valid):
{{
  "action": "delegate_task",
  "reasoning": "why this subtask is needed",
  "params": {{
    "task_instruction": "clear subtask instruction",
    "context": "optional hints",
    "model": "one of {model_choices}"
  }}
}}

Parallel multi-action format:
{{
  "actions": [
    {{
      "action": "delegate_task",
      "reasoning": "subtask 1 reasoning",
      "params": {{
        "task_instruction": "subtask 1",
        "context": "optional hints",
        "model": "one of {model_choices}"
      }}
    }},
    {{
      "action": "delegate_task",
      "reasoning": "subtask 2 reasoning",
      "params": {{
        "task_instruction": "subtask 2",
        "context": "optional hints",
        "model": "one of {model_choices}"
      }}
    }}
  ]
}}""".strip()


def append_parallel_delegate_instructions(
    base_prompt: str,
    sub_models: Optional[List[str]] = None,
) -> str:
    """Append parallel-delegation instructions to an existing prompt."""
    suffix = _parallel_suffix(sub_models)
    if not base_prompt:
        return suffix
    return f"{base_prompt.rstrip()}\n\n{suffix}"
