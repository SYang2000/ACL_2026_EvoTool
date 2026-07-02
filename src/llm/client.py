"""Unified LLM client against an OpenAI-compatible endpoint (vLLM, OpenAI, etc.).

Includes lightweight token accounting so runs can report the token-efficiency
comparison (paper Figure 3b).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from src.config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        from openai import OpenAI

        self.config = config
        base_url = config.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.total_tokens = 0  # cumulative prompt+completion tokens this run

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            # Context-length overflow (e.g. long tau-bench tool/DB prompts): truncate
            # the longest message and retry once, then fall back to "" so one oversized
            # prompt degrades that single step instead of crashing the whole run.
            if "maximum context length" in str(e) or "context length" in str(e).lower():
                kwargs["messages"] = _truncate_messages(messages)
                kwargs["max_tokens"] = min(kwargs.get("max_tokens") or 512, 512)
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                except Exception:  # noqa: BLE001
                    return ""
            else:
                raise
        if getattr(resp, "usage", None):
            self.total_tokens += resp.usage.total_tokens or 0
        return resp.choices[0].message.content or ""

    def generate_json(self, messages: list[dict[str, str]]) -> dict:
        text = self.generate(messages, json_mode=True)
        return _parse_json(text)


def _truncate_messages(messages: list[dict], char_budget: int = 24000) -> list[dict]:
    """Shrink messages to fit the model context when a prompt overflows. Keeps the
    system message intact and trims the MIDDLE of the longest (usually user) message,
    preserving its head (task/tools) and tail (recent trajectory) — the parts that
    matter most for the next decision. ~24k chars ≈ ~6k tokens, safely under 8192."""
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= char_budget:
        return messages
    longest = max(range(len(messages)), key=lambda i: len(messages[i].get("content", "")))
    overflow = total - char_budget
    content = messages[longest].get("content", "")
    keep = max(2000, len(content) - overflow - 200)
    if keep < len(content):
        head = content[: keep // 2]
        tail = content[-keep // 2:]
        content = head + "\n...[truncated to fit context]...\n" + tail
    out = list(messages)
    out[longest] = {**messages[longest], "content": content}
    return out


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
