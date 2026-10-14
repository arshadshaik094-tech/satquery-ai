"""
Provider-agnostic LLM router.

Reads GROQ_API_KEY / OPENAI_API_KEY from settings (.env). Whichever is
present is used; if LLM_PROVIDER_PRIORITY="auto", Groq is preferred first
(cheaper/faster for a hackathon demo) and OpenAI is the fallback. If
NEITHER key is set, `chat()` raises LLMUnavailableError and the
orchestrator falls back to a deterministic, keyword-based router
(core/keyword_router.py) so the app still functions end-to-end without any
paid API — just with less flexible query understanding.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger("satquery.llm")


class LLMUnavailableError(RuntimeError):
    pass


class LLMRouter:
    def __init__(self):
        self.settings = get_settings()

    def _provider_order(self) -> list[str]:
        p = self.settings.LLM_PROVIDER_PRIORITY
        if p == "groq":
            return ["groq"]
        if p == "openai":
            return ["openai"]
        order = []
        if self.settings.GROQ_API_KEY:
            order.append("groq")
        if self.settings.OPENAI_API_KEY:
            order.append("openai")
        return order

    def is_available(self) -> bool:
        return len(self._provider_order()) > 0

    def chat_with_tools(
        self, system_prompt: str, user_message: str, tools: list[dict]
    ) -> dict[str, Any]:
        """Returns {"tool_calls": [...], "content": str}. Raises
        LLMUnavailableError if no provider is configured."""
        for provider in self._provider_order():
            try:
                if provider == "groq":
                    return self._call_groq(system_prompt, user_message, tools)
                if provider == "openai":
                    return self._call_openai(system_prompt, user_message, tools)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider %s failed (%s), trying next.", provider, exc)
                continue
        raise LLMUnavailableError(
            "No LLM provider configured or all configured providers failed. "
            "Set GROQ_API_KEY or OPENAI_API_KEY in .env."
        )

    def _call_groq(self, system_prompt, user_message, tools) -> dict:
        from groq import Groq

        client = Groq(api_key=self.settings.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=self.settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                })
        return {"tool_calls": tool_calls, "content": msg.content or ""}

    def _call_openai(self, system_prompt, user_message, tools) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=self.settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                })
        return {"tool_calls": tool_calls, "content": msg.content or ""}

    def summarize(self, system_prompt: str, user_message: str) -> str:
        """Plain (no tool-calling) completion — used to turn raw model
        outputs into a natural-language final answer."""
        for provider in self._provider_order():
            try:
                if provider == "groq":
                    from groq import Groq
                    client = Groq(api_key=self.settings.GROQ_API_KEY)
                    resp = client.chat.completions.create(
                        model=self.settings.GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.3,
                    )
                    return resp.choices[0].message.content or ""
                if provider == "openai":
                    from openai import OpenAI
                    client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
                    resp = client.chat.completions.create(
                        model=self.settings.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.3,
                    )
                    return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider %s summarize failed (%s)", provider, exc)
                continue
        raise LLMUnavailableError("No LLM provider available for summarization.")


_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
