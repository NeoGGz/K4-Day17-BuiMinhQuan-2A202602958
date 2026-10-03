from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                # Fallback to deterministic offline mode on any provider/network error
                return self._reply_offline(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior for baseline agent."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Baseline carries all historical messages in this thread into prompt
        current_context_tokens = sum(
            estimate_tokens(m["content"]) for m in session.messages
        ) + estimate_tokens(message)
        session.prompt_tokens_processed += current_context_tokens

        # If this is a new thread (e.g., recall question thread), baseline knows nothing
        if not session.messages:
            response_text = "Chào bạn! Tôi là trợ lý AI. Trong phiên này tôi chưa có thông tin trước đó."
        else:
            response_text = "Đã nhận thông tin trong phiên hiện tại của bạn."

        resp_tokens = estimate_tokens(response_text)
        session.token_usage += resp_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "token_usage": session.token_usage,
            "prompt_tokens": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        history = [SystemMessage(content="Bạn là một trợ lý AI hữu ích.")]
        for m in session.messages:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            else:
                history.append(AIMessage(content=m["content"]))
        history.append(HumanMessage(content=message))

        current_prompt_tokens = sum(estimate_tokens(str(m.content)) for m in history)
        session.prompt_tokens_processed += current_prompt_tokens

        res = self.langchain_agent.invoke(history)
        response_text = res.content if hasattr(res, "content") else str(res)
        resp_tokens = estimate_tokens(response_text)
        session.token_usage += resp_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "token_usage": session.token_usage,
            "prompt_tokens": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        try:
            if self.config.model.api_key or self.config.model.provider == "ollama":
                self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
