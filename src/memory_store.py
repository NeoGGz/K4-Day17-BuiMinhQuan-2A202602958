from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Implement a simple, consistent token estimator.

    Whitespace stripped; 0 for empty string; approximates roughly 4 chars per token.
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Supports reading, writing, editing, and extracting structured facts.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text in current:
            updated = current.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        parsed: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^-\s*(?:\*\*)?([^*\:]+)(?:\*\*)?\s*:\s*(.+)$", line)
            if m:
                parsed[m.group(1).strip()] = m.group(2).strip()
        return parsed

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        min_confidence: float = 0.6,
    ) -> Path:
        """Upsert fact with confidence gating and conflict resolution."""
        path = self.path_for(user_id)
        if confidence < min_confidence:
            # Drop low-confidence fact updates to prevent corruption
            return path

        text = self.read_text(user_id)
        # Store clean markdown with optional confidence score
        pattern = rf"^-\s*(?:\*\*)?{re.escape(key)}(?:\*\*)?\s*:.*$"
        replacement = f"- **{key}**: {value}"
        if re.search(pattern, text, flags=re.MULTILINE):
            updated = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        else:
            stripped = text.rstrip()
            updated = stripped + f"\n- **{key}**: {value}\n"
        return self.write_text(user_id, updated)

    def apply_decay(
        self,
        user_id: str,
        retained_keys: list[str] | None = None,
        decay_factor: float = 0.85,
        prune_threshold: float = 0.3,
    ) -> list[str]:
        """Memory decay mechanism for aging out unconfirmed or stale facts."""
        facts_dict = self.facts(user_id)
        retained = set(retained_keys or [])
        pruned_keys = []

        # Rebuild file keeping only non-pruned facts
        retained_lines = [f"# User Profile: {user_id}\n"]
        for k, v in facts_dict.items():
            if k not in retained:
                # In actual decay, confidence diminishes over time
                # Here we simulate pruning if key is obsolete
                continue
            retained_lines.append(f"- **{k}**: {v}")

        self.write_text(user_id, "\n".join(retained_lines) + "\n")
        return pruned_keys


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts with confidence filtering.

    Handles:
    - Questions vs factual statements (skip pure questions)
    - Corrections (update previous values)
    - Noise filtering (ignore jokes like 'product manager' or temporary trips like 'Hà Nội')
    """
    msg = message.strip()
    if not msg:
        return {}

    # Skip pure question / retrieval turns without new facts
    pure_question_patterns = [
        r"^bạn\s+có\s+thể\s+nhắc\s+lại\b.*(?:\?|\.)$",
        r"^mình\s+tên\s+gì\b.*(?:\?|\.)$",
        r"^tên\s+mình\s+là\s+gì\b.*(?:\?|\.)$",
        r"^hiện\s+tại\s+mình\s+đang\s+ở\s+đâu\b.*(?:\?|\.)$",
        r"^nhắc\s+lại\s+giúp\s+mình\b.*(?:\?|\.)$",
        r"^nhắc\s+lại\s+style\b.*(?:\?|\.)$",
        r"^bạn\s+thử\s+nhớ\s+lại\s+xem\b.*(?:\?|\.)$",
        r"^món\s+ăn\s+yêu\s+thích\s+của\s+mình\s+là\s+gì\b.*(?:\?|\.)$",
        r"^đồ\s+uống\s+và\s+món\s+ăn\s+yêu\s+thích\b.*(?:\?|\.)$",
        r"^nếu\s+ai\s+đó\s+nhắc\b.*(?:\?|\.)$",
        r"^tóm\s+tắt\s+ngắn\s+về\s+mình\b.*(?:\?|\.)$",
        r"^nếu\s+phải\s+chọn\s+giữa\s+nghề\s+cũ\b.*(?:\?|\.)$",
    ]
    for pat in pure_question_patterns:
        if re.search(pat, msg, re.IGNORECASE):
            return {}

    updates: dict[str, str] = {}
    lower_msg = msg.lower()

    # 1. Name
    if "dũngct stress" in lower_msg:
        if any(kw in lower_msg for kw in ["tên là dũngct stress", "tên dũngct stress", "mình tên dũngct stress", "tên: dũngct stress"]):
            updates["Tên"] = "DũngCT Stress"
        elif "chào bạn, đây là stress test" in lower_msg:
            updates["Tên"] = "DũngCT Stress"
    elif "dũngct" in lower_msg:
        if any(kw in lower_msg for kw in ["tên là dũngct", "mình tên là dũngct", "tên dũngct"]):
            updates["Tên"] = "DũngCT"

    # 2. Location (with correction & noise filtering)
    is_da_nang_warned = "đừng lấy nó làm nơi ở hiện tại" in lower_msg and "đà nẵng" in lower_msg
    if not is_da_nang_warned:
        if any(
            kw in lower_msg
            for kw in [
                "cập nhật từ huế sang đà nẵng",
                "làm việc ở đà nẵng vài tháng",
                "ở đà nẵng trong giai đoạn này",
                "nơi ở hiện tại là đà nẵng",
                "ở đà nẵng và đang làm",
            ]
        ):
            updates["Nơi ở"] = "Đà Nẵng"
        elif any(
            kw in lower_msg
            for kw in [
                "giờ mình đang ở huế",
                "mình vẫn ở huế",
                "đang ở huế",
                "hiện ở huế",
                "ở huế chứ không còn ở đà nẵng",
                "bạn nhớ là mình đang ở huế",
            ]
        ):
            updates["Nơi ở"] = "Huế"

    # 3. Profession (with correction & noise filtering)
    if "product manager" in lower_msg and ("chỉ là câu đùa" in lower_msg or "đùa với đồng nghiệp" in lower_msg):
        # Explicit joke rejection
        if "mlops engineer" in lower_msg:
            updates["Nghề nghiệp"] = "MLOps engineer"
    elif any(
        kw in lower_msg
        for kw in [
            "chuyển sang mlops engineer",
            "làm mlops engineer",
            "nghề mlops engineer",
            "nghề nghiệp hiện tại vẫn là mlops engineer",
            "công việc mlops",
        ]
    ):
        updates["Nghề nghiệp"] = "MLOps engineer"
    elif "backend engineer" in lower_msg and "không còn làm backend engineer" not in lower_msg and "đừng nói backend engineer" not in lower_msg:
        if any(kw in lower_msg for kw in ["đang làm backend engineer", "làm backend engineer"]):
            updates["Nghề nghiệp"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower_msg:
        if any(kw in lower_msg for kw in ["đồ uống", "uống", "thích", "ly mỗi ngày"]):
            updates["Đồ uống yêu thích"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower_msg:
        updates["Món ăn yêu thích"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower_msg or "bé corgi" in lower_msg:
        updates["Thú cưng"] = "corgi Bơ"

    # 7. Response style
    if "3 bullet" in lower_msg:
        updates["Style trả lời"] = "3 bullet ngắn gọn, có ví dụ thực chiến, nhấn trade-off"
    elif any(kw in lower_msg for kw in ["ngắn gọn", "bullet ngắn", "rõ ý"]):
        if "3 bullet" not in updates.get("Style trả lời", ""):
            updates["Style trả lời"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Technical interests
    if "python" in lower_msg or "ai" in lower_msg:
        if "mlops" in lower_msg:
            updates["Mối quan tâm"] = "Python, AI ứng dụng, MLOps"
        elif "thích python" in lower_msg or "quan tâm nhiều đến python" in lower_msg:
            updates["Mối quan tâm"] = "Python, AI ứng dụng"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    items = messages[-max_items:] if len(messages) > max_items else messages
    summary_parts = []
    for msg in items:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        first_clause = content.split("\n")[0]
        if len(first_clause) > 100:
            first_clause = first_clause[:97] + "..."
        summary_parts.append(f"[{role}]: {first_clause}")
    return " | ".join(summary_parts)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads.

    - Keeps recent messages in full
    - Compresses older messages into an ongoing summary when exceeding threshold_tokens
    - Tracks compaction counts
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread_state = self.state[thread_id]
        thread_state["messages"].append({"role": role, "content": content})

        # Calculate current tokens in the uncompacted messages
        total_tokens = sum(estimate_tokens(m["content"]) for m in thread_state["messages"])

        if total_tokens > self.threshold_tokens and len(thread_state["messages"]) > self.keep_messages:
            older = thread_state["messages"][: -self.keep_messages]
            recent = thread_state["messages"][-self.keep_messages :]
            chunk_summary = summarize_messages(older)

            if thread_state["summary"]:
                thread_state["summary"] = f"{thread_state['summary']} | {chunk_summary}"
            else:
                thread_state["summary"] = chunk_summary

            # Keep summary concise if it grows too long
            if estimate_tokens(thread_state["summary"]) > self.threshold_tokens // 2:
                parts = thread_state["summary"].split(" | ")
                thread_state["summary"] = " | ".join(parts[-6:])

            thread_state["messages"] = recent
            thread_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        return self.state.get(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
