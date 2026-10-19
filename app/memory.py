"""记忆机制：短期会话记忆 + 长期经验记忆。

- SessionMemory：单个会话内的对话、规划、工具调用轨迹与产物，供多轮追问与报告汇总使用；
- LongTermMemory（JSONL 落盘）：跨会话沉淀“问题 → 规划 → 关键结论”，在新任务规划时召回相似历史，
  实现经验复用（检索增强式记忆）。
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings, settings as default_settings


@dataclass
class Artifact:
    """分析产物：图 / 表 / 栅格 / 报告。"""

    type: str          # image | csv | tif | report
    path: str
    title: str
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolRecord:
    """一次工具调用的完整记录。"""

    seq: int
    name: str
    args: dict[str, Any]
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    elapsed: float = 0.0
    step_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [a.to_dict() for a in self.artifacts]
        return payload


class SessionMemory:
    """单个会话的短期记忆。"""

    def __init__(self, session_id: str | None = None, max_records: int = 60) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.created_at = time.time()
        self.messages: list[dict[str, str]] = []
        self.records: list[ToolRecord] = []
        self.understanding: dict[str, Any] = {}
        self.plan: dict[str, Any] = {}
        self.max_records = max_records
        self._seq = 0

    # -- 对话 ---------------------------------------------------------
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > 40:
            self.messages = self.messages[-40:]

    def recent_messages(self, limit: int = 8) -> list[dict[str, str]]:
        return self.messages[-limit:]

    def last_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    # -- 工具记录 -----------------------------------------------------
    def add_record(self, record: ToolRecord) -> None:
        self._seq += 1
        record.seq = self._seq
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

    @property
    def artifacts(self) -> list[Artifact]:
        out: list[Artifact] = []
        seen: set[str] = set()
        for record in self.records:
            for art in record.artifacts:
                if art.path in seen:
                    continue
                seen.add(art.path)
                out.append(art)
        return out

    def results_digest(self, limit: int = 10) -> str:
        """把最近若干次工具结果压缩成文本，供模型续接推理。"""
        lines: list[str] = []
        for record in self.records[-limit:]:
            status = "成功" if record.ok else "失败"
            lines.append(f"[{record.seq}] {record.name}({_short_args(record.args)}) → {status}: {record.summary}")
        return "\n".join(lines)

    def find_records(self, name: str) -> list[ToolRecord]:
        return [r for r in self.records if r.name == name]

    def digest(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "messages": len(self.messages),
            "records": len(self.records),
            "artifacts": len(self.artifacts),
        }


class SessionStore:
    """会话仓库（进程内）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> SessionMemory:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            session = SessionMemory(session_id)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> SessionMemory | None:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> SessionMemory:
        with self._lock:
            self._sessions.pop(session_id, None)
        return self.get_or_create(None)


class LongTermMemory:
    """跨会话经验记忆，JSONL 持久化。"""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.path: Path = self.cfg.state_dir / "memory.jsonl"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def add(self, query: str, plan: dict[str, Any], insight: str, datasets: list[str] | None = None) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query.strip(),
            "plan": plan,
            "insight": insight.strip()[:1200],
            "datasets": datasets or [],
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._compact()

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        items: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        items = self.all()
        if not items or not query:
            return []
        q_tokens = _tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            text = item.get("query", "") + " " + " ".join(item.get("datasets", [])) + " " + item.get("insight", "")
            tokens = _tokens(text)
            if not tokens:
                continue
            overlap = len(q_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / (len(q_tokens) ** 0.5 + 1)
            scored.append((score, item))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [item for _, item in scored[:k]]

    def context(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k=k)
        if not hits:
            return ""
        lines = ["以下是与当前任务相似的历史分析经验，可作为规划参考（不要直接复用结论，需重新验证）："]
        for item in hits:
            steps = item.get("plan", {}).get("steps", [])
            step_desc = " → ".join(s.get("tool", "") for s in steps if s.get("tool"))
            lines.append(f"- 历史问题：{item.get('query', '')}\n  工具链：{step_desc}\n  结论：{item.get('insight', '')[:200]}")
        return "\n".join(lines)

    def _compact(self, keep: int = 200) -> None:
        items = self.all()
        if len(items) <= keep:
            return
        kept = items[-keep:]
        with self.path.open("w", encoding="utf-8") as fh:
            for item in kept:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _short_args(args: dict[str, Any], limit: int = 90) -> str:
    parts = [f"{k}={v}" for k, v in args.items() if v not in (None, "", [])]
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tokens(text: str) -> set[str]:
    text = text.lower()
    latin = set(re.findall(r"[a-z0-9_]{2,}", text))
    chinese = set(re.findall(r"[\u4e00-\u9fff]{1,}", text))
    grams: set[str] = set(latin) | chinese
    for chunk in chinese:
        for i in range(len(chunk) - 1):
            grams.add(chunk[i : i + 2])
    return grams


session_store = SessionStore()
long_term_memory = LongTermMemory()
