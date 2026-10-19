"""OpenAI 兼容大模型客户端（DeepSeek / 通义千问 / 智谱 / 月之暗面 / Ollama 均可）。

只依赖 requests，实现 Chat Completions 协议下的普通对话与 Function Calling。
未配置 API Key 时 `available` 为 False，编排引擎会自动降级为离线规则模式。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import Settings, settings as default_settings

logger = logging.getLogger("geoagent.llm")


class LLMError(RuntimeError):
    """大模型调用失败。"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """轻量 Chat Completions 客户端。"""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings

    @property
    def available(self) -> bool:
        return self.cfg.llm_enabled

    # ------------------------------------------------------------------
    # 基础调用
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        if not self.available:
            raise LLMError("未配置 LLM_API_KEY，无法调用大模型")

        payload: dict[str, Any] = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=self.cfg.llm_timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"请求大模型失败: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(f"大模型返回 {resp.status_code}: {resp.text[:500]}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise LLMError(f"大模型返回非 JSON: {resp.text[:300]}") from exc

        return self._parse(body)

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return self.chat(messages, **kwargs).content

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """要求模型返回 JSON 对象并解析；解析失败时尝试从文本中抽取。"""
        if schema_hint:
            messages = messages + [{"role": "user", "content": schema_hint}]
        resp = self.chat(messages, json_mode=True, **kwargs)
        text = resp.content.strip()
        if not text:
            raise LLMError("大模型未返回内容")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            extracted = _extract_json(text)
            if extracted is None:
                raise LLMError(f"无法解析为 JSON: {text[:300]}") from None
            return extracted

    # ------------------------------------------------------------------
    def _parse(self, body: dict[str, Any]) -> LLMResponse:
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"大模型返回缺少 choices: {str(body)[:300]}")
        message = choices[0].get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw or {}
            calls.append(ToolCall(id=item.get("id") or fn.get("name", "call"), name=fn.get("name", ""), arguments=args))
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=body.get("usage") or {},
            raw=body,
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    """从可能包含 ```json 包裹的文本中抽取第一个 JSON 对象。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


llm_client = LLMClient()
