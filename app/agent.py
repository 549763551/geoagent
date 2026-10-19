"""智能体编排引擎：任务理解 → 自主规划 → ReAct 工具调用 → 结果交付。

两种运行模式：
- 在线模式：配置了 LLM_API_KEY 时，用大模型完成任务理解与规划，并在执行后进入
  Function Calling 的 ReAct 循环，按工具返回结果决定是否追加调用，最后生成自然语言结论；
- 离线模式：未配置 Key 时，退化为关键词规则解析 + 规则化工具链，保证功能可完整演示。

引擎以事件流（emit 回调）方式向外暴露全过程，供 Web 层以 SSE 实时推送。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .catalog import DERIVED_INDICES, catalog
from .config import Settings, settings as default_settings
from .llm import LLMClient, LLMError, llm_client
from .memory import LongTermMemory, SessionMemory, ToolRecord, long_term_memory
from .tools import registry

logger = logging.getLogger("geoagent.agent")

EmitFn = Callable[[dict[str, Any]], None]

INDEX_ALIASES: dict[str, list[str]] = {
    "NDVI": ["ndvi", "植被指数", "绿度"],
    "NDWI": ["ndwi", "水体指数", "地表水指数"],
    "MNDWI": ["mndwi"],
    "NDBI": ["ndbi", "建筑指数", "不透水"],
    "SAVI": ["savi", "土壤调节植被指数"],
    "WET": ["wet", "湿度分量", "缨帽"],
    "NDBSI": ["ndbsi", "干度"],
    "RSEI": ["rsei", "遥感生态指数", "生态指数", "生态质量", "生态状况"],
}

# 支持直接分析派生指数（index 参数）的工具
INDEX_TOOLS = {"trend_analysis", "change_detection", "compose_map"}

TASK_KEYWORDS: dict[str, list[str]] = {
    "trend": ["趋势", "演变", "逐年变化", "显著性", "mann", "kendall", "sen", "改善", "退化", "变好", "变差", "倾向"],
    "change": ["变化检测", "变化最", "变化明显", "变化区域", "变化情况", "两年", "对比两年",
               "增加了多少", "减少了多少", "差异", "变化量", "前后对比"],
    "scenario": ["情景", "未来", "ssp", "cmip", "2030", "2050", "2100", "排放路径", "预估", "预测"],
    "compose": ["制图", "绘制", "绘图", "作图", "出图", "成图", "对比图", "分布图", "并排", "拼图",
                "多期", "放在一张", "专题图"],
    "zonal": ["分区", "地类", "土地利用", "各类型", "分地类", "哪类", "耕地", "林地", "草地"],
    "timeseries": ["时间序列", "逐年", "年际", "折线", "区域平均变化"],
    "stats": ["统计", "平均", "分布", "极值", "范围", "多少", "概况"],
    "index": ["指数", "生态", "绿度", "热度", "干度", "湿度", "综合评估"],
    "report": ["报告", "总结", "汇总", "形成报告", "出具", "撰写", "汇总成文"],
    "describe": ["有哪些数据", "数据目录", "可用数据", "什么是", "介绍", "字段说明"],
}


# ======================================================================
# 事件
# ======================================================================
def evt(kind: str, **payload: Any) -> dict[str, Any]:
    return {"type": kind, "ts": time.time(), **payload}


@dataclass
class StepPlan:
    title: str
    tool: str
    args: dict[str, Any]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "tool": self.tool, "args": self.args, "reason": self.reason}


# ======================================================================
# 轻量解析工具（离线模式与参数补全共用）
# ======================================================================
def extract_years(text: str) -> tuple[int | None, int | None, list[int]]:
    span = re.search(r"(19|20)\d{2}\s*(?:-|–|~|—|至|到|—)\s*((19|20)\d{2})", text)
    years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)]
    if span:
        a, b = int(span.group(0)[:4]), int(re.findall(r"(?:19|20)\d{2}", span.group(0))[-1])
        return min(a, b), max(a, b), sorted(set(years))
    if len(years) >= 2:
        return min(years), max(years), sorted(set(years))
    return (years[0] if years else None), (years[0] if years else None), sorted(set(years))


def detect_indices(text: str) -> list[str]:
    """识别文本中提到的派生指数。拉丁指数名按词边界匹配，避免 MNDWI 内含 NDWI 造成误判。"""
    upper = text.upper()
    hits: list[str] = []
    for name, aliases in INDEX_ALIASES.items():
        hit = bool(re.search(rf"(?<![A-Z0-9]){name}(?![A-Z0-9])", upper))
        if not hit:
            for alias in aliases:
                if alias.isascii():
                    if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", upper):
                        hit = True
                        break
                elif alias in text:
                    hit = True
                    break
        if hit:
            hits.append(name)
    return hits


def _object_patterns() -> list[re.Pattern[str]]:
    """分析对象名称（数据集标题/短标题 + 派生指数名）的匹配模式。

    判定任务类型前先屏蔽这些词，避免对象名本身污染类型识别：
    对象「土地利用/覆被」含 zonal 关键词、「地形湿度指数」含 index 关键词、
    对象「CMIP6 未来降水情景」含 scenario 关键词等。"""
    tokens: set[str] = set(INDEX_ALIASES)
    for entry in catalog.all():
        title = entry.title or ""
        tokens.add(title)
        tokens.add(title.split("（")[0].split("(")[0].strip())
    return [re.compile(re.escape(t), re.IGNORECASE) for t in sorted((t for t in tokens if t), key=len, reverse=True)]


_OBJECT_PATTERNS = _object_patterns()


def detect_task_type(text: str) -> str:
    masked = text
    for pat in _OBJECT_PATTERNS:
        masked = pat.sub(" ", masked)
    low = masked.lower()
    scores = {k: sum(1 for kw in kws if kw in low) for k, kws in TASK_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "stats"


# ======================================================================
# 引擎
# ======================================================================
class AgentEngine:
    def __init__(self, cfg: Settings | None = None, llm: LLMClient | None = None,
                 memory: LongTermMemory | None = None) -> None:
        self.cfg = cfg or default_settings
        self.llm = llm or llm_client
        self.memory = memory or long_term_memory

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return "llm" if self.llm.available else "offline"

    def run(self, query: str, session: SessionMemory, emit: EmitFn) -> dict[str, Any]:
        """执行一段端到端任务，向 emit 推送事件；返回最终交付结果。"""
        query = (query or "").strip()
        if not query:
            emit(evt("error", text="请输入任务描述"))
            return {"answer": "", "artifacts": []}

        session.add_message("user", query)
        emit(evt("status", stage="start", text=f"接收任务（{'大模型' if self.llm.available else '离线规则'}模式）"))
        try:
            return self._run_pipeline(query, session, emit)
        except LLMError as exc:
            logger.warning("大模型调用失败，降级为离线模式: %s", exc)
            emit(evt("status", stage="fallback", text=f"大模型不可用（{exc}），已切换离线规则模式"))
            try:
                return self._run_pipeline(query, session, emit, force_offline=True)
            except Exception as inner:  # pragma: no cover
                logger.exception("离线执行失败")
                emit(evt("error", text=f"任务执行失败：{inner}"))
                return {"answer": "", "artifacts": []}
        except Exception as exc:  # pragma: no cover
            logger.exception("任务执行失败")
            emit(evt("error", text=f"任务执行失败：{exc}"))
            return {"answer": "", "artifacts": []}

    # ------------------------------------------------------------------
    def _run_pipeline(self, query: str, session: SessionMemory, emit: EmitFn,
                      force_offline: bool = False) -> dict[str, Any]:
        use_llm = self.llm.available and not force_offline

        # 1) 任务理解 ---------------------------------------------------
        emit(evt("status", stage="understanding", text="理解任务意图与数据需求…"))
        understanding = (self._understand_llm(query) if use_llm else None) or self._understand_rules(query)
        session.understanding = understanding
        emit(evt("understanding", data=understanding))

        # 2) 记忆召回（长期经验 + 目录检索） ---------------------------
        memory_ctx = self.memory.context(query)
        hits = catalog.search(query, k=4)
        if memory_ctx:
            emit(evt("status", stage="recall", text="召回历史经验，辅助规划"))
        emit(evt("recall", memory=memory_ctx, catalog=[{"key": h["key"], "title": h["title"], "score": h["score"]}
                                                       for h in hits]))

        # 3) 自主规划 ---------------------------------------------------
        emit(evt("status", stage="planning", text="制定分析计划…"))
        plan_steps = (self._plan_llm(query, understanding, memory_ctx) if use_llm else None)
        if not plan_steps:
            plan_steps = self._plan_rules(query, understanding)
        session.plan = {"steps": [s.to_dict() for s in plan_steps]}
        emit(evt("plan", steps=session.plan["steps"], mode=self.mode if not force_offline else "offline"))

        # 4) 执行：先按计划执行，再由大模型 ReAct 追加 -----------------
        executed: list[ToolRecord] = []
        for step in plan_steps:
            if len(executed) >= self.cfg.max_steps:
                emit(evt("status", stage="limit", text=f"已达最大步数 {self.cfg.max_steps}，停止追加"))
                break
            rec = self._execute_step(step, session, emit)
            if rec:
                executed.append(rec)

        if use_llm:
            self._react_loop(query, understanding, session, emit, executed)

        # 5) 结果交付 ---------------------------------------------------
        emit(evt("status", stage="finalizing", text="汇总分析结论…"))
        answer = None
        if use_llm and session.records:
            answer = self._final_answer_llm(query, session)
        if not answer:
            answer = self._final_answer_rules(query, understanding, session)

        session.add_message("assistant", answer)
        artifacts = [a.to_dict() for a in session.artifacts]

        # 沉淀长期记忆
        try:
            self.memory.add(query, session.plan, insight=answer,
                            datasets=[d for d in understanding.get("datasets", []) if d])
        except Exception:  # pragma: no cover
            logger.warning("长期记忆写入失败", exc_info=True)

        payload = {
            "answer": answer,
            "mode": self.mode if not force_offline else "offline",
            "understanding": understanding,
            "plan": session.plan,
            "steps": [r.to_dict() for r in session.records],
            "artifacts": artifacts,
            "count": {"tools": len(session.records), "artifacts": len(artifacts)},
        }
        emit(evt("final", data=payload))
        return payload

    # ==================================================================
    # 任务理解
    # ==================================================================
    def _understand_llm(self, query: str) -> dict[str, Any] | None:
        system = (
            "你是遥感栅格分析智能体的【任务理解】模块。请把用户的中文需求解析为严格的 json 对象，不要输出解释。\n"
            "字段：\n"
            "- goal: 一句话概括用户目标\n"
            "- task_type: 主任务类型，从 trend/change/scenario/compose/zonal/timeseries/stats/index/report/describe 中择一\n"
            "- datasets: 需要的数据集 key 列表（只能从下方目录中选，选择最相关的 1-3 个）\n"
            "- indices: 需要计算的派生指数列表（可空），取值 NDVI/NDWI/MNDWI/NDBI/SAVI/WET/NDBSI/RSEI\n"
            "- start_year / end_year: 年份（整数，未提及填 null）\n"
            "- deliverable: 期望交付物（如“趋势分级图与统计表”）\n"
            "数据目录：\n" + catalog.summary() + "\n派生指数：\n" +
            "\n".join(f"- {k}: {v}" for k, v in DERIVED_INDICES.items())
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
        try:
            data = self.llm.chat_json(messages, max_tokens=4000)
        except LLMError as exc:
            logger.warning("任务理解失败：%s", exc)
            return None
        return self._normalize_understanding(data, query)

    def _understand_rules(self, query: str) -> dict[str, Any]:
        start, end, years = extract_years(query)
        indices = detect_indices(query)
        task_type = detect_task_type(query)
        datasets: list[str] = []
        for hit in catalog.search(query, k=2):
            datasets.append(hit["key"])
        if not datasets and indices:
            datasets = ["reflectance"]
        return self._normalize_understanding(
            {"goal": query, "task_type": task_type, "datasets": datasets, "indices": indices,
             "start_year": start, "end_year": end,
             "deliverable": "生成分析报告" if "报告" in query else ""},
            query, years=years,
        )

    def _normalize_understanding(self, data: dict[str, Any], query: str,
                                 years: list[int] | None = None) -> dict[str, Any]:
        valid_keys = set(catalog.entries)
        datasets = [d for d in (data.get("datasets") or []) if isinstance(d, str) and d in valid_keys]
        indices = [i for i in (data.get("indices") or []) if isinstance(i, str) and i.upper() in INDEX_ALIASES]
        indices = [i.upper() for i in indices]
        start = data.get("start_year")
        end = data.get("end_year")
        start = int(start) if isinstance(start, (int, float)) else None
        end = int(end) if isinstance(end, (int, float)) else None
        if years:
            if start is None:
                start = min(years)
            if end is None:
                end = max(years)
        return {
            "goal": (data.get("goal") or query).strip(),
            "task_type": data.get("task_type") or detect_task_type(query),
            "datasets": datasets,
            "indices": indices,
            "start_year": start,
            "end_year": end,
            "deliverable": (data.get("deliverable") or "").strip(),
        }

    # ==================================================================
    # 规划
    # ==================================================================
    def _plan_llm(self, query: str, understanding: dict[str, Any],
                  memory_ctx: str) -> list[StepPlan] | None:
        system = (
            "你是遥感栅格分析智能体的【规划】模块。请基于任务理解，给出 1-5 步可执行的工具调用计划，"
            "输出严格 json：{\"steps\":[{\"title\":\"步骤名\",\"tool\":\"工具名\",\"args\":{...},\"reason\":\"依据\"}]}。\n"
            "要求：\n"
            "1. tool 必须来自下方工具清单；args 必须符合该工具参数（年份用整数，index 用大写）；\n"
            "2. 不要编造数据集 key，只能用目录中存在的（或 index 派生指数）；\n"
            "3. 若用户要求形成报告，最后一步用 generate_report；\n"
            "4. 步骤尽量精简，避免重复读取同一数据。\n\n"
            "可用工具：\n" + registry.catalog_text() +
            "\n\n数据目录：\n" + catalog.summary() +
            ("\n\n历史经验参考：\n" + memory_ctx if memory_ctx else "")
        )
        user = (f"任务理解：{json.dumps(understanding, ensure_ascii=False)}\n用户原话：{query}\n"
                f"请给出执行计划 JSON。")
        try:
            data = self.llm.chat_json([{"role": "system", "content": system},
                                       {"role": "user", "content": user}], max_tokens=4000)
        except LLMError as exc:
            logger.warning("规划失败：%s", exc)
            return None
        steps = self._validate_steps(data.get("steps") or [])
        return steps or None

    def _validate_steps(self, raw: list[Any]) -> list[StepPlan]:
        out: list[StepPlan] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool") or "").strip()
            if registry.get(name) is None:
                continue
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            out.append(StepPlan(title=str(item.get("title") or name), tool=name, args=args,
                                reason=str(item.get("reason") or "")))
        return out

    def _plan_rules(self, query: str, understanding: dict[str, Any]) -> list[StepPlan]:
        task = understanding["task_type"]
        start, end = understanding.get("start_year"), understanding.get("end_year")
        index = (understanding.get("indices") or [None])[0]
        datasets = understanding.get("datasets") or []
        dataset = datasets[0] if datasets else None
        steps: list[StepPlan] = []

        def add(title: str, tool: str, args: dict[str, Any], reason: str = "") -> None:
            steps.append(StepPlan(title=title, tool=tool, args=args, reason=reason))

        def subject_for(tool: str, ds: str | None = None) -> tuple[dict[str, Any] | None, str]:
            """按工具能力构造分析对象；返回 (参数字典, 展示名)，None 表示该工具无法处理。
            ds 可显式指定数据集（如分区统计需排除作为分区底图的 landuse）。"""
            target = ds if ds is not None else dataset
            if tool in INDEX_TOOLS:
                if index:
                    return {"index": index}, index
                if target:
                    return {"dataset": target}, target
                return {}, "区域"
            # 仅接受 dataset 的工具：优先把指数映射到同名数据集
            if index:
                if catalog.has(index.lower()):
                    return {"dataset": index.lower()}, index
                return None, index
            if target:
                return {"dataset": target}, target
            return {}, "区域"

        if task == "describe" and not (index or dataset):
            add("检索数据目录", "search_catalog", {"query": query}, "先明确可用数据")
            return steps

        if task in ("trend", "change", "compose"):
            tool = {"trend": "trend_analysis", "change": "change_detection", "compose": "compose_map"}[task]
            subject, name = subject_for(tool)
            if subject is not None:
                if task == "change":
                    add(f"{name} 变化检测", tool, {**subject, "start_year": start, "end_year": end}, "两期差异对比")
                elif task == "trend":
                    add(f"{name} 趋势分析", tool, {**subject, "start_year": start, "end_year": end},
                        "逐像元显著性趋势")
                else:
                    years = [y for y in (start, end) if y]
                    if start and end and start != end:
                        years = sorted({start, (start + end) // 2, end})
                    add(f"{name} 多时相制图", tool, {**subject, "years": years or None}, "多期对比成图")
                return self._maybe_report(steps, query, understanding)
        elif task == "scenario":
            variable = "temp" if any(k in query for k in ["气温", "温度", "升温", "变暖", "tas"]) else "precip"
            add("CMIP6 情景对比", "scenario_compare", {"variable": variable, "scenario": "all"}, "未来气候情景")
            return self._maybe_report(steps, query, understanding)
        elif task == "zonal":
            # 分区底图固定为 landuse，分析对象应取检索结果里首个非 landuse 的数据集
            zonal_ds = next((d for d in datasets if d != "landuse"), dataset)
            subject, name = subject_for("zonal_by_landuse", zonal_ds)
            if subject is not None:
                add(f"{name} 分区统计", "zonal_by_landuse", {**subject, "year": end}, "按土地利用分区统计")
                return self._maybe_report(steps, query, understanding)
        elif task == "timeseries":
            subject, name = subject_for("time_series")
            if subject is not None:
                add(f"{name} 时间序列", "time_series",
                    {**subject, "start_year": start, "end_year": end}, "区域均值时间序列")
                return self._maybe_report(steps, query, understanding)
        elif task == "index":
            name = index or "RSEI"
            args = {"index": name, "year": end}
            if not catalog.has("reflectance") and name != "RSEI":
                args = {"index": "RSEI", "year": end}
                name = "RSEI"
            add(f"计算 {name}", "compute_index", args, "合成遥感生态指数")
            if start and end and start != end:
                add("生态指数变化检测", "change_detection",
                    {"index": name, "start_year": start, "end_year": end}, "对比首末期")
            return self._maybe_report(steps, query, understanding)
        elif task == "report":
            subject, _ = subject_for("time_series")
            target = subject or ({"dataset": "ndvi"} if catalog.has("ndvi") else {})
            add("基础统计分析", "time_series", {**target, "start_year": start, "end_year": end}, "为报告准备数据")
            add("生成分析报告", "generate_report", {"question": query}, "输出结论文档")
            return steps[:5]

        # stats / 兜底：能映射到数据集就统计，否则退化为指数合成
        subject, name = subject_for("raster_stats")
        if subject is not None and subject:
            add(f"{name} 统计", "raster_stats", {**subject, "year": end}, "描述统计")
        elif index:
            add(f"计算 {index}", "compute_index", {"index": index, "year": end}, "该指数需现算")
        else:
            add("检索数据目录", "search_catalog", {"query": query}, "定位数据")
            if catalog.has("ndvi"):
                add("统计 NDVI", "raster_stats", {"dataset": "ndvi", "year": end}, "示例统计")
        return self._maybe_report(steps, query, understanding)

    def _maybe_report(self, steps: list[StepPlan], query: str,
                      understanding: dict[str, Any]) -> list[StepPlan]:
        deliverable = understanding.get("deliverable") or ""
        if ("报告" in deliverable) and not any(s.tool == "generate_report" for s in steps):
            steps.append(StepPlan(title="生成分析报告", tool="generate_report",
                                  args={"question": query}, reason="按需输出报告"))
        return steps[:5]

    # ==================================================================
    # 执行
    # ==================================================================
    def _execute_step(self, step: StepPlan, session: SessionMemory, emit: EmitFn,
                      source: str = "plan") -> ToolRecord | None:
        emit(evt("step_start", title=step.title, tool=step.tool, args=step.args, reason=step.reason))
        emit(evt("status", stage="acting", text=f"执行：{step.title}"))

        def progress(text: str, pct: float = -1.0) -> None:
            emit(evt("progress", text=text, pct=pct, tool=step.tool))

        from .tools.base import ToolContext
        ctx = ToolContext(settings=self.cfg, session=session, progress=progress)
        started = time.time()
        result = registry.run(step.tool, step.args, ctx)
        elapsed = time.time() - started
        record = ToolRecord(seq=0, name=step.tool, args=step.args, ok=result.ok, summary=result.summary,
                            data=result.data, artifacts=result.artifacts, elapsed=round(elapsed, 2),
                            step_title=step.title)
        session.add_record(record)
        emit(evt("tool_result", data=record.to_dict()))
        return record

    def _react_loop(self, query: str, understanding: dict[str, Any], session: SessionMemory,
                    emit: EmitFn, executed: list[ToolRecord]) -> None:
        """执行计划后，允许大模型基于结果追加工具调用（ReAct）。"""
        tools = registry.openai_schemas()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": (
                "你是遥感栅格分析智能体。已完成既定计划，你可以根据已有结果决定："
                "继续调用工具补足信息，或直接给出最终结论。若信息已足够，请不要再调用工具，"
                "直接用中文给出结论（含关键数值与产物说明）。"
                "\n\n数据目录：\n" + catalog.summary())
             },
            {"role": "user", "content": f"用户任务：{query}\n\n已完成步骤与结果：\n{session.results_digest()}"},
        ]
        extra = 0
        while len(executed) < self.cfg.max_steps and extra < 3:
            try:
                resp = self.llm.chat(messages, tools=tools, temperature=0.1, max_tokens=4000)
            except LLMError as exc:
                logger.warning("ReAct 调用失败：%s", exc)
                break
            if not resp.tool_calls:
                if resp.content.strip():
                    session.add_message("assistant", resp.content.strip())
                    emit(evt("reflection", text=resp.content.strip()))
                break
            messages.append({
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in resp.tool_calls
                ],
            })
            for tc in resp.tool_calls:
                if registry.get(tc.name) is None:
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps({"error": f"未知工具 {tc.name}"}, ensure_ascii=False)})
                    continue
                if len(executed) >= self.cfg.max_steps:
                    break
                step = StepPlan(title=f"追加调用 {tc.name}", tool=tc.name, args=tc.arguments,
                                reason="根据中间结果自主追加")
                record = self._execute_step(step, session, emit, source="react")
                if record:
                    executed.append(record)
                    extra += 1
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"ok": record.ok if record else False,
                                           "summary": record.summary if record else "执行失败"},
                                          ensure_ascii=False),
                })

    # ==================================================================
    # 结果交付
    # ==================================================================
    def _final_answer_llm(self, query: str, session: SessionMemory) -> str | None:
        system = (
            "你是遥感栅格分析智能体的【结果交付】模块。请基于真实工具执行结果，用中文撰写最终答复："
            "先直接给出核心结论（含关键数值），再简要说明数据与方法，最后列出已生成的产物。"
            "严禁编造任何结果中没有的数据；若某步失败，请如实说明。"
        )
        user = f"用户任务：{query}\n\n任务理解：{json.dumps(session.understanding, ensure_ascii=False)}\n\n" \
               f"执行轨迹：\n{session.results_digest()}"
        try:
            text = self.llm.chat_text([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      temperature=0.3, max_tokens=8000)
        except LLMError as exc:
            logger.warning("结果交付失败：%s", exc)
            return None
        return text.strip() or None

    def _final_answer_rules(self, query: str, understanding: dict[str, Any],
                            session: SessionMemory) -> str:
        lines: list[str] = []
        goal = understanding.get("goal") or query
        lines.append(f"已完成“{goal}”的分析。")
        ok = [r for r in session.records if r.ok]
        bad = [r for r in session.records if not r.ok]
        if ok:
            lines.append("")
            lines.append("分析结论：")
            for rec in ok:
                lines.append(f"{rec.seq}. {rec.step_title or rec.name}：{rec.summary}")
        if bad:
            lines.append("")
            lines.append("未完成的步骤：")
            for rec in bad:
                lines.append(f"- {rec.step_title or rec.name}：{rec.summary}")
        arts = session.artifacts
        if arts:
            lines.append("")
            lines.append(f"已生成 {len(arts)} 项产物：" + "、".join(a.title for a in arts[:8]) +
                         ("…" if len(arts) > 8 else ""))
        return "\n".join(lines)


engine = AgentEngine()
