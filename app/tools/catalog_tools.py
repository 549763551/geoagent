"""目录类工具：数据检索（RAG）与数据集详情。"""

from __future__ import annotations

from ..catalog import DERIVED_INDICES, catalog
from .base import ToolResult, stats_of, tool, read_raster


@tool(
    name="search_catalog",
    description="在遥感数据目录中检索可用数据集。当不确定某个主题（如“干旱”“建成区”）有哪些数据时先调用它。"
                "返回数据集 key、单位、可用年份、分辨率等，后续工具用 key 引用数据。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "自然语言关键词，如“干旱”“地表温度”“土地利用”"},
            "top_k": {"type": "integer", "description": "返回条数，默认 5"},
        },
        "required": ["query"],
    },
    category="数据检索",
)
def search_catalog(ctx, query: str, top_k: int = 5) -> ToolResult:
    hits = catalog.search(query, k=int(top_k))
    if not hits:
        available = "、".join(f"{e.key}（{e.title}）" for e in catalog.all())
        return ToolResult(
            ok=True,
            summary=f"未检索到与“{query}”直接匹配的数据集。当前目录可用：{available}",
            data={"query": query, "matches": [], "available_keys": [e.key for e in catalog.all()]},
        )
    lines = [
        f"- {h['key']}｜{h['title']}｜单位 {h['unit']}｜"
        f"{('年份 ' + str(h['year_span'][0]) + '-' + str(h['year_span'][1])) if h['year_span'] else '情景数据'}｜"
        f"匹配度 {h['score']}"
        for h in hits
    ]
    derived = [f"{k}：{v}" for k, v in DERIVED_INDICES.items() if any(tok in query for tok in ("指数", "生态", "rsei", "ndvi", "水体", "干度"))]
    summary = "检索到以下数据：\n" + "\n".join(lines)
    if derived:
        summary += "\n可由 compute_index 计算的派生指标：\n" + "\n".join(f"- {d}" for d in derived)
    return ToolResult(ok=True, summary=summary, data={"query": query, "matches": hits})


@tool(
    name="describe_dataset",
    description="查看某个数据集的详细信息（单位、时间范围、分辨率、波段、分类编码），并抽样给出数值范围，"
                "用于判断数据是否适配当前分析任务。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "数据集 key，如 ndvi、lst、landuse"},
            "year": {"type": "integer", "description": "抽样年份，默认取最早年份"},
        },
        "required": ["dataset"],
    },
    category="数据检索",
)
def describe_dataset(ctx, dataset: str, year: int | None = None) -> ToolResult:
    entry = catalog.resolve(dataset)
    if entry is None:
        return ToolResult.failure(f"目录中不存在数据集“{dataset}”，可用：{', '.join(e.key for e in catalog.all())}")

    info = entry.to_dict()
    lines = [
        f"数据集 {entry.key}：{entry.title}",
        f"物理量：{entry.variable}（{entry.unit}）；分辨率：{entry.resolution_text()}；坐标系：{entry.crs}；栅格尺寸：{entry.shape[1]}×{entry.shape[0]}",
        f"说明：{entry.description}",
    ]
    if entry.classes:
        lines.append("分类编码：" + "，".join(f"{k}={v}" for k, v in entry.classes.items()))
    if entry.years:
        lines.append(f"可用年份：{entry.available_years}")
    if entry.scenario_files:
        lines.append("可用情景：" + "，".join(sorted(entry.scenario_files)))

    sample: dict = {}
    path = entry.path_for(year)
    if path is not None:
        arr, _, _ = read_raster(path)
        sample = stats_of(arr, entry.unit, {"sample_year": year or entry.nearest_year(None)})
        lines.append(
            f"抽样 {sample.get('sample_year')} 年：有效像元 {sample.get('valid_pixels')}，"
            f"均值 {sample.get('mean')}，中位数 {sample.get('median')}，P5~P95 = {sample.get('p5')}~{sample.get('p95')}"
        )
    return ToolResult(ok=True, summary="\n".join(lines), data={"dataset": info, "sample_stats": sample})
