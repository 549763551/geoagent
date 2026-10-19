"""情景对比类工具：CMIP6 未来气候情景（SSP2-4.5 / SSP5-8.5）与历史基准的对比。

重要：CMIP6 数据为粗分辨率（约 0.22°，≈25 km），与主数据集 1 km 网格不一致，
因此这里只做“区域统计对比 + 情景差异空间格局”，不做与 1 km 数据的像元级差分。
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from ..catalog import catalog
from .base import (
    ToolResult,
    read_raster,
    render_raster_png,
    slugify,
    tool,
    write_csv,
)

SCENARIO_LABELS = {
    "hist": "历史基准",
    "ssp245": "SSP2-4.5 中等排放",
    "ssp585": "SSP5-8.5 高排放",
}
PERIOD_LABELS = {2030: "2030s（近期）", 2050: "2050s（中期）", 2100: "2100s（远期）"}
PERIODS = (2030, 2050, 2100)
SCENARIOS = ("ssp245", "ssp585")
_YEAR_RE = re.compile(r"\d{4}")


def _parse_periods(period) -> tuple[list[int], str | None]:
    """宽容解析模型给出的时段写法，返回（时段列表, 错误信息）。

    大模型偶尔忽略 schema 的类型约束，把时段写成 "2030s"、"2030s,2050s,2100s"、
    "2030s/2050s/2100s" 或列表形式；这里统一抽取其中的四位年份并归一化为
    PERIODS 的子集，避免因书写差异直接调用失败。
    """
    if period in (None, "", "all", "ALL"):
        return list(PERIODS), None
    text = (" ".join(str(x) for x in period)
            if isinstance(period, (list, tuple, set)) else str(period))
    picked: list[int] = []
    for token in re.split(r"[,，/、;；|\s]+", text.strip()):
        if not token:
            continue
        years = _YEAR_RE.findall(token)
        if not years:
            return [], f"无法识别时段“{token}”，可选：2030、2050、2100"
        for raw_year in years:
            year = int(raw_year)
            if year not in PERIODS:
                return [], f"不支持的时段“{token}”，可选：2030、2050、2100"
            if year not in picked:
                picked.append(year)
    return (picked or list(PERIODS)), None


def _parse_scenarios(scenario) -> tuple[list[str], str | None]:
    """宽容解析模型给出的排放路径写法，返回（情景列表, 错误信息）。

    模型经常写 "SSP2-4.5"、"SSP5-8.5"、"SSP245" 这类人类可读名称，而 schema
    声明的是 ssp245/ssp585 枚举；这里统一归一化，避免因书写差异调用失败。
    """
    if scenario in (None, "", "all", "ALL"):
        return list(SCENARIOS), None
    text = (" ".join(str(x) for x in scenario)
            if isinstance(scenario, (list, tuple, set)) else str(scenario))
    picked: list[str] = []
    for token in re.split(r"[,，/、;；|\s]+", text.strip()):
        if not token:
            continue
        compact = re.sub(r"[^0-9a-z]", "", token.lower())
        if "245" in compact:
            key = "ssp245"
        elif "585" in compact:
            key = "ssp585"
        else:
            return [], f"无法识别排放路径“{token}”，可选：ssp245（SSP2-4.5）、ssp585（SSP5-8.5）"
        if key not in picked:
            picked.append(key)
    return (picked or list(SCENARIOS)), None


VARIABLES = {
    "precip": ("cmip_precip", "降水", "mm/日"),
    "temp": ("cmip_temp", "气温", "摄氏度(°C)"),
}


def _grouped_bars(groups, series, path, title, ylabel, note="", colors=None) -> None:
    """分组柱状图：groups 为 x 轴标签，series 为 {系列名: [值...]}。"""
    positions = np.arange(len(groups))
    names = list(series)
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(positions + offset, series[name], width * 0.92,
                      label=name, color=(colors or {}).get(name), edgecolor="#555")
        for bar, val in zip(bars, series[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    if note:
        ax.text(0.01, -0.20, note, transform=ax.transAxes, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@tool(
    name="scenario_compare",
    description="对比 CMIP6 未来气候情景：以历史基准为参照，计算 SSP2-4.5 / SSP5-8.5 在 2030s / 2050s / 2100s 的"
                "降水或气温区域平均及其变化幅度，输出对比表、分组柱状图与情景差异空间图。"
                "适合回答“未来降水/气温会怎样变化、不同排放路径差异多大”。"
                "注意：CMIP6 为约 25km 粗分辨率，仅做区域统计与格局对比，不与其他 1km 数据做像元级差分。",
    parameters={
        "type": "object",
        "properties": {
            "variable": {"type": "string", "enum": ["precip", "temp"],
                         "description": "precip=降水，temp=气温，默认 precip"},
            "scenario": {"type": "string", "enum": ["all", "ssp245", "ssp585"],
                         "description": "排放路径，默认 all（两条路径都给）。ssp245 即 SSP2-4.5，ssp585 即 SSP5-8.5。"},
            "period": {"type": "integer", "enum": [2030, 2050, 2100],
                       "description": "只分析某一个时段（单个整数，如 2030）；不填则三个时段都给。"
                                      "若要多个时段，请分多次调用本工具。"},
        },
        "required": [],
    },
    category="情景分析",
)
def scenario_compare(ctx, variable: str = "precip", scenario: str = "all",
                     period: int | str | list | None = None) -> ToolResult:
    variable = (variable or "precip").strip().lower()
    if variable not in VARIABLES:
        return ToolResult.failure(f"不支持的情景变量“{variable}”，可选：precip、temp")
    entry_key, var_label, unit = VARIABLES[variable]
    entry = catalog.get(entry_key)
    if entry is None or not entry.scenario_files:
        return ToolResult.failure(f"未找到 CMIP6 {var_label}情景数据")

    hist_path = entry.scenario_path("hist")
    if hist_path is None:
        return ToolResult.failure(f"缺少 CMIP6 {var_label}历史基准（hist）")

    wanted_scenarios, scenario_err = _parse_scenarios(scenario)
    if scenario_err:
        return ToolResult.failure(scenario_err)
    wanted_periods, period_err = _parse_periods(period)
    if period_err:
        return ToolResult.failure(period_err)

    ctx.report(f"读取 CMIP6 {var_label}历史基准与情景数据…", 20)
    hist_arr, hist_profile, _ = read_raster(hist_path)
    hist_mean = float(np.nanmean(hist_arr))

    rows: list[list] = []
    details: list[dict] = []
    for scen in wanted_scenarios:
        if scen not in SCENARIO_LABELS:
            return ToolResult.failure(f"未知情景“{scen}”，可选：ssp245、ssp585")
        for per in wanted_periods:
            path = entry.scenario_path(f"{scen}_{per}")
            if path is None:
                continue
            arr, _, _ = read_raster(path)
            mean = float(np.nanmean(arr))
            delta = mean - hist_mean
            pct = delta / hist_mean * 100 if hist_mean else float("nan")
            rows.append([SCENARIO_LABELS[scen], PERIOD_LABELS.get(per, str(per)),
                         round(mean, 3), round(delta, 3), round(pct, 2)])
            details.append({"scenario": scen, "scenario_label": SCENARIO_LABELS[scen],
                            "period": per, "mean": round(mean, 3),
                            "delta": round(delta, 3), "delta_pct": round(pct, 2)})

    if not details:
        return ToolResult.failure("给定条件下没有可用的 CMIP6 情景数据")

    ctx.report("计算情景差异并制图…", 60)
    # 三个时段齐全时统一用 all 命名，与"不填 period"的默认行为保持一致
    period_tag = ("all" if tuple(wanted_periods) == PERIODS
                  else "-".join(str(p) for p in wanted_periods))
    scenario_tag = ("all" if tuple(wanted_scenarios) == SCENARIOS
                    else "-".join(wanted_scenarios))
    tag = f"cmip_{variable}_{scenario_tag}_{period_tag}"

    chart_periods = sorted({d["period"] for d in details})
    series: dict[str, list[float]] = {}
    for scen in wanted_scenarios:
        vals = [next((d["mean"] for d in details if d["scenario"] == scen and d["period"] == p), np.nan)
                for p in chart_periods]
        if any(np.isfinite(vals)):
            series[SCENARIO_LABELS[scen]] = vals
    if series:
        series["历史基准"] = [hist_mean] * len(chart_periods)
    colors = {"SSP2-4.5 中等排放": "#F0AD4E", "SSP5-8.5 高排放": "#D9534F", "历史基准": "#4C8BF5"}
    bar_png = ctx.out_dir("figures") / f"{slugify(tag)}_bars.png"
    _grouped_bars(
        [PERIOD_LABELS.get(p, str(p)) for p in chart_periods], series, bar_png,
        title=f"CMIP6 {var_label}区域平均情景对比",
        ylabel=f"{var_label}（{unit}）",
        note="历史基准为虚线参照；CMIP6 多模式集合，约 25km 分辨率",
        colors=colors,
    )

    # 情景差异空间格局：取幅度最大的情景-时段
    focus = max(details, key=lambda d: abs(d["delta"]))
    focus_arr, _, _ = read_raster(entry.scenario_path(f"{focus['scenario']}_{focus['period']}"))
    delta_arr = focus_arr - hist_arr
    diff_png = ctx.out_dir("figures") / f"{slugify(tag)}_diff.png"
    lim = float(np.nanpercentile(np.abs(delta_arr), 98)) or 1e-6
    render_raster_png(
        delta_arr, hist_profile, diff_png,
        title=f"{var_label}情景差异（{focus['scenario_label']} {PERIOD_LABELS.get(focus['period'], focus['period'])} − 历史基准）",
        unit=f"Δ{var_label}（{unit}）", cmap="RdBu_r", vmin=-lim, vmax=lim,
        note="正值（蓝）表示升高，负值（红）表示降低",
    )

    csv_path = ctx.out_dir("tables") / f"{slugify(tag)}.csv"
    write_csv(csv_path, ["情景", "时段", f"区域平均({unit})", f"相对历史变化({unit})", "变化率(%)"], rows)

    strongest = max(details, key=lambda d: d["delta"])
    weakest = min(details, key=lambda d: d["delta"])
    trend_dir = "升高" if strongest["delta"] >= 0 else "降低"
    summary = (
        f"CMIP6 {var_label}情景对比（历史基准 {hist_mean:.3f} {unit}）："
        f"{strongest['scenario_label']} {PERIOD_LABELS.get(strongest['period'], strongest['period'])} "
        f"区域平均 {strongest['mean']:.3f} {unit}（{strongest['delta']:+.3f}，{strongest['delta_pct']:+.1f}%），为所选情景中变化最大；"
        f"{weakest['scenario_label']} {PERIOD_LABELS.get(weakest['period'], weakest['period'])} 为 {weakest['delta']:+.3f}。"
        f"整体呈{trend_dir}态势。已输出对比表、分组柱状图与情景差异空间图。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={"variable": variable, "unit": unit, "hist_mean": round(hist_mean, 3),
              "scenarios": details},
        artifacts=[
            ctx.artifact(bar_png, "image", f"CMIP6 {var_label}情景对比柱状图"),
            ctx.artifact(diff_png, "image", f"CMIP6 {var_label}情景差异图"),
            ctx.artifact(csv_path, "csv", f"CMIP6 {var_label}情景对比表"),
        ],
    )
