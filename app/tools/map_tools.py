"""制图类工具：把同一对象的多期栅格拼成多面板对比图（专题制图）。

单期专题图由各分析工具自带；本工具解决“把若干年份放在一张图上横向对比”的需求，
例如“把 NDVI 2000 / 2010 / 2020 三期并排成图”。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from rasterio.transform import array_bounds

from .base import ToolResult, slugify, tool
from .trend_tools import _resolve

# 不同指数/变量选用合适色带
CMAPS = {
    "NDVI": "YlGn", "SAVI": "YlGn", "WET": "BrBG", "NDWI": "Blues", "MNDWI": "Blues",
    "NDBI": "YlOrRd", "NDBSI": "YlOrBr", "RSEI": "RdYlGn", "LST": "magma",
    "Precipitation": "YlGnBu", "MeanTemperature": "coolwarm", "Deficit": "YlOrRd",
    "Elevation": "terrain", "Slope": "cividis",
}


def _extent(profile: dict, shape: tuple[int, int]) -> list[float]:
    if "transform" in profile:
        left, bottom, right, top = array_bounds(profile["height"], profile["width"], profile["transform"])
        return [left, right, bottom, top]
    return [0, shape[1], 0, shape[0]]


@tool(
    name="compose_map",
    description="把同一数据集或派生指数在若干年份的栅格拼成多面板专题对比图，共享统一色标，"
                "适合“把 X 的几期结果放在一张图上对比”“出一张多时相对比图”。最多 6 期。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "数据集 key，如 ndvi、lst、precip"},
            "index": {"type": "string", "enum": ["NDVI", "NDWI", "MNDWI", "NDBI", "SAVI", "WET", "NDBSI", "RSEI"],
                      "description": "改为绘制派生指数时填写，优先于 dataset"},
            "years": {"type": "array", "items": {"type": "integer"},
                      "description": "要对比的年份列表（按需取最接近的实际年份），最多 6 期"},
            "title": {"type": "string", "description": "自定义图名，可留空"},
        },
        "required": [],
    },
    category="制图",
)
def compose_map(ctx, dataset: str | None = None, index: str | None = None,
                years: list[int] | None = None, title: str | None = None) -> ToolResult:
    try:
        label, variable, unit, avail_years, fetch = _resolve(dataset, index)
    except (ValueError, FileNotFoundError) as exc:
        return ToolResult.failure(str(exc))

    if not avail_years:
        return ToolResult.failure(f"{label} 没有可用于制图的年份数据")

    if years:
        picks: list[int] = []
        for y in years:
            y = int(y)
            nearest = min(avail_years, key=lambda a: abs(a - y))
            if nearest not in picks:
                picks.append(nearest)
        if not picks:
            return ToolResult.failure("给定的年份无法匹配到任何数据期")
    else:
        picks = avail_years[:6]

    if len(picks) > 6:
        return ToolResult.failure(f"最多对比 6 期，当前给定 {len(picks)} 期")
    picks = sorted(picks)

    panels: list[tuple[int, np.ndarray, dict]] = []
    for year in picks:
        try:
            arr, profile = fetch(year)
        except (FileNotFoundError, ValueError):
            continue
        if not np.isfinite(arr).any():
            continue
        panels.append((year, arr, profile))
    if not panels:
        return ToolResult.failure(f"{label} 在给定年份没有任何有效数据可制图")

    ctx.report(f"绘制 {label} 多时相对比图（{len(panels)} 期）…", 40)

    combined = np.concatenate([p[1][np.isfinite(p[1])] for p in panels])
    vmin, vmax = float(np.percentile(combined, 2)), float(np.percentile(combined, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-12:
        vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined) + 1e-6)

    cmap = CMAPS.get(variable, "RdYlGn")
    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows + 0.8))
    axes = np.atleast_1d(axes).ravel()
    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = None
    for ax, (year, arr, profile) in zip(axes, panels):
        shown = np.where(np.isfinite(arr), arr, np.nan)
        im = ax.imshow(shown, cmap=cmap_obj, norm=norm, extent=_extent(profile, arr.shape), aspect="auto")
        ax.set_title(f"{year} 年", fontsize=11)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("经度 (°E)", fontsize=8)
        ax.set_ylabel("纬度 (°N)", fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title or f"{label} 多时相对比（{picks[0]}–{picks[-1]}）", fontsize=14)
    cbar = fig.colorbar(im, ax=list(axes[: len(panels)]), fraction=0.02, pad=0.015)
    cbar.set_label(unit or "", fontsize=9)
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.12)

    png = ctx.out_dir("figures") / f"{slugify(label)}_{'_'.join(map(str, picks))}_panels.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    summary = (
        f"已生成 {label} 多时相对比图（{len(panels)} 期：{'、'.join(str(y) for y in picks)}，单位 {unit}），"
        f"统一色标范围 {vmin:.4g}~{vmax:.4g}，便于跨年份横向比较。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={"target": label, "unit": unit, "years": picks, "vmin": round(vmin, 5), "vmax": round(vmax, 5)},
        artifacts=[ctx.artifact(png, "image", f"{label} 多时相对比图")],
    )
