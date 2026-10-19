"""栅格统计类工具：单/多年统计、区域时间序列、按土地利用分区统计。"""

from __future__ import annotations

import numpy as np

from ..catalog import catalog
from .base import (
    ToolResult,
    fmt,
    render_bars_png,
    render_lines_png,
    read_raster,
    slugify,
    stats_of,
    tool,
    write_csv,
)

LANDUSE_COLORS = {
    1: "#F5D76E",  # 耕地
    2: "#4E9F3D",  # 林地
    3: "#A8D08D",  # 草地
    4: "#4C8BF5",  # 水域
    5: "#D9534F",  # 城乡工矿居民用地
    6: "#D9B382",  # 未利用土地
    9: "#2C7FB8",  # 海洋
}


def _resolve_entry(dataset: str):
    entry = catalog.resolve(dataset)
    if entry is None:
        raise FileNotFoundError(f"目录中不存在数据集“{dataset}”")
    return entry


def _align_zone(zone_arr: np.ndarray, src_profile: dict, dst_profile: dict) -> np.ndarray:
    """把分类底图最近邻重采样到目标网格（分类数据只能最近邻，避免产生无效类别）。"""
    from rasterio.warp import Resampling, reproject

    src = np.where(np.isfinite(zone_arr), zone_arr, -1).astype("float32")
    dst = np.full((int(dst_profile["height"]), int(dst_profile["width"])), -1, dtype="float32")
    reproject(
        source=src, destination=dst,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        src_nodata=-1, dst_nodata=-1, resampling=Resampling.nearest,
    )
    out = dst.astype(np.float64)
    out[out < 0] = np.nan
    return out


@tool(
    name="raster_stats",
    description="计算某个数据集在指定年份（或全部年份）的统计特征：均值、中位数、极值、分位数、有效像元数。"
                "适合回答“平均是多少”“分布如何”“有没有变化趋势”这类问题。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "数据集 key，如 ndvi、lst、precip"},
            "year": {"type": "integer", "description": "单一年份，与 years 二选一"},
            "years": {"type": "array", "items": {"type": "integer"}, "description": "多个年份"},
        },
        "required": ["dataset"],
    },
    category="统计分析",
)
def raster_stats(ctx, dataset: str, year: int | None = None, years: list[int] | None = None) -> ToolResult:
    entry = _resolve_entry(dataset)
    if year is not None:
        targets = [int(year)]
    elif years:
        targets = [int(y) for y in years]
    else:
        targets = entry.available_years or [None]  # type: ignore[list-item]

    if not targets:
        return ToolResult.failure(f"{entry.key} 没有可用年份数据")

    rows: list[list] = []
    details: list[dict] = []
    for target in targets:
        path = entry.path_for(target)
        if path is None:
            continue
        arr, _, _ = read_raster(path)
        stat = stats_of(arr, entry.unit, {"year": target, "actual_year": entry.nearest_year(target) if target else None})
        details.append(stat)
        rows.append([
            target if target is not None else "静态", stat.get("mean"), stat.get("median"),
            stat.get("std"), stat.get("min"), stat.get("max"), stat.get("p5"), stat.get("p95"),
            stat.get("valid_pixels"),
        ])

    if not details:
        return ToolResult.failure(f"{entry.key} 在给定年份没有数据文件")

    artifacts = []
    if len(details) > 1:
        csv_path = ctx.out_dir("tables") / f"{slugify(entry.key)}_stats.csv"
        write_csv(csv_path, ["年份", "均值", "中位数", "标准差", "最小", "最大", "P5", "P95", "有效像元"], rows)
        artifacts.append(ctx.artifact(csv_path, "csv", f"{entry.title} 逐年统计表"))

    if len(details) == 1:
        stat = details[0]
        summary = (
            f"{entry.title} {stat.get('year')} 年统计（单位 {entry.unit}）：均值 {stat['mean']}，中位数 {stat['median']}，"
            f"标准差 {stat['std']}，范围 {stat['min']}~{stat['max']}，P5~P95 为 {stat['p5']}~{stat['p95']}，"
            f"有效像元 {stat['valid_pixels']:,}。"
        )
    else:
        means = [d["mean"] for d in details]
        first, last = details[0], details[-1]
        delta = last["mean"] - first["mean"]
        pct = (delta / first["mean"] * 100) if first["mean"] else float("nan")
        summary = (
            f"{entry.title} 共统计 {len(details)} 期（{first.get('year')}–{last.get('year')}，单位 {entry.unit}）："
            f"均值由 {first['mean']} 变为 {last['mean']}，变化 {delta:+.4f}（{pct:+.2f}%）；"
            f"区间 {min(means)}~{max(means)}。已输出逐年统计表。"
        )
    return ToolResult(ok=True, summary=summary, data={"dataset": entry.key, "unit": entry.unit, "stats": details},
                      artifacts=artifacts)


@tool(
    name="time_series",
    description="提取某个数据集在研究区内的逐年平均值时间序列，并绘制折线图、导出 CSV。"
                "适合回答“2000-2025 年 X 如何演变”这类趋势性问题（描述性趋势，非逐像元显著性检验）。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "数据集 key"},
            "start_year": {"type": "integer", "description": "起始年份，默认数据最早年份"},
            "end_year": {"type": "integer", "description": "结束年份，默认数据最晚年份"},
        },
        "required": ["dataset"],
    },
    category="统计分析",
)
def time_series(ctx, dataset: str, start_year: int | None = None, end_year: int | None = None) -> ToolResult:
    entry = _resolve_entry(dataset)
    years = [y for y in entry.available_years
             if (start_year is None or y >= start_year) and (end_year is None or y <= end_year)]
    if len(years) < 2:
        return ToolResult.failure(f"{entry.key} 在 {start_year}-{end_year} 区间内不足 2 期数据，无法绘制时间序列")

    years_out: list[int] = []
    means: list[float] = []
    for y in years:
        arr, _, _ = read_raster(entry.path_for(y))
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            continue
        years_out.append(y)
        means.append(round(float(vals.mean()), 4))

    png = ctx.out_dir("figures") / f"{slugify(entry.key)}_timeseries.png"
    render_lines_png(
        years_out, {entry.title: means}, png,
        title=f"{entry.title} 区域平均时间序列",
        xlabel="年份", ylabel=f"{entry.variable}（{entry.unit}）",
        note="区域均值按全部有效像元计算",
    )
    csv_path = ctx.out_dir("tables") / f"{slugify(entry.key)}_timeseries.csv"
    write_csv(csv_path, ["年份", f"{entry.variable}({entry.unit})"], [[y, m] for y, m in zip(years_out, means)])

    change = means[-1] - means[0]
    pct = change / means[0] * 100 if means[0] else float("nan")
    slope = float(np.polyfit(years_out, means, 1)[0]) if len(years_out) >= 3 else float("nan")
    summary = (
        f"{entry.title}：{years_out[0]}–{years_out[-1]} 年区域均值由 {means[0]} 变为 {means[-1]}"
        f"（{change:+.4f}，{pct:+.2f}%），线性倾向 {slope:+.5f}/年；区间 {min(means)}~{max(means)}。"
        f"已输出折线图与 CSV。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={"dataset": entry.key, "unit": entry.unit, "years": years_out, "mean": means,
              "linear_slope_per_year": round(slope, 6) if np.isfinite(slope) else None,
              "change_abs": round(change, 4), "change_pct": round(pct, 2)},
        artifacts=[ctx.artifact(png, "image", f"{entry.title} 时间序列"), ctx.artifact(csv_path, "csv", f"{entry.title} 时间序列数据")],
    )


@tool(
    name="zonal_by_landuse",
    description="以土地利用一级类（耕地/林地/草地/水域/城乡工矿居民用地/未利用土地）为分区，"
                "统计某数据集在各分区内的均值与面积占比，并输出对比柱状图。"
                "适合回答“哪类地表的 X 最高/最低”“不同地类差异如何”。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "待统计数据集 key，如 ndvi、lst、deficit"},
            "year": {"type": "integer", "description": "统计年份"},
            "zone_year": {"type": "integer", "description": "分区底图年份，默认取与 year 最接近的期"},
        },
        "required": ["dataset"],
    },
    category="统计分析",
)
def zonal_by_landuse(ctx, dataset: str, year: int | None = None, zone_year: int | None = None) -> ToolResult:
    entry = _resolve_entry(dataset)
    zone = catalog.get("landuse")
    if zone is None:
        return ToolResult.failure("未找到土地利用分区底图（landuse），无法执行分区统计")

    year = entry.nearest_year(year) if entry.years else None
    if entry.years and year is None:
        return ToolResult.failure(f"{entry.key} 无可用年份")
    var_path = entry.path_for(year)
    zone_target = zone.nearest_year(zone_year or year)
    zone_path = zone.path_for(zone_target)
    if var_path is None or zone_path is None:
        return ToolResult.failure("缺少待统计数据或分区底图文件")

    var_arr, var_profile, _ = read_raster(var_path)
    zone_arr, zone_profile, _ = read_raster(zone_path)
    if var_arr.shape != zone_arr.shape:
        ctx.report(f"分区底图分辨率不一致，最近邻重采样到 {entry.title} 网格…", 20)
        zone_arr = _align_zone(zone_arr, zone_profile, var_profile)

    labels = zone.classes or {}
    stats_rows: list[list] = []
    chart_labels: list[str] = []
    chart_values: list[float] = []
    chart_colors: list[str] = []
    details: list[dict] = []
    total_valid = int(np.isfinite(var_arr).sum())

    for code, name in labels.items():
        mask = (zone_arr == code) & np.isfinite(var_arr)
        count = int(mask.sum())
        if count == 0:
            continue
        vals = var_arr[mask]
        mean = float(vals.mean())
        stats_rows.append([name, round(mean, 4), round(float(np.median(vals)), 4),
                           round(float(vals.std()), 4), count, round(count / total_valid * 100, 2)])
        chart_labels.append(name)
        chart_values.append(mean)
        chart_colors.append(LANDUSE_COLORS.get(code, "#999999"))
        details.append({"class_code": code, "class_name": name, "mean": round(mean, 4),
                        "median": round(float(np.median(vals)), 4), "pixels": count,
                        "area_pct": round(count / total_valid * 100, 2)})

    if not details:
        return ToolResult.failure("分区统计未得到有效结果")

    tag = f"{entry.key}_{year}" if year else entry.key
    png = ctx.out_dir("figures") / f"zonal_{slugify(tag)}.png"
    render_bars_png(chart_labels, chart_values, png,
                    title=f"{entry.title} 各地类均值对比（{year if year is not None else '静态'}）",
                    ylabel=f"{entry.variable}（{entry.unit}）",
                    note=f"分区底图：土地利用一级类 {zone_target}", colors=chart_colors)
    csv_path = ctx.out_dir("tables") / f"zonal_{slugify(tag)}.csv"
    write_csv(csv_path, ["地类", "均值", "中位数", "标准差", "像元数", "占有效区%"], stats_rows)

    ranked = sorted(details, key=lambda d: d["mean"], reverse=True)
    top, bottom = ranked[0], ranked[-1]
    summary = (
        f"{entry.title}（{year if year is not None else '静态'}，单位 {entry.unit}）分地类统计："
        f"最高为{top['class_name']}（{top['mean']}，占 {top['area_pct']}%），最低为{bottom['class_name']}（{bottom['mean']}，占 {bottom['area_pct']}%）。"
        f"已输出柱状图与分区统计表。"
    )
    return ToolResult(ok=True, summary=summary,
                      data={"dataset": entry.key, "year": year, "zone_year": zone_target,
                            "unit": entry.unit, "zonal": details},
                      artifacts=[ctx.artifact(png, "image", f"{entry.title} 分地类对比"),
                                 ctx.artifact(csv_path, "csv", f"{entry.title} 分区统计表")])
