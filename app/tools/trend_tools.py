"""趋势与变化检测类工具。

1. trend_analysis —— 逐像元 Mann-Kendall 趋势检验 + Sen's slope，输出 9 级趋势分级；
2. change_detection —— 两期栅格差分，输出变化量与 5 级变化分级。

Mann-Kendall / Sen's slope 逻辑与 `实验/rsei/calc_rsei_trend.py` 保持一致（含 tie 修正、
以真实年份差为分母），但去掉了 GPU 依赖并加入交互友好的抽稀与批处理。
趋势分级采用“显著性 p + 相对变化幅度”的 9 级方案，幅度阈值以 2%-98% 分位区间归一化，
因此对不同量纲的数据集（NDVI、LST、降水、RSEI 等）都适用。
"""

from __future__ import annotations

import numpy as np

from ..catalog import catalog
from .base import (
    NODATA,
    ToolResult,
    read_raster,
    render_class_map_png,
    render_raster_png,
    save_geotiff,
    slugify,
    tool,
    write_csv,
)
from .index_tools import INDEX_SPECS, _compute_rsei, _reflectance_bands, _simple_index

# 9 级趋势分级（与 calc_rsei_trend.py 的语义一致）
TREND_LABELS = {
    1: "极强改善", 2: "强改善", 3: "中等改善", 4: "弱改善", 5: "无显著变化",
    6: "弱退化", 7: "中等退化", 8: "强退化", 9: "极强退化",
}
TREND_COLORS = {
    1: "#1A9850", 2: "#66BD63", 3: "#A6D96A", 4: "#D9EF8B", 5: "#FFFFBF",
    6: "#FEE08B", 7: "#FDAE61", 8: "#F46D43", 9: "#D73027",
}
# 归一化斜率阈值（每年），沿用 RSEI 趋势分析的经验阈值
S_STRONG, S_MID, S_WEAK = 0.005, 0.002, 0.001

CHANGE_LABELS = {1: "显著减少", 2: "减少", 3: "基本不变", 4: "增加", 5: "显著增加"}
CHANGE_COLORS = {1: "#B2182B", 2: "#EF8A62", 3: "#F7F7F7", 4: "#67A9CF", 5: "#2166AC"}


def _resolve(dataset: str | None, index: str | None):
    """确定分析对象：优先指数，其次数据集。返回 (标签, 变量名, 单位, 年份序列, 取数函数)。"""
    if index:
        name = index.strip().upper()
        if name not in INDEX_SPECS:
            raise ValueError(f"不支持的指数“{index}”，可选：{', '.join(INDEX_SPECS)}")
        # 若目录中已有同名数据集（目前仅 NDVI），直接复用该数据集。否则同一问题会因为
        # 模型选择 index 还是 dataset 而落到两套数据源（目录里的年最大值合成产品和由
        # reflectance 现算的结果），给出不一致的数字。
        entry = catalog.get(name.lower())
        if entry is not None and not entry.scenario_files and entry.years:
            def fetch_dataset(year: int):
                arr, profile, _ = read_raster(entry.path_for(year))
                return arr, profile

            return entry.key, entry.variable, entry.unit, entry.available_years, fetch_dataset
        if name == "RSEI":
            ndvi_entry, lst_entry = catalog.get("ndvi"), catalog.get("lst")
            if ndvi_entry is None or lst_entry is None:
                raise FileNotFoundError("计算 RSEI 需要 NDVI 与 LST 数据集")
            years = sorted(set(ndvi_entry.available_years) & set(lst_entry.available_years))

            def fetch(year: int):
                arr, profile, _ = _compute_rsei(year)
                return arr, profile
        else:
            entry = catalog.get("reflectance")
            if entry is None:
                raise FileNotFoundError("计算光谱指数需要反射率数据（reflectance）")
            years = entry.available_years

            def fetch(year: int):
                bands, profile, _ = _reflectance_bands(year)
                return _simple_index(name, bands), profile

        return name, name, INDEX_SPECS[name]["unit"], years, fetch

    if not dataset:
        raise ValueError("请指定 dataset 或 index")

    entry = catalog.resolve(dataset)
    if entry is None:
        raise FileNotFoundError(f"目录中不存在数据集“{dataset}”")
    if entry.kind == "categorical":
        raise ValueError(f"{entry.title} 为分类数据，不适合做趋势/变化检测，请改用面积转移分析")
    if entry.scenario_files and not entry.years:
        raise ValueError(f"{entry.title} 为情景数据（无连续年份），请改用情景对比工具")
    if not entry.years:
        raise ValueError(f"{entry.title} 为静态数据（单期），无法做趋势分析")

    def fetch(year: int):
        arr, profile, _ = read_raster(entry.path_for(year))
        return arr, profile

    return entry.key, entry.variable, entry.unit, entry.available_years, fetch


def _collect_stack(fetch, years: list[int]) -> tuple[np.ndarray, dict, list[int]]:
    arrays: list[np.ndarray] = []
    profile: dict | None = None
    used: list[int] = []
    shape: tuple[int, int] | None = None
    for year in years:
        try:
            arr, prof = fetch(year)
        except (FileNotFoundError, ValueError):
            continue
        if not np.isfinite(arr).any():
            continue
        if shape is None:
            shape, profile = arr.shape, prof
        elif arr.shape != shape:
            continue
        arrays.append(arr.astype(np.float32))
        used.append(int(year))
    if len(arrays) < 5 or profile is None:
        raise ValueError(f"有效年份不足 5 期（实际 {len(arrays)} 期），无法做趋势分析")
    return np.stack(arrays, axis=0), profile, used


def _mann_kendall_sen(series: np.ndarray, years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """逐像元 MK 检验 + Sen's slope。series 形状 (n_years, n_pixels)。

    返回 (slope, pvalue)，均为 (n_pixels,)。p 为双尾 p 值，含 tie 修正。
    """
    from scipy.special import erfc

    n_years, n_pixels = series.shape
    year_arr = np.asarray(years, dtype=np.float64)
    iu, ju = np.triu_indices(n_years, k=1)
    year_diff = year_arr[ju] - year_arr[iu]

    # [A] 缺失值线性插值（按真实年份）
    nan_mask = ~np.isfinite(series)
    rows_with_nan = np.where(nan_mask.any(axis=0))[0]
    for col in rows_with_nan:
        row = series[:, col]
        good = np.isfinite(row)
        if not good.any():
            series[:, col] = 0.0
            continue
        row[~good] = np.interp(year_arr[~good], year_arr[good], row[good])

    base_var = n_years * (n_years - 1) * (2 * n_years + 5) / 18.0

    # [B+C] 排序后用 run-length 编码一次性算出所有并列（tie）修正量：
    # 原实现对每个含并列的像元各调用一次 np.unique，是整段代码最慢的环节。
    sorted_series = np.sort(series, axis=0)
    new_run = np.empty(sorted_series.shape, dtype=bool)
    new_run[0] = True
    np.not_equal(sorted_series[1:], sorted_series[:-1], out=new_run[1:])
    row_no = np.arange(n_years, dtype=np.int64)[:, None]
    run_start = np.where(new_run, row_no, 0)
    np.maximum.accumulate(run_start, axis=0, out=run_start)   # 向前填充每个 run 的起点
    run_len = (row_no + 1) - run_start
    end_of_run = np.empty(sorted_series.shape, dtype=bool)
    end_of_run[-1] = True
    end_of_run[:-1] = new_run[1:]                             # run 结束处 = 下一元素开启新 run
    lengths = np.where(end_of_run, run_len, 0).astype(np.float64)
    tie_correction = (lengths * (lengths - 1) * (2 * lengths + 5) / 18.0).sum(axis=0)
    del sorted_series, new_run, run_start, run_len, end_of_run, lengths

    # [D] 逐年份对累加符号和：325 次向量减法，避免生成 (n_pairs, n_pixels) 大矩阵
    s_stat = np.zeros(n_pixels, dtype=np.float64)
    tmp = np.empty(n_pixels, dtype=np.float64)
    for k in range(iu.size):
        np.subtract(series[ju[k]], series[iu[k]], out=tmp)
        np.sign(tmp, out=tmp)
        s_stat += tmp
    del tmp

    var_s = base_var - tie_correction
    sqrt_var = np.sqrt(np.maximum(var_s, 1e-10))
    z_stat = np.where(s_stat > 0, (s_stat - 1) / sqrt_var,
                      np.where(s_stat < 0, (s_stat + 1) / sqrt_var, 0.0))
    pvalue = erfc(np.abs(z_stat) / np.sqrt(2.0))

    # [E] Sen 斜率中位数：分块 + np.partition 只取中位所在的两个位置，
    # 结果与 np.median 逐位一致，但省去整块排序。
    n_pairs = iu.size
    half = n_pairs // 2
    kth = [half] if n_pairs % 2 else [half - 1, half]
    slope = np.empty(n_pixels, dtype=np.float64)
    batch = 50_000
    for start in range(0, n_pixels, batch):
        end = min(start + batch, n_pixels)
        diffs = np.empty((n_pairs, end - start), dtype=np.float64)
        for k in range(n_pairs):
            np.subtract(series[ju[k], start:end], series[iu[k], start:end], out=diffs[k])
        diffs /= year_diff[:, None]
        diffs.partition(kth, axis=0)
        slope[start:end] = diffs[half] if n_pairs % 2 else (diffs[half - 1] + diffs[half]) * 0.5

    return slope, pvalue


def _split_range(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 1.0
    p2, p98 = np.percentile(vals, [2, 98])
    span = float(p98 - p2)
    return span if span > 1e-12 else float(np.nanstd(vals)) or 1.0


def _classify_trend(norm_slope: np.ndarray, pvalue: np.ndarray) -> np.ndarray:
    """按显著性 + 归一化幅度划分 9 级趋势。"""
    cls = np.full(norm_slope.shape, 5, dtype=np.int8)
    mag = np.abs(norm_slope)
    pos, neg = norm_slope > 0, norm_slope < 0

    sig_weak = pvalue >= 0.05
    cls[sig_weak & pos & (mag > S_WEAK)] = 4
    cls[sig_weak & neg & (mag > S_WEAK)] = 6

    sig_mid = (pvalue >= 0.01) & (pvalue < 0.05)
    cls[sig_mid & pos] = 3
    cls[sig_mid & neg] = 7

    sig_high = pvalue < 0.01
    cls[sig_high & (mag > S_STRONG) & pos] = 1
    cls[sig_high & (mag > S_STRONG) & neg] = 9
    cls[sig_high & (mag > S_MID) & (mag <= S_STRONG) & pos] = 2
    cls[sig_high & (mag > S_MID) & (mag <= S_STRONG) & neg] = 8
    cls[sig_high & (mag <= S_MID) & pos] = 3
    cls[sig_high & (mag <= S_MID) & neg] = 7
    return cls


def _save_int8(arr: np.ndarray, profile: dict, path, nodata: int = 0) -> None:
    import rasterio

    out_profile = profile.copy()
    out_profile.update(dtype="int8", count=1, nodata=nodata, compress="lzw")
    data = np.where(np.isfinite(arr), arr, nodata).astype("int8")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)


@tool(
    name="trend_analysis",
    description="逐像元 Mann-Kendall 趋势检验 + Sen's slope，判断每个像元在研究期内是改善还是退化、是否显著，"
                "输出 9 级趋势分级图、Sen 斜率图与统计表。适合回答“X 在 2000-2025 年哪里变好/变差、是否显著”。"
                "可分析目录数据集（dataset），也可分析派生指数（index，如 RSEI/NDVI）。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "数据集 key，如 ndvi、lst、precip、deficit"},
            "index": {"type": "string", "enum": ["NDVI", "NDWI", "MNDWI", "NDBI", "SAVI", "WET", "NDBSI", "RSEI"],
                      "description": "改为分析派生指数时填写，优先于 dataset"},
            "start_year": {"type": "integer", "description": "起始年份"},
            "end_year": {"type": "integer", "description": "结束年份"},
        },
        "required": [],
    },
    category="趋势分析",
)
def trend_analysis(ctx, dataset: str | None = None, index: str | None = None,
                   start_year: int | None = None, end_year: int | None = None) -> ToolResult:
    try:
        label, variable, unit, years, fetch = _resolve(dataset, index)
    except (ValueError, FileNotFoundError) as exc:
        return ToolResult.failure(str(exc))

    years = [y for y in years
             if (start_year is None or y >= start_year) and (end_year is None or y <= end_year)]
    if len(years) < 5:
        return ToolResult.failure(f"{label} 在给定区间内不足 5 期数据，无法做趋势分析")

    ctx.report(f"读取 {label} 共 {len(years)} 期数据…", 10)
    try:
        stack, profile, used = _collect_stack(fetch, years)
    except ValueError as exc:
        return ToolResult.failure(str(exc))

    n_years, rows, cols = stack.shape
    valid_count = np.isfinite(stack).sum(axis=0)
    valid_mask = valid_count >= int(np.ceil(n_years * 0.5))
    n_valid = int(valid_mask.sum())
    if n_valid < 100:
        return ToolResult.failure(f"{label} 有效像元不足 100，无法做趋势分析")

    # 抽稀：控制单次计算规模，保证交互响应
    limit = int(ctx.settings.trend_max_pixels)
    row_idx, col_idx = np.nonzero(valid_mask)
    if n_valid > limit:
        rng = np.random.default_rng(42)
        pick = rng.choice(n_valid, size=limit, replace=False)
        row_idx, col_idx = row_idx[pick], col_idx[pick]
        n_valid = limit
        sampled = True
    else:
        sampled = False

    ctx.report(f"逐像元 Mann-Kendall 检验（{n_valid:,} 像元）…", 40)
    series = stack[:, row_idx, col_idx].astype(np.float64)
    slope, pvalue = _mann_kendall_sen(series, used)

    value_range = _split_range(stack)
    norm_slope = slope / value_range
    trend_cls = _classify_trend(norm_slope, pvalue)

    slope_map = np.full((rows, cols), np.nan)
    pvalue_map = np.full((rows, cols), np.nan)
    class_map = np.full((rows, cols), np.nan)
    slope_map[row_idx, col_idx] = slope
    pvalue_map[row_idx, col_idx] = pvalue
    class_map[row_idx, col_idx] = trend_cls.astype(np.float64)

    ctx.report("趋势计算完成，正在制图…", 80)
    tag = f"{slugify(label)}_{used[0]}_{used[-1]}_trend"
    class_png = ctx.out_dir("figures") / f"{tag}_class.png"
    render_class_map_png(
        class_map, profile, class_png,
        title=f"{label} 变化趋势分级（{used[0]}–{used[-1]}）",
        labels=TREND_LABELS, colors=TREND_COLORS,
        note=f"Mann-Kendall 检验（α=0.05）+ Sen's slope；幅度以 2%-98% 分位区间归一化"
             + ("；已随机抽稀" if sampled else ""),
    )
    slope_png = ctx.out_dir("figures") / f"{tag}_slope.png"
    lim = float(np.nanpercentile(np.abs(slope), 98)) or 1e-6
    render_raster_png(
        slope_map, profile, slope_png,
        title=f"{label} Sen's slope（{used[0]}–{used[-1]}）",
        unit=f"{unit}/年", cmap="RdYlGn", vmin=-lim, vmax=lim,
        note="正值表示改善/升高，负值表示退化/降低",
    )
    class_tif = ctx.out_dir("rasters") / f"{tag}_class.tif"
    _save_int8(class_map, profile, class_tif)
    slope_tif = ctx.out_dir("rasters") / f"{tag}_slope.tif"
    save_geotiff(slope_map, profile, slope_tif)
    pvalue_tif = ctx.out_dir("rasters") / f"{tag}_pvalue.tif"
    save_geotiff(pvalue_map, profile, pvalue_tif)

    counts = {code: int(np.sum(trend_cls == code)) for code in TREND_LABELS}
    improve = sum(counts[c] for c in (1, 2, 3, 4))
    degrade = sum(counts[c] for c in (6, 7, 8, 9))
    stable = counts[5]
    significant = int(np.sum(pvalue < 0.05))
    rows_csv = [[TREND_LABELS[c], counts[c], round(counts[c] / n_valid * 100, 2)] for c in TREND_LABELS]
    rows_csv.append(["合计", n_valid, 100.0])
    csv_path = ctx.out_dir("tables") / f"{tag}.csv"
    write_csv(csv_path, ["趋势等级", "像元数", "占比(%)"], rows_csv)

    summary = (
        f"{label} {used[0]}–{used[-1]}（{n_years} 期，单位 {unit}）逐像元趋势："
        f"改善像元占 {improve / n_valid * 100:.1f}%（显著 {sum(counts[c] for c in (1, 2, 3))} 个），"
        f"退化像元占 {degrade / n_valid * 100:.1f}%（显著 {sum(counts[c] for c in (7, 8, 9))} 个），"
        f"无显著变化 {stable / n_valid * 100:.1f}%；通过 0.05 显著性检验的像元占 {significant / n_valid * 100:.1f}%。"
        f"Sen 斜率中位数 {np.median(slope):+.5f} {unit}/年。已输出趋势分级图、斜率图与统计表。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={
            "target": label, "variable": variable, "unit": unit,
            "years": [used[0], used[-1]], "valid_pixels": n_valid, "sampled": sampled,
            "improve_pct": round(improve / n_valid * 100, 2),
            "degrade_pct": round(degrade / n_valid * 100, 2),
            "stable_pct": round(stable / n_valid * 100, 2),
            "significant_pct": round(significant / n_valid * 100, 2),
            "median_sen_slope_per_year": round(float(np.median(slope)), 6),
            "classes": {TREND_LABELS[c]: counts[c] for c in TREND_LABELS},
        },
        artifacts=[
            ctx.artifact(class_png, "image", f"{label} 趋势分级图"),
            ctx.artifact(slope_png, "image", f"{label} Sen 斜率图"),
            ctx.artifact(class_tif, "tif", f"{label} 趋势分级栅格"),
            ctx.artifact(slope_tif, "tif", f"{label} Sen 斜率栅格"),
            ctx.artifact(pvalue_tif, "tif", f"{label} 显著性 p 值栅格"),
            ctx.artifact(csv_path, "csv", f"{label} 趋势分级统计表"),
        ],
    )


@tool(
    name="change_detection",
    description="对比某数据集（或派生指数）在两个年份之间的差异，输出变化量图、5 级变化分级图与面积统计，"
                "适合回答“从 A 年到 B 年 X 增加/减少了多少、哪里变化最大”。",
    parameters={
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "数据集 key"},
            "index": {"type": "string", "enum": ["NDVI", "NDWI", "MNDWI", "NDBI", "SAVI", "WET", "NDBSI", "RSEI"],
                      "description": "改为对比派生指数时填写，优先于 dataset"},
            "start_year": {"type": "integer", "description": "基准年"},
            "end_year": {"type": "integer", "description": "对比年"},
        },
        "required": ["start_year", "end_year"],
    },
    category="趋势分析",
)
def change_detection(ctx, start_year: int, end_year: int,
                     dataset: str | None = None, index: str | None = None) -> ToolResult:
    if int(start_year) == int(end_year):
        return ToolResult.failure("起始年与结束年相同，无法做变化检测")
    try:
        label, variable, unit, years, fetch = _resolve(dataset, index)
    except (ValueError, FileNotFoundError) as exc:
        return ToolResult.failure(str(exc))

    def nearest(year: int) -> int:
        return min(years, key=lambda y: abs(y - year))

    y0, y1 = nearest(int(start_year)), nearest(int(end_year))
    if y0 == y1:
        return ToolResult.failure(f"{label} 在 {start_year} 与 {end_year} 次年没有可区分的数据期")

    ctx.report(f"读取 {label} {y0} 与 {y1} 年数据…", 25)
    arr0, profile = fetch(y0)
    arr1, _ = fetch(y1)
    if arr0.shape != arr1.shape:
        return ToolResult.failure(f"两期栅格网格不一致（{arr0.shape} vs {arr1.shape}），无法做像元级差分")

    diff = arr1 - arr0
    valid = np.isfinite(diff)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return ToolResult.failure("两期共同有效像元不足 100，无法做变化检测")

    span = _split_range(np.stack([arr0, arr1]))
    thr = 0.05 * span
    cls = np.full(diff.shape, np.nan)
    cls[valid] = 3
    cls[valid & (diff > thr)] = 4
    cls[valid & (diff > 3 * thr)] = 5
    cls[valid & (diff < -thr)] = 2
    cls[valid & (diff < -3 * thr)] = 1

    ctx.report("正在生成变化图与统计…", 70)
    tag = f"{slugify(label)}_{y0}_{y1}_change"
    diff_png = ctx.out_dir("figures") / f"{tag}_diff.png"
    lim = float(np.nanpercentile(np.abs(diff[valid]), 98)) or 1e-6
    render_raster_png(
        diff, profile, diff_png,
        title=f"{label} 变化量（{y0} → {y1}）",
        unit=f"{variable} 变化量（{unit}）", cmap="RdYlBu", vmin=-lim, vmax=lim,
        note="暖色为减少，冷色为增加",
    )
    class_png = ctx.out_dir("figures") / f"{tag}_class.png"
    render_class_map_png(
        cls, profile, class_png,
        title=f"{label} 变化分级（{y0} → {y1}）",
        labels=CHANGE_LABELS, colors=CHANGE_COLORS,
        note=f"以 2%-98% 分位区间的 5% 与 15% 作为变化阈值（{thr:.4g} / {3 * thr:.4g} {unit}）",
    )
    diff_tif = ctx.out_dir("rasters") / f"{tag}_diff.tif"
    save_geotiff(diff, profile, diff_tif)

    counts = {code: int(np.sum(cls == code)) for code in CHANGE_LABELS}
    rows_csv = [[CHANGE_LABELS[c], counts[c], round(counts[c] / n_valid * 100, 2)] for c in CHANGE_LABELS]
    avg = float(np.nanmean(diff))
    changed = counts[1] + counts[2] + counts[4] + counts[5]
    rows_csv.append(["变化合计", changed, round(changed / n_valid * 100, 2)])
    csv_path = ctx.out_dir("tables") / f"{tag}.csv"
    write_csv(csv_path, ["变化等级", "像元数", "占比(%)"], rows_csv)

    summary = (
        f"{label} {y0}→{y1} 年变化（单位 {unit}）：平均变化 {avg:+.4f}，"
        f"增加区占 {((counts[4] + counts[5]) / n_valid * 100):.1f}%，减少区占 {((counts[1] + counts[2]) / n_valid * 100):.1f}%，"
        f"基本不变占 {counts[3] / n_valid * 100:.1f}%。已输出变化量图、变化分级图与统计表。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={
            "target": label, "unit": unit, "years": [y0, y1],
            "mean_change": round(avg, 4),
            "increase_pct": round((counts[4] + counts[5]) / n_valid * 100, 2),
            "decrease_pct": round((counts[1] + counts[2]) / n_valid * 100, 2),
            "stable_pct": round(counts[3] / n_valid * 100, 2),
            "classes": {CHANGE_LABELS[c]: counts[c] for c in CHANGE_LABELS},
        },
        artifacts=[
            ctx.artifact(diff_png, "image", f"{label} 变化量图（{y0}→{y1}）"),
            ctx.artifact(class_png, "image", f"{label} 变化分级图（{y0}→{y1}）"),
            ctx.artifact(diff_tif, "tif", f"{label} 变化量栅格（{y0}→{y1}）"),
            ctx.artifact(csv_path, "csv", f"{label} 变化面积统计表"),
        ],
    )
