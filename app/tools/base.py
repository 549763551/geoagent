"""工具层公共基础设施：工具协议、注册表、栅格/绘图通用函数。"""

from __future__ import annotations

import functools
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # 无界面后端，服务端渲染

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import array_bounds

from ..config import Settings, settings as default_settings
from ..memory import Artifact, SessionMemory

logger = logging.getLogger("geoagent.tools")

# 中文字体，避免图上出现方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

NODATA = -9999.0
ProgressFn = Callable[[str, float], None]


# ======================================================================
# 结果与工具协议
# ======================================================================
@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)

    @classmethod
    def failure(cls, message: str) -> "ToolResult":
        return cls(ok=False, summary=message, data={"error": message})


@dataclass
class ToolContext:
    """工具执行上下文。"""

    settings: Settings
    session: SessionMemory
    progress: ProgressFn | None = None

    def out_dir(self, *parts: str) -> Path:
        path = self.settings.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def report(self, text: str, pct: float = -1.0) -> None:
        if self.progress:
            self.progress(text, pct)

    def artifact(self, path: Path, kind: str, title: str, **meta: Any) -> Artifact:
        rel = path.relative_to(self.settings.output_dir).as_posix()
        return Artifact(type=kind, path=str(path), title=title, url=f"/files/{rel}", meta=meta)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    category: str = "通用"

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：统一管理 JSON Schema 与执行入口。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._order: list[str] = []

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        if spec.name not in self._order:
            self._order.append(spec.name)
        return spec

    def tool(self, name: str, description: str, parameters: dict[str, Any], category: str = "通用"):
        def decorator(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            self.register(ToolSpec(name=name, description=description, parameters=parameters,
                                   handler=func, category=category))
            return func

        return decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._order)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self._order]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in self.specs()]

    def catalog_text(self) -> str:
        lines = []
        for spec in self.specs():
            required = spec.parameters.get("required", [])
            params = ", ".join(
                f"{k}{'*' if k in required else ''}" for k in spec.parameters.get("properties", {})
            )
            lines.append(f"- {spec.name}({params})：{spec.description}")
        return "\n".join(lines)

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult.failure(f"未知工具：{name}")
        allowed = set(spec.parameters.get("properties", {}))
        clean = {k: v for k, v in (args or {}).items() if k in allowed}
        try:
            return spec.handler(ctx, **clean)
        except TypeError as exc:
            return ToolResult.failure(f"参数不合法（{exc}），请检查 {name} 的入参")
        except FileNotFoundError as exc:
            return ToolResult.failure(f"数据文件缺失：{exc}")
        except Exception as exc:  # pragma: no cover - 兜底
            logger.error("工具 %s 执行失败: %s\n%s", name, exc, traceback.format_exc())
            return ToolResult.failure(f"{name} 执行异常：{exc}")


registry = ToolRegistry()


def tool(name: str, description: str, parameters: dict[str, Any], category: str = "通用"):
    return registry.tool(name, description, parameters, category)


# ======================================================================
# 栅格通用函数
# ======================================================================
def read_raster(path: Path | str, band: int = 1) -> tuple[np.ndarray, dict[str, Any], float]:
    """读取单波段，nodata 转为 NaN。返回 (数组, profile, nodata)。"""
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else NODATA
    arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr, profile, float(nodata)


def read_bands(path: Path | str) -> tuple[list[np.ndarray], dict[str, Any], float]:
    """读取多波段，返回波段数组列表。"""
    with rasterio.open(path) as src:
        bands = [src.read(i).astype(np.float64) for i in range(1, src.count + 1)]
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else NODATA
    for arr in bands:
        arr[arr == nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
    return bands, profile, float(nodata)


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) < eps, np.nan, a / b)
    return out


def percentile_stretch(arr: np.ndarray, valid: np.ndarray | None = None,
                       low: float = 2, high: float = 98) -> np.ndarray:
    """2%-98% 分位拉伸到 [0,1]，非有限值保持 NaN。"""
    mask = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    vals = arr[mask]
    if vals.size == 0:
        return np.full_like(arr, np.nan, dtype=np.float64)
    p_low, p_high = np.percentile(vals, low), np.percentile(vals, high)
    if p_high - p_low < 1e-12:
        return np.where(mask, 0.0, np.nan)
    out = (arr - p_low) / (p_high - p_low)
    return np.clip(out, 0.0, 1.0)


def stats_of(arr: np.ndarray, unit: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """通用描述统计。"""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {"error": "无有效像元", "unit": unit}
    result = {
        "unit": unit,
        "valid_pixels": int(vals.size),
        "mean": round(float(np.mean(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
        "p5": round(float(np.percentile(vals, 5)), 4),
        "p25": round(float(np.percentile(vals, 25)), 4),
        "p75": round(float(np.percentile(vals, 75)), 4),
        "p95": round(float(np.percentile(vals, 95)), 4),
    }
    if extra:
        result.update(extra)
    return result


def save_geotiff(arr: np.ndarray, profile: dict[str, Any], path: Path, nodata: float = NODATA) -> Path:
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    data = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)
    return path


def stack_years(paths: list[Path]) -> tuple[np.ndarray, dict[str, Any]]:
    """按年份顺序读取并堆叠为 (n, rows, cols)。要求网格一致。"""
    arrays: list[np.ndarray] = []
    profile: dict[str, Any] | None = None
    shape: tuple[int, int] | None = None
    for path in paths:
        arr, prof, _ = read_raster(path)
        if shape is None:
            shape, profile = arr.shape, prof
        elif arr.shape != shape:
            raise ValueError(f"栅格网格不一致：{path.name} {arr.shape} != {shape}")
        arrays.append(arr)
    if not arrays or profile is None:
        raise ValueError("没有可用的栅格数据")
    return np.stack(arrays, axis=0), profile


# ======================================================================
# 绘图通用函数
# ======================================================================
def render_raster_png(
    arr: np.ndarray,
    profile: dict[str, Any],
    path: Path,
    title: str,
    unit: str = "",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    note: str = "",
) -> Path:
    """栅格 → PNG 专题图（带经纬度坐标轴与色标）。"""
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError("该栅格没有有效像元，无法出图")
    shown = np.where(valid, arr, np.nan)
    if vmin is None or vmax is None:
        vals = shown[valid]
        vmin = float(np.percentile(vals, 2)) if vmin is None else vmin
        vmax = float(np.percentile(vals, 98)) if vmax is None else vmax

    if "transform" in profile:
        left, bottom, right, top = array_bounds(profile["height"], profile["width"], profile["transform"])
        extent = [left, right, bottom, top]
    else:
        extent = [0, arr.shape[1], 0, arr.shape[0]]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    # 必须用 nearest：栅格行列数远大于成图像素，默认插值会把含 NaN 的邻域加权平均，
    # 导致整幅图被冲淡成近乎空白（尤其分类/稀疏有效像元场景）。
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto",
                   interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("经度 (°E)", fontsize=10)
    ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(unit or "", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    if note:
        ax.text(0.01, -0.16, note, transform=ax.transAxes, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_class_map_png(
    arr: np.ndarray,
    profile: dict[str, Any],
    path: Path,
    title: str,
    labels: dict[int, str],
    colors: dict[int, str],
    note: str = "",
) -> Path:
    """离散分类栅格 → PNG。"""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    codes = sorted(labels)
    cmap = ListedColormap([colors.get(c, "#cccccc") for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)

    if "transform" in profile:
        left, bottom, right, top = array_bounds(profile["height"], profile["width"], profile["transform"])
        extent = [left, right, bottom, top]
    else:
        extent = [0, arr.shape[1], 0, arr.shape[0]]

    shown = np.where(np.isin(arr, codes), arr, np.nan)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    # 同 render_raster_png：分类栅格必须 nearest，否则降采样插值会把类别码平均成连续色而失真。
    ax.imshow(shown, cmap=cmap, norm=norm, extent=extent, aspect="auto",
              interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("经度 (°E)", fontsize=10)
    ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    handles = [Patch(facecolor=colors.get(c, "#cccccc"), edgecolor="#666", label=labels[c]) for c in codes]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9, ncol=2)
    if note:
        ax.text(0.01, -0.16, note, transform=ax.transAxes, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_lines_png(
    x: list[Any],
    series: dict[str, list[float]],
    path: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    note: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (label, ys) in enumerate(series.items()):
        ax.plot(x, ys, marker=markers[i % len(markers)], linewidth=1.8, markersize=5, label=label)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel or "年份", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)
    if len(series) > 1:
        ax.legend(fontsize=9)
    if note:
        ax.text(0.01, -0.18, note, transform=ax.transAxes, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_bars_png(
    labels: list[str],
    values: list[float],
    path: Path,
    title: str,
    ylabel: str = "",
    note: str = "",
    colors: list[str] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors or "#4C8BF5", edgecolor="#2c5f9e")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)
    for pos, val in zip(positions, values):
        ax.text(pos, val, f"{val:.3g}", ha="center", va="bottom", fontsize=8)
    if note:
        ax.text(0.01, -0.28, note, transform=ax.transAxes, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def slugify(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned.strip("_")[:40] or "task"


def dedupe(func):
    """给工具函数加轻量日志。"""

    @functools.wraps(func)
    def wrapper(ctx: ToolContext, **kwargs: Any) -> ToolResult:
        logger.info("执行工具 %s 参数=%s", func.__name__, _short(kwargs))
        result = func(ctx, **kwargs)
        logger.info("工具 %s 完成 ok=%s", func.__name__, result.ok)
        return result

    return wrapper


def _short(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, "", []))[:160]


__all__ = [
    "ToolResult", "ToolContext", "ToolSpec", "ToolRegistry", "registry", "tool",
    "read_raster", "read_bands", "safe_div", "percentile_stretch", "stats_of", "save_geotiff",
    "stack_years", "render_raster_png", "render_class_map_png", "render_lines_png",
    "render_bars_png", "write_csv", "fmt", "slugify", "NODATA", "Settings", "default_settings",
]
