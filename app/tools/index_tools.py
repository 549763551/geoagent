"""指数计算类工具：由多波段反射率派生光谱指数，以及 RSEI 综合生态指数。

算法与 `实验/rsei/calc_rsei.py`（第五版：标准四分量）保持一致：
  NDVI/WET/SAVI/NDBI/NDBSI 由 7 波段反射率计算；
  RSEI = PCA(NDVI, WET, LST_inv, NDBSI) 第一主成分，2%-98% 分位拉伸。
"""

from __future__ import annotations

import numpy as np

from ..catalog import catalog
from .base import (
    ToolResult,
    percentile_stretch,
    read_bands,
    read_raster,
    render_raster_png,
    safe_div,
    save_geotiff,
    slugify,
    stats_of,
    tool,
)

# 缨帽变换湿度分量系数（Liu 2016，MODIS 反射率 b1-b7）
TCT_COEFFS = np.array([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109, -0.3034])
REFL_BANDS = ["Red", "NIR", "Blue", "Green", "SWIR1", "SWIR2", "SWIR3"]

INDEX_SPECS: dict[str, dict] = {
    "NDVI": {"unit": "无量纲(-1~1)", "cmap": "RdYlGn", "vrange": (-0.5, 0.8),
             "desc": "归一化植被指数（NIR-Red)/(NIR+Red)，反映绿度与植被覆盖，越大越绿。"},
    "NDWI": {"unit": "无量纲(-1~1)", "cmap": "Blues", "vrange": (-0.6, 0.4),
             "desc": "归一化水体指数（Green-NIR)/(Green+NIR)，正值指示水体。"},
    "MNDWI": {"unit": "无量纲(-1~1)", "cmap": "Blues", "vrange": (-0.6, 0.4),
              "desc": "改进水体指数（Green-SWIR1)/(Green+SWIR1)，比 NDWI 更抗建筑干扰。"},
    "NDBI": {"unit": "无量纲(-1~1)", "cmap": "YlOrBr", "vrange": (-0.5, 0.4),
             "desc": "归一化建筑指数（SWIR1-NIR)/(SWIR1+NIR)，正值指示不透水面。"},
    "SAVI": {"unit": "无量纲", "cmap": "RdYlGn", "vrange": (-0.3, 0.8),
             "desc": "土壤调节植被指数，L=0.5，抑制干旱区裸土背景影响。"},
    "WET": {"unit": "无量纲", "cmap": "BrBG", "vrange": None,
            "desc": "缨帽变换湿度分量，反映地表水分状况，越大越湿润。"},
    "NDBSI": {"unit": "无量纲", "cmap": "YlOrBr", "vrange": None,
              "desc": "干度指数 (SI+IBI)/2，综合裸土与建筑，越大越干。"},
    "RSEI": {"unit": "无量纲(0~1)", "cmap": "RdYlGn", "vrange": (0.0, 1.0),
             "desc": "遥感生态指数，由绿度/湿度/热度/干度经主成分分析合成，0~1 越大生态越好。"},
}


def _reflectance_bands(year: int | None):
    """读取反射率 7 波段，并按物理有效范围 [0,1] 过滤异常值。"""
    entry = catalog.get("reflectance")
    if entry is None:
        raise FileNotFoundError("目录中缺少反射率数据（reflectance）")
    actual = entry.nearest_year(year)
    path = entry.path_for(actual)
    if path is None:
        raise FileNotFoundError("反射率数据没有可用年份文件")
    bands, profile, _ = read_bands(path)
    if len(bands) < 7:
        raise ValueError(f"反射率波段数应为 7，实际 {len(bands)}")
    bands = bands[:7]
    for band in bands:
        band[(band < 0) | (band > 1)] = np.nan
    return bands, profile, actual


def _si_ibi_ndbsi(bands: list[np.ndarray]) -> np.ndarray:
    """干度 NDBSI = (SI + IBI) / 2（徐涵秋标准）。"""
    red, nir, blue, green, swir1 = bands[0], bands[1], bands[2], bands[3], bands[4]
    si = np.clip(safe_div((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue), eps=0.01), -1.0, 1.0)
    level = 0.5
    ndbi = safe_div(swir1 - nir, swir1 + nir, eps=0.01)
    savi = safe_div(nir - red, nir + red + level, eps=0.01) * (1.0 + level)
    mndwi = safe_div(green - swir1, green + swir1, eps=0.01)
    mix = (savi + mndwi) / 2.0
    ibi = safe_div(ndbi - mix, ndbi + mix, eps=0.01)
    return (si + ibi) / 2.0


def _wet(bands: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(bands[0])
    for coeff, band in zip(TCT_COEFFS, bands):
        out = out + coeff * band
    return out


def _simple_index(index: str, bands: list[np.ndarray]) -> np.ndarray:
    red, nir, blue, green, swir1 = bands[0], bands[1], bands[2], bands[3], bands[4]
    if index == "NDVI":
        return safe_div(nir - red, nir + red)
    if index == "NDWI":
        return safe_div(green - nir, green + nir)
    if index == "MNDWI":
        return safe_div(green - swir1, green + swir1)
    if index == "NDBI":
        return safe_div(swir1 - nir, swir1 + nir)
    if index == "SAVI":
        level = 0.5
        return safe_div(nir - red, nir + red + level) * (1.0 + level)
    if index == "WET":
        return _wet(bands)
    if index == "NDBSI":
        return _si_ibi_ndbsi(bands)
    raise ValueError(f"不支持的指数：{index}")


def _compute_rsei(year: int | None) -> tuple[np.ndarray, dict, dict]:
    """RSEI = PCA(NDVI, WET, LST_inv, NDBSI) 的 PC1（2%-98% 拉伸）。"""
    from sklearn.decomposition import PCA

    ndvi_entry = catalog.get("ndvi")
    lst_entry = catalog.get("lst")
    if ndvi_entry is None or lst_entry is None:
        raise FileNotFoundError("计算 RSEI 需要 NDVI 与 LST 数据集")

    ref_bands, ref_profile, ref_year = _reflectance_bands(year)
    ndvi_year = ndvi_entry.nearest_year(year)
    lst_year = lst_entry.nearest_year(year)
    ndvi_raw, ndvi_profile, _ = read_raster(ndvi_entry.path_for(ndvi_year))
    lst_raw, _, _ = read_raster(lst_entry.path_for(lst_year))

    if not (ndvi_raw.shape == lst_raw.shape == ref_bands[0].shape):
        raise ValueError(
            f"网格不一致：NDVI {ndvi_raw.shape} / LST {lst_raw.shape} / 反射率 {ref_bands[0].shape}"
        )

    wet = _wet(ref_bands)
    ndbsi = _si_ibi_ndbsi(ref_bands)

    valid = np.isfinite(ndvi_raw) & np.isfinite(lst_raw) & np.isfinite(wet) & np.isfinite(ndbsi)
    if int(valid.sum()) < 100:
        raise ValueError("有效像元不足 100，无法计算 RSEI")

    ndvi_n = percentile_stretch(ndvi_raw, valid)
    wet_n = percentile_stretch(wet, valid)
    lst_n = percentile_stretch(lst_raw, valid)
    ndbsi_n = percentile_stretch(ndbsi, valid)
    lst_inv = 1.0 - lst_n

    matrix = np.column_stack([ndvi_n[valid], lst_inv[valid], wet_n[valid], ndbsi_n[valid]])
    pca = PCA(n_components=2)
    comps = pca.fit_transform(matrix)
    loadings = pca.components_
    if (loadings[0][0] + loadings[0][2]) < 0:  # 主导项（NDVI+WET）应为正
        comps[:, 0] = -comps[:, 0]
        loadings[0] = -loadings[0]

    pc1 = np.full(ndvi_raw.shape, np.nan, dtype=np.float64)
    pc1[valid] = comps[:, 0]
    rsei = percentile_stretch(pc1, valid)

    meta = {
        "source_year": ref_year,
        "lst_year": lst_year,
        "ndvi_year": ndvi_year,
        "pc1_variance_ratio": round(float(pca.explained_variance_ratio_[0]), 4),
        "pc12_variance_ratio": round(float(pca.explained_variance_ratio_[:2].sum()), 4),
        "loadings": {
            name: round(float(v), 4)
            for name, v in zip(["NDVI", "LST_inv", "WET", "NDBSI"], loadings[0])
        },
        "components": {
            "NDVI": round(float(ndvi_n[valid].mean()), 4),
            "WET": round(float(wet_n[valid].mean()), 4),
            "LST_inv": round(float(lst_inv[valid].mean()), 4),
            "NDBSI": round(float(ndbsi_n[valid].mean()), 4),
        },
    }
    return rsei, ndvi_profile, meta


def _interpret(index: str, mean: float, meta: dict) -> str:
    if index == "RSEI":
        level = ("优" if mean >= 0.8 else "良好" if mean >= 0.6 else
                 "中等" if mean >= 0.4 else "较差" if mean >= 0.2 else "差")
        return f"研究区整体生态状况为“{level}”等级；PC1 贡献率 {meta.get('pc1_variance_ratio', 0)*100:.1f}%。"
    if index in ("NDVI", "SAVI"):
        if mean >= 0.4:
            return "区域植被覆盖总体较好。"
        if mean >= 0.15:
            return "植被覆盖中等，存在明显空间差异。"
        return "植被稀疏，符合干旱区特征，局部绿洲为高值区。"
    if index in ("NDWI", "MNDWI"):
        return "正值区域为水体，负值区为陆地。"
    if index == "NDBI":
        return "正值区域指示建成区/不透水面，负值区为植被或裸地。"
    if index == "WET":
        return "高值区地表水分条件较好，低值区偏干。"
    if index == "NDBSI":
        return "高值区地表干燥、裸土或建筑占比大。"
    return ""


@tool(
    name="compute_index",
    description="由 MODIS 反射率计算遥感光谱指数，可选 NDVI/NDWI/MNDWI/NDBI/SAVI/WET/NDBSI/RSEI。"
                "RSEI 为绿度/湿度/热度/干度主成分合成的综合生态指数（0~1，越大越好）。"
                "输出 GeoTIFF、专题图与统计量。回答“植被/水体/生态质量如何”“计算某指数”时使用。",
    parameters={
        "type": "object",
        "properties": {
            "index": {
                "type": "string",
                "enum": ["NDVI", "NDWI", "MNDWI", "NDBI", "SAVI", "WET", "NDBSI", "RSEI"],
                "description": "指数名称",
            },
            "year": {"type": "integer", "description": "年份，默认取与数据最接近的一期"},
        },
        "required": ["index"],
    },
    category="指数计算",
)
def compute_index(ctx, index: str, year: int | None = None) -> ToolResult:
    index = (index or "").strip().upper()
    if index not in INDEX_SPECS:
        return ToolResult.failure(f"不支持的指数“{index}”，可选：{', '.join(INDEX_SPECS)}")
    spec = INDEX_SPECS[index]

    ctx.report(f"正在计算 {index} 指数…", 20)
    try:
        if index == "RSEI":
            arr, profile, meta = _compute_rsei(year)
            tag = f"RSEI_{meta.get('source_year')}"
        else:
            bands, profile, actual = _reflectance_bands(year)
            arr = _simple_index(index, bands)
            meta = {"source_year": actual}
            tag = f"{index}_{actual}"
    except (FileNotFoundError, ValueError) as exc:
        return ToolResult.failure(str(exc))

    if not np.isfinite(arr).any():
        return ToolResult.failure(f"{index} 计算结果没有有效像元，请检查输入数据")

    ctx.report(f"{index} 计算完成，正在制图…", 70)
    tif_path = ctx.out_dir("rasters") / f"{slugify(tag)}.tif"
    save_geotiff(arr, profile, tif_path)
    png_path = ctx.out_dir("figures") / f"{slugify(tag)}.png"
    vrange = spec["vrange"] or (None, None)
    render_raster_png(
        arr, profile, png_path,
        title=f"{index}（{meta.get('source_year')} 年）",
        unit=spec["unit"], cmap=spec["cmap"], vmin=vrange[0], vmax=vrange[1],
        note=f"{spec['desc']}{'；' if spec['desc'] else ''}像元级结果，2%-98% 分位显示",
    )

    stat = stats_of(arr, spec["unit"], {"index": index, "year": meta.get("source_year")})
    stat.update({k: v for k, v in meta.items() if k not in ("source_year",)})
    note = _interpret(index, stat["mean"], meta)
    summary = (
        f"{index}（{meta.get('source_year')} 年）：均值 {stat['mean']}，中位数 {stat['median']}，"
        f"P5~P95 为 {stat['p5']}~{stat['p95']}，有效像元 {stat['valid_pixels']:,}。{note}"
        f"已输出 GeoTIFF 与专题图。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={"index": index, "unit": spec["unit"], "stats": stat},
        artifacts=[
            ctx.artifact(png_path, "image", f"{index} 专题图（{meta.get('source_year')}）"),
            ctx.artifact(tif_path, "tif", f"{index} 栅格结果（{meta.get('source_year')}）"),
        ],
    )
