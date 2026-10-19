"""遥感数据目录（Data Catalog）与检索增强（RAG）。

职责：
1. 扫描 `GEO_DATA_ROOT`，把散落的 GeoTIFF 组织成带语义描述的“数据集条目”；
2. 提供中文别名 / 标签 / 描述的关键词打分检索，为智能体提供“数据在哪、能不能用”的知识；
3. 生成紧凑目录摘要，注入规划提示词。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import rasterio

from .config import Settings, settings as default_settings

logger = logging.getLogger("geoagent.catalog")

# ----------------------------------------------------------------------
# 数据集声明：把目录里的文件“翻译”成语义化的数据资产
# ----------------------------------------------------------------------

LANDUSE_CLASSES = {
    1: "耕地",
    2: "林地",
    3: "草地",
    4: "水域",
    5: "城乡工矿居民用地",
    6: "未利用土地",
    9: "海洋",
}

# 关键：中文别名表驱动检索增强（RAG）
ALIASES: dict[str, list[str]] = {
    "ndvi": ["ndvi", "植被指数", "植被", "绿度", "植被覆盖", "绿化", "长势", "生态质量"],
    "lst": ["lst", "地表温度", "陆表温度", "地表热量", "热度", "热环境", "城市热岛"],
    "reflectance": ["reflectance", "反射率", "波段", "光谱", "swir", "nir", "红光", "近红外"],
    "elevation": ["dem", "高程", "海拔", "地形", "地势", "srtm"],
    "slope": ["坡度", "slope", "陡坡", "地形起伏"],
    "twi": ["twi", "地形湿度指数", "湿度指数", "汇流", "地形湿度"],
    "soil_moisture": ["土壤水", "土壤湿度", "土壤水分", "墒情", "gldas", "soilmoi"],
    "water_occurrence": ["水体", "水域", "地表水", "水面", "河流", "湖泊", "jrc", "occurrence", "水文"],
    "landuse": ["土地利用", "土地覆被", "覆被", "土地分类", "地类", "耕地", "林地", "草地", "未利用",
                "荒漠", "沙地", "城乡用地", "一级类", "土地类型"],
    "population": ["人口", "人口密度", "人口分布", "gpw", "人居"],
    "built": ["建成区", "建设用地", "城市扩张", "城市化", "不透水", "ghs", "人造地表"],
    "precip": ["降水", "降雨", "雨量", "年降水量", "湿润", "pr"],
    "temp": ["气温", "年均温", "平均气温", "温度", "气候变暖", "tmm"],
    "pet": ["蒸散", "潜在蒸散", "蒸散发", "pet", "蒸发"],
    "deficit": ["水分亏缺", "气候亏缺", "干旱", "缺水", "干旱胁迫", "def", "干燥"],
    "aridity": ["干燥度", "干旱指数", "aridity", "ai指数"],
    "cmip_precip": ["未来降水", "情景降水", "ssp降水", "cmip降水", "未来降雨"],
    "cmip_temp": ["未来气温", "情景气温", "ssp气温", "cmip气温", "未来温度", "未来情景"],
}

# 派生指标（不是磁盘上的数据集，而是由工具按需计算）
DERIVED_INDICES = {
    "NDVI": "归一化植被指数，反映绿度与植被覆盖（-1~1，越大越好）",
    "NDWI": "归一化水体指数 (Green-NIR)/(Green+NIR)，反映地表水体",
    "MNDWI": "改进归一化水体指数 (Green-SWIR1)/(Green+SWIR1)",
    "NDBI": "归一化建筑指数 (SWIR1-NIR)/(SWIR1+NIR)，反映不透水面",
    "NDBSI": "干度指数，(SI+IBI)/2，综合裸土与建筑，越大越干",
    "WET": "缨帽变换湿度分量，反映地表水分状况",
    "SAVI": "土壤调节植被指数",
    "RSEI": "遥感生态指数，由绿度/湿度/热度/干度经主成分分析合成，0~1 越大生态越好",
}


@dataclass
class DatasetSpec:
    key: str
    title: str
    variable: str
    unit: str
    description: str
    tags: list[str]
    subdir: str
    template: str = ""
    years: list[int] = field(default_factory=list)
    bands: list[str] = field(default_factory=lambda: ["band1"])
    kind: str = "continuous"  # continuous | categorical | scenario
    nodata: float | None = None
    classes: dict[int, str] | None = None
    scenario_files: dict[str, str] = field(default_factory=dict)


def _year_range(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        key="ndvi", title="NDVI 植被指数（年最大值合成）", variable="NDVI", unit="无量纲(-1~1)",
        description="基于 MODIS 反射率计算的归一化植被指数，按年取最大值合成（MVC），表征区域绿度与植被覆盖状况。",
        tags=["绿度", "植被", "生态", "逐年"],
        subdir="NDVI", template="NDVI_MVC_{year}.tif", years=_year_range(2000, 2025),
    ),
    DatasetSpec(
        key="lst", title="地表温度（年均值）", variable="LST", unit="摄氏度(°C)",
        description="MODIS 地表温度年均值，反映地表热环境，可用于城市热岛与干旱胁迫分析。",
        tags=["热度", "温度", "热环境", "逐年"],
        subdir="LST", template="LST_mean_{year}.tif", years=_year_range(2000, 2025),
    ),
    DatasetSpec(
        key="reflectance", title="地表反射率（7 波段中值合成）", variable="Reflectance", unit="反射率(0~1)",
        description="MODIS 地表反射率 7 波段中值合成影像，波段顺序为 Red/NIR/Blue/Green/SWIR1/SWIR2/SWIR3，"
                    "是计算各类光谱指数（NDVI/NDWI/NDBI/干度/湿度等）的原始输入。",
        tags=["光谱", "基础数据", "逐年"],
        subdir="Reflectance", template="Reflectance_median_{year}.tif", years=_year_range(2000, 2025),
        bands=["Red", "NIR", "Blue", "Green", "SWIR1", "SWIR2", "SWIR3"],
    ),
    DatasetSpec(
        key="elevation", title="高程 DEM", variable="Elevation", unit="米(m)",
        description="SRTM 数字高程模型，1km 重采样，用于地形分异与高程带分析。",
        tags=["地形", "静态"], subdir="DEM", template="SRTM_elevation_1km.tif",
    ),
    DatasetSpec(
        key="slope", title="坡度", variable="Slope", unit="度(°)",
        description="由 DEM 计算的坡度，反映地形陡缓程度。",
        tags=["地形", "静态"], subdir="DEM", template="SRTM_slope_1km.tif",
    ),
    DatasetSpec(
        key="twi", title="地形湿度指数 TWI", variable="TWI", unit="无量纲",
        description="地形湿度指数，反映地形对水分汇集的潜在影响，值越大越湿润。",
        tags=["地形", "水分", "静态"], subdir="DEM", template="TWI_1km.tif",
    ),
    DatasetSpec(
        key="soil_moisture", title="土壤水分（GLDAS）", variable="SoilMoisture", unit="kg/m²",
        description="GLDAS 陆面模型表层土壤水分含量，反映地表干湿状况（5 年间隔）。",
        tags=["水分", "土壤"], subdir="Water", template="GLDAS_SoilMoi_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="water_occurrence", title="地表水出现频率（JRC）", variable="WaterOccurrence", unit="%(0~100)",
        description="JRC 全球地表水数据集，表示 1984 年以来卫星观测到水体的时间占比，反映永久/季节性水体分布。",
        tags=["水体", "静态"], subdir="Water", template="JRC_occurrence.tif",
    ),
    DatasetSpec(
        key="landuse", title="土地利用/覆被（1km 一级类）", variable="LandUse", unit="类别码",
        description="中科院 1km 土地利用一级类数据，编码：1 耕地 / 2 林地 / 3 草地 / 4 水域 / "
                    "5 城乡工矿居民用地 / 6 未利用土地 / 9 海洋，可作为分区统计的分区底图。",
        tags=["分区", "地类", "覆被"], subdir="中科院", template="{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025], kind="categorical",
        nodata=-9999.0, classes=LANDUSE_CLASSES,
    ),
    DatasetSpec(
        key="population", title="人口密度（GPW）", variable="Population", unit="人/km²",
        description="GPW 人口密度栅格，反映人类活动强度与压力（5 年间隔）。",
        tags=["社会经济", "压力"], subdir="Population", template="GPW_pop_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020],
    ),
    DatasetSpec(
        key="built", title="建成区（GHS-BUILT）", variable="BuiltUp", unit="建成比例(0~1)",
        description="GHS-BUILT 建成区栅格，反映人造地表与城市扩张强度。",
        tags=["社会经济", "城市"], subdir="Built", template="GHS_built_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="precip", title="年降水量（TerraClimate）", variable="Precipitation", unit="mm/年",
        description="TerraClimate 年累计降水量，反映区域水分供给（5 年间隔）。",
        tags=["气候", "水分"], subdir="TerraClimate", template="TC_pr_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="temp", title="年均气温（TerraClimate）", variable="MeanTemperature", unit="摄氏度(°C)",
        description="TerraClimate 年平均气温，反映热量条件与气候变暖趋势（5 年间隔）。",
        tags=["气候", "温度"], subdir="TerraClimate", template="TC_tmm_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="pet", title="潜在蒸散（TerraClimate）", variable="PET", unit="mm/年",
        description="TerraClimate 年潜在蒸散量，反映大气蒸发需求（5 年间隔）。",
        tags=["气候", "蒸发"], subdir="TerraClimate", template="TC_pet_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="deficit", title="气候水分亏缺（TerraClimate）", variable="Deficit", unit="mm/年",
        description="TerraClimate 年气候水分亏缺（PET−实际蒸散），是刻画干旱胁迫的关键指标（5 年间隔）。",
        tags=["气候", "干旱"], subdir="TerraClimate", template="TC_def_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="aridity", title="干燥度指数（TerraClimate）", variable="AridityIndex", unit="无量纲",
        description="TerraClimate 干燥度指数（降水/潜在蒸散），值越小越干旱（5 年间隔）。",
        tags=["气候", "干旱"], subdir="TerraClimate", template="TC_AI_{year}.tif",
        years=[2000, 2005, 2010, 2015, 2020, 2025],
    ),
    DatasetSpec(
        key="cmip_precip", title="CMIP6 未来降水情景", variable="Precipitation", unit="mm/日",
        description="CMIP6 多模式集合的降水情景数据，含历史基准 hist 与 ssp245 / ssp585 两种共享社会经济路径，"
                    "时段为 2030 / 2050 / 2100，可用于未来水分条件对比（分辨率约 0.22°）。",
        tags=["情景", "未来", "降水"], subdir="CMIP6", kind="scenario",
        scenario_files={
            "hist": "CMIP6_hist_pr.tif",
            "ssp245_2030": "CMIP6_ssp245_2030_pr.tif",
            "ssp245_2050": "CMIP6_ssp245_2050_pr.tif",
            "ssp245_2100": "CMIP6_ssp245_2100_pr.tif",
            "ssp585_2030": "CMIP6_ssp585_2030_pr.tif",
            "ssp585_2050": "CMIP6_ssp585_2050_pr.tif",
            "ssp585_2100": "CMIP6_ssp585_2100_pr.tif",
        },
    ),
    DatasetSpec(
        key="cmip_temp", title="CMIP6 未来气温情景", variable="Temperature", unit="摄氏度(°C)",
        description="CMIP6 多模式集合的气温情景数据，含历史基准 hist 与 ssp245 / ssp585 两种路径，"
                    "时段为 2030 / 2050 / 2100，可用于未来增温幅度对比（分辨率约 0.22°）。",
        tags=["情景", "未来", "气温"], subdir="CMIP6", kind="scenario",
        scenario_files={
            "hist": "CMIP6_hist_tas.tif",
            "ssp245_2030": "CMIP6_ssp245_2030_tas.tif",
            "ssp245_2050": "CMIP6_ssp245_2050_tas.tif",
            "ssp245_2100": "CMIP6_ssp245_2100_tas.tif",
            "ssp585_2030": "CMIP6_ssp585_2030_tas.tif",
            "ssp585_2050": "CMIP6_ssp585_2050_tas.tif",
            "ssp585_2100": "CMIP6_ssp585_2100_tas.tif",
        },
    ),
]


@dataclass
class DatasetEntry:
    key: str
    title: str
    variable: str
    unit: str
    description: str
    tags: list[str]
    kind: str
    root: Path
    bands: list[str]
    nodata: float | None
    classes: dict[int, str] | None
    years: dict[int, Path] = field(default_factory=dict)
    scenario_files: dict[str, Path] = field(default_factory=dict)
    static_path: Path | None = None
    crs: str = ""
    shape: tuple[int, int] = (0, 0)
    resolution_deg: float = 0.0

    # -- 便捷访问 ----------------------------------------------------
    @property
    def available_years(self) -> list[int]:
        return sorted(self.years)

    @property
    def is_temporal(self) -> bool:
        return bool(self.years)

    def path_for(self, year: int | None) -> Path | None:
        if year is None:
            return next(iter(self.years.values()), next(iter(self.scenario_files.values()), self.static_path))
        if year in self.years:
            return self.years[year]
        if not self.years:
            return next(iter(self.scenario_files.values()), self.static_path)
        nearest = min(self.years, key=lambda y: abs(y - year))
        return self.years[nearest]

    def nearest_year(self, year: int | None) -> int | None:
        if not self.years:
            return None
        if year is None or year in self.years:
            return min(self.years) if year is None else year
        return min(self.years, key=lambda y: abs(y - year))

    def scenario_path(self, key: str) -> Path | None:
        return self.scenario_files.get(key)

    def resolution_text(self) -> str:
        if not self.resolution_deg:
            return "未知"
        km = self.resolution_deg * 111.32
        return f"约 {km:.2f} km（{self.resolution_deg:.5f}°）"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "variable": self.variable,
            "unit": self.unit,
            "description": self.description,
            "tags": self.tags,
            "kind": self.kind,
            "years": self.available_years,
            "year_span": [min(self.years), max(self.years)] if self.years else None,
            "scenarios": sorted(self.scenario_files),
            "bands": self.bands if len(self.bands) > 1 else [],
            "resolution": self.resolution_text(),
            "crs": self.crs,
            "shape": list(self.shape),
            "classes": self.classes,
        }

    def brief(self) -> str:
        if self.years:
            span = f"{min(self.years)}-{max(self.years)}（{len(self.years)}期）"
        elif self.scenario_files:
            span = "情景期：" + "/".join(sorted(self.scenario_files))
        else:
            span = "静态"
        return f"- {self.key}｜{self.title}｜单位:{self.unit}｜时间:{span}｜{self.description[:60]}"


class DataCatalog:
    """数据资产目录，负责扫描与检索。"""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.entries: dict[str, DatasetEntry] = {}
        self.scan()

    # ------------------------------------------------------------------
    def scan(self) -> None:
        self.entries = {}
        for spec in DATASET_SPECS:
            root = self.cfg.data_root / spec.subdir
            if not root.exists():
                logger.warning("数据目录不存在，跳过 %s: %s", spec.key, root)
                continue
            entry = DatasetEntry(
                key=spec.key, title=spec.title, variable=spec.variable, unit=spec.unit,
                description=spec.description, tags=spec.tags, kind=spec.kind, root=root,
                bands=spec.bands, nodata=spec.nodata, classes=spec.classes,
            )
            for year in spec.years:
                path = root / spec.template.format(year=year)
                if path.exists():
                    entry.years[year] = path
            for key, name in spec.scenario_files.items():
                path = root / name
                if path.exists():
                    entry.scenario_files[key] = path
            if not entry.years and not entry.scenario_files and spec.template and "{year}" not in spec.template:
                static_file = root / spec.template
                if static_file.exists():
                    entry.static_path = static_file

            probe = next(
                iter(entry.years.values()),
                next(iter(entry.scenario_files.values()), entry.static_path),
            )
            if probe is not None:
                try:
                    with rasterio.open(probe) as src:
                        entry.crs = str(src.crs) if src.crs else "未定义"
                        entry.shape = (src.height, src.width)
                        entry.resolution_deg = abs(src.res[0])
                        if (entry.nodata is None) and (src.nodata is not None):
                            entry.nodata = float(src.nodata)
                        if len(entry.bands) != src.count and src.count > 1:
                            entry.bands = [f"band{i}" for i in range(1, src.count + 1)]
                except Exception as exc:  # pragma: no cover
                    logger.warning("读取元数据失败 %s: %s", probe, exc)

            if not entry.years and not entry.scenario_files and entry.static_path is None:
                logger.warning("数据集 %s 无可用文件，跳过", spec.key)
                continue
            self.entries[spec.key] = entry

    # ------------------------------------------------------------------
    def get(self, key: str) -> DatasetEntry | None:
        return self.entries.get(key.strip())

    def resolve(self, token: str) -> DatasetEntry | None:
        """把用户/模型给出的模糊名称解析为目录条目。"""
        if not token:
            return None
        token = token.strip()
        if token in self.entries:
            return self.entries[token]
        low = token.lower()
        for key, entry in self.entries.items():
            if low == key.lower() or low == entry.variable.lower():
                return entry
            if low in entry.title.lower():
                return entry
        hits = self.search(token, k=1)
        return self.entries[hits[0]["key"]] if hits else None

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict]:
        """关键词加权的目录检索（轻量 RAG）。"""
        if not query:
            return []
        text = query.lower()
        latin_tokens = set(re.findall(r"[a-z0-9_]+", text))
        scores: dict[str, float] = {key: 0.0 for key in self.entries}

        for key, entry in self.entries.items():
            score = 0.0
            for alias in ALIASES.get(key, []):
                if alias.lower() in text:
                    # 越长的别名越具体（干燥度 > 干燥、未来降水 > 降水），避免短别名遮蔽长别名
                    score += 3.0 + 0.5 * len(alias)
            for token in latin_tokens:
                if token in key or token in entry.variable.lower():
                    score += 2.0
            for tag in entry.tags:
                if tag in text:
                    score += 2.0
            if entry.variable.lower() in text or entry.variable.lower() in latin_tokens:
                score += 2.0
            for token in latin_tokens:
                if token and token in entry.description.lower():
                    score += 0.6
            if score:
                scores[key] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        for key, score in ranked:
            if score <= 0:
                continue
            entry = self.entries[key]
            results.append({**entry.to_dict(), "score": round(score, 2)})
            if len(results) >= k:
                break
        return results

    # ------------------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self.entries

    def all(self) -> list[DatasetEntry]:
        return list(self.entries.values())

    def summary(self) -> str:
        """紧凑目录摘要，用于注入提示词。"""
        lines = [e.brief() for e in self.all()]
        return "\n".join(lines)

    def dump(self, path: Path | None = None) -> Path:
        target = path or (self.cfg.state_dir / "catalog.json")
        payload = {
            "data_root": str(self.cfg.data_root),
            "datasets": [e.to_dict() for e in self.all()],
            "derived_indices": DERIVED_INDICES,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


catalog = DataCatalog()
