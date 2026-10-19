"""生成一份与项目目录约定一致的合成演示数据。

真实遥感数据体积达数 GB，不适合入库。本脚本按 `app/catalog.py` 中的
`DATASET_SPECS`（子目录 + 文件名模板）生成一套**小尺寸、可复现**的合成栅格，
让 GeoAgent 在没有任何真实数据的情况下也能完整跑通「理解 → 规划 → 工具 → 交付」。

⚠️ 生成的数据全部是数学合成结果，不映射任何真实地点，仅供功能演示与测试。

用法：
    python scripts/make_demo_data.py --out raw
    python scripts/make_demo_data.py --out raw --size 100x150 --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# 合成研究区的地理参考：任意选取，不代表真实地点
WEST, NORTH = 100.0, 40.0
CRS = "EPSG:4326"

LANDUSE_CLASSES = {1: "耕地", 2: "林地", 3: "草地", 4: "水域",
                   5: "城乡工矿居民用地", 6: "未利用土地", 9: "海洋"}
NDVI_BASE = {1: 0.55, 2: 0.62, 3: 0.35, 4: 0.05, 5: 0.18, 6: 0.09, 9: 0.0}

ANNUAL_YEARS = list(range(2000, 2026))          # 逐年
FIVE_YEARS = [2000, 2005, 2010, 2015, 2020, 2025]  # 五年间隔
POP_YEARS = [2000, 2005, 2010, 2015, 2020]

SCENARIOS = {
    "ssp245": {"2030": (0.05, 0.8), "2050": (0.09, 1.6), "2100": (0.14, 2.9)},
    "ssp585": {"2030": (0.08, 1.2), "2050": (0.15, 2.6), "2100": (0.26, 4.8)},
}


def smooth(field: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """轻度平滑，避免合成场显得像纯噪声。scipy 缺失时原样返回。"""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma)
    except Exception:
        return field


def base_terrain(shape: tuple[int, int], seed: int) -> dict[str, np.ndarray]:
    """构造一套自洽的地形与地类底图。"""
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u, v = xx / max(w - 1, 1), yy / max(h - 1, 1)   # v=0 为北

    elev = (1200
            + 1800 * (1 - v)
            + 250 * np.sin(2.2 * np.pi * u)
            + 180 * np.cos(1.7 * np.pi * v)
            + 90 * np.sin(5.1 * np.pi * u * v))
    elev = elev + smooth(rng.normal(0, 60, shape), 2.0)

    # 地类：按高程带与位置划分，并留出小片海洋以覆盖 9 类
    landuse = np.full(shape, 6, dtype=np.int16)
    landuse[elev < 2600] = 3
    landuse[(elev >= 2600) & (elev < 3200)] = 6
    landuse[(u < 0.28) & (elev < 2300)] = 1
    landuse[(u >= 0.28) & (u < 0.45) & (elev < 2100)] = 2
    river = np.abs(u - (0.05 + 0.5 * v ** 1.4)) < 0.012
    landuse[river & (elev < 2500)] = 4
    landuse[(u > 0.93) & (v > 0.88)] = 9
    built = ((u - 0.62) ** 2 + (v - 0.45) ** 2) < 0.006
    landuse[built] = 5

    # 坡度 / TWI
    dzdy, dzdx = np.gradient(elev)
    dy_m = 0.01 * 111_320.0
    dx_m = dy_m * np.cos(np.deg2rad(NORTH))
    slope = np.degrees(np.arctan(np.hypot(dzdx / dx_m, dzdy / dy_m)))
    slope = np.clip(slope, 0.0, 60.0)
    twi = np.log1p(1.0 / np.maximum(np.tan(np.deg2rad(slope)), 1e-3))

    return {"elev": elev, "slope": slope, "twi": twi, "landuse": landuse, "u": u, "v": v}


def climate_fields(t: dict, year: int, seed: int) -> dict[str, np.ndarray]:
    """与年份相关的气候场。"""
    rng = np.random.default_rng(seed + year)
    u, v, elev = t["u"], t["v"], t["elev"]
    warming = 0.028 * (year - 2000)      # 年均增温
    precip = (140 + 520 * (0.35 + 0.65 * (1 - u)) + 120 * (1 - v)
              + 1.6 * (year - 2000) + rng.normal(0, 18, elev.shape))
    precip = np.clip(precip, 10, 1600)
    temp = 4 + 16 * u - 0.004 * elev + warming + rng.normal(0, 0.6, elev.shape)
    pet = 820 + 340 * (1 - v) + 12 * warming + rng.normal(0, 20, elev.shape)
    deficit = np.clip(pet - 0.55 * precip, 0, None)
    aridity = precip / np.maximum(pet, 1e-6)
    return {"precip": precip, "temp": temp, "pet": pet, "deficit": deficit, "aridity": aridity}


def ndvi_field(t: dict, year: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000 + year)
    landuse = t["landuse"]
    base = np.zeros(landuse.shape, dtype=float)
    for cls, val in NDVI_BASE.items():
        base[landuse == cls] = val
    trend = 0.0011 * (year - 2000) * (0.6 + 0.8 * t["u"])   # 空间上略有差异的改善速率
    ndvi = base + trend + rng.normal(0, 0.035, landuse.shape)
    return np.clip(ndvi, -0.2, 0.95).astype(np.float32)


def lst_field(t: dict, ndvi: np.ndarray, year: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 2000 + year)
    lst = (34.0 - 26.0 * ndvi - 0.0035 * (t["elev"] - 1200)
           + 0.02 * (year - 2000) + rng.normal(0, 1.4, ndvi.shape))
    return np.clip(lst, -20, 55).astype(np.float32)


def reflectance_from_ndvi(t: dict, ndvi: np.ndarray, year: int, seed: int) -> np.ndarray:
    """构造 7 波段反射率，使 (NIR-Red)/(NIR+Red) 恰好等于给定 NDVI。

    注意：Red / NIR 两个波段不加独立噪声，否则会破坏上述恒等关系，
    导致 compute_index 现算的 NDVI 与 NDVI 数据集不一致。其余 5 个波段可以加噪声。
    """
    rng = np.random.default_rng(seed + 3000 + year)
    n = np.clip(ndvi, -0.95, 0.95)
    total = 0.50                                    # NIR + Red
    nir = total * (1 + n) / 2
    red = total * (1 - n) / 2
    noisy = lambda x: np.clip(x + rng.normal(0, 0.004, n.shape), 0, 1)  # noqa: E731
    bands = [
        red,                                        # 1 Red   —— 保持精确
        nir,                                        # 2 NIR   —— 保持精确
        noisy(red * 0.82),                          # 3 Blue
        noisy(red * 1.12),                          # 4 Green
        noisy(0.30 - 0.16 * n),                     # 5 SWIR1
        noisy(0.34 - 0.10 * n),                     # 6 SWIR2
        noisy(0.26 - 0.07 * n),                     # 7 SWIR3
    ]
    return np.stack([np.clip(b, 0, 1) for b in bands]).astype(np.float32)


def write_tif(path: Path, array: np.ndarray, *, nodata=None, descriptions=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    count, h, w = array.shape
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": count,
        "dtype": array.dtype, "crs": CRS,
        "transform": from_origin(WEST, NORTH, 0.01, 0.01),
        "compress": "deflate", "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        for i, desc in enumerate(descriptions or [], start=1):
            if i <= count:
                dst.set_band_description(i, desc)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 GeoAgent 合成演示数据")
    ap.add_argument("--out", default="raw", help="输出根目录（默认 raw）")
    ap.add_argument("--size", default="100x150", help="栅格尺寸 高x宽，默认 100x150")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = ap.parse_args()

    try:
        h, w = (int(x) for x in args.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size 需形如 100x150")

    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        print(f"⚠️  {root} 已存在且有内容，新文件将覆盖同名文件")

    t0 = time.time()
    shape = (h, w)
    t = base_terrain(shape, args.seed)
    written = 0

    def put(rel: str, arr: np.ndarray, **kw) -> None:
        nonlocal written
        write_tif(root / rel, arr, **kw)
        written += 1

    # ---- 静态地形 ----
    put("DEM/SRTM_elevation_1km.tif", t["elev"].astype(np.float32), nodata=-9999.0)
    put("DEM/SRTM_slope_1km.tif", t["slope"].astype(np.float32), nodata=-9999.0)
    put("DEM/TWI_1km.tif", t["twi"].astype(np.float32), nodata=-9999.0)
    print(f"  地形 3 个")

    # ---- 土地利用（分类）----
    for year in FIVE_YEARS:
        lu = t["landuse"].astype(np.int16)
        if year >= 2015:                        # 后期建成区略有扩张
            grow = ((t["u"] - 0.62) ** 2 + (t["v"] - 0.45) ** 2) < 0.011
            lu[grow] = 5
        put(f"中科院/{year}.tif", lu, nodata=-9999.0)
    print(f"  土地利用 {len(FIVE_YEARS)} 个")

    # ---- 逐年：NDVI / LST / 反射率 ----
    for year in ANNUAL_YEARS:
        ndvi = ndvi_field(t, year, args.seed)
        put(f"NDVI/NDVI_MVC_{year}.tif", ndvi, nodata=-9999.0,
            descriptions=["NDVI"])
        lst = lst_field(t, ndvi, year, args.seed)
        put(f"LST/LST_mean_{year}.tif", lst, nodata=-9999.0,
            descriptions=["LST"])
        refl = reflectance_from_ndvi(t, ndvi, year, args.seed)
        put(f"Reflectance/Reflectance_median_{year}.tif", refl, nodata=-9999.0,
            descriptions=["Red", "NIR", "Blue", "Green", "SWIR1", "SWIR2", "SWIR3"])
    print(f"  NDVI/LST/反射率 各 {len(ANNUAL_YEARS)} 个（共 {len(ANNUAL_YEARS) * 3} 个文件）")

    # ---- 水体与土壤水分 ----
    water = np.where(np.isin(t["landuse"], [4, 9]), 100.0, np.clip(6 * t["twi"], 0, 40))
    put("Water/JRC_occurrence.tif", water.astype(np.float32), nodata=-9999.0)
    for year in FIVE_YEARS:
        clim = climate_fields(t, year, args.seed)
        soil = np.clip(0.06 * clim["precip"] + 2.0 * t["twi"] - 0.02 * clim["temp"], 0, None)
        put(f"Water/GLDAS_SoilMoi_{year}.tif", soil.astype(np.float32), nodata=-9999.0)
    print(f"  水体 1 个 + 土壤水分 {len(FIVE_YEARS)} 个")

    # ---- 社会经济 ----
    rng = np.random.default_rng(args.seed + 7)
    for year in POP_YEARS:
        base = np.zeros(shape)
        base[t["landuse"] == 5] = 900
        base[t["landuse"] == 1] = 130
        base[t["landuse"] == 3] = 12
        base[t["landuse"] == 6] = 1.5
        pop = np.clip(base * (1 + 0.021 * (year - 2000)) + rng.normal(0, 6, shape), 0, None)
        put(f"Population/GPW_pop_{year}.tif", pop.astype(np.float32), nodata=-9999.0)
    for year in FIVE_YEARS:
        built = np.where(t["landuse"] == 5, 0.55 + 0.01 * (year - 2000), 0.02)
        put(f"Built/GHS_built_{year}.tif", built.astype(np.float32), nodata=-9999.0)
    print(f"  人口 {len(POP_YEARS)} 个 + 建成区 {len(FIVE_YEARS)} 个")

    # ---- TerraClimate 五要素 ----
    for year in FIVE_YEARS:
        clim = climate_fields(t, year, args.seed)
        put(f"TerraClimate/TC_pr_{year}.tif", clim["precip"].astype(np.float32), nodata=-9999.0)
        put(f"TerraClimate/TC_tmm_{year}.tif", clim["temp"].astype(np.float32), nodata=-9999.0)
        put(f"TerraClimate/TC_pet_{year}.tif", clim["pet"].astype(np.float32), nodata=-9999.0)
        put(f"TerraClimate/TC_def_{year}.tif", clim["deficit"].astype(np.float32), nodata=-9999.0)
        put(f"TerraClimate/TC_AI_{year}.tif", clim["aridity"].astype(np.float32), nodata=-9999.0)
    print(f"  TerraClimate 5 变量 × {len(FIVE_YEARS)} 年 = {5 * len(FIVE_YEARS)} 个")

    # ---- CMIP6 情景 ----
    hist = climate_fields(t, FIVE_YEARS[0], args.seed)
    put("CMIP6/CMIP6_hist_pr.tif", (hist["precip"] / 365.0).astype(np.float32), nodata=-9999.0)
    put("CMIP6/CMIP6_hist_tas.tif", hist["temp"].astype(np.float32), nodata=-9999.0)
    for ssp, periods in SCENARIOS.items():
        for period, (pr_pct, tas_add) in periods.items():
            put(f"CMIP6/CMIP6_{ssp}_{period}_pr.tif",
                (hist["precip"] * (1 + pr_pct) / 365.0).astype(np.float32), nodata=-9999.0)
            put(f"CMIP6/CMIP6_{ssp}_{period}_tas.tif",
                (hist["temp"] + tas_add).astype(np.float32), nodata=-9999.0)
    print(f"  CMIP6 情景 {2 + sum(len(v) for v in SCENARIOS.values()) * 2} 个")

    total_mb = sum(p.stat().st_size for p in root.rglob("*.tif")) / 1024 / 1024
    print(f"\n✅ 完成：{written} 个文件，{total_mb:.1f} MB，用时 {time.time() - t0:.1f}s")
    print(f"   输出目录：{root.resolve()}")
    print("   启动服务：python -m app.server  →  http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
