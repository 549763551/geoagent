# GeoAgent · 遥感栅格分析智能体

用一句自然语言下达遥感分析任务，智能体自主完成
**任务理解 → 记忆召回 → 自主规划 → 工具调用（ReAct）→ 结果交付** 的全过程，
并实时流式呈现规划时间线、工具执行进度与产物（专题图 / 统计表 / GeoTIFF / 分析报告）。

面向多源栅格数据（NDVI、地表温度、反射率、地形、气候、土地利用等），
不绑定特定区域与特定数据源：把数据按目录约定放好，智能体即可在其上工作。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![rasterio](https://img.shields.io/badge/rasterio-GDAL-4B8BBE)
![LLM](https://img.shields.io/badge/LLM-可选-8B5CF6)

---

## 核心能力

| 能力 | 实现方式 |
| --- | --- |
| 任务理解 | LLM 结构化解析意图（目标 / 任务类型 / 数据集 / 指数 / 年份 / 交付物），未配置 Key 时退化为关键词规则解析 |
| 自主规划 | 生成 1–5 步工具调用计划，执行后由大模型基于中间结果进入 **ReAct 循环**自主追加调用 |
| 工具调用 | 11 个遥感分析工具，以 OpenAI Function Calling（JSON Schema）协议暴露，统一注册表调度 |
| 结果交付 | 自动汇总结论（含关键数值）+ 产物清单；支持一键生成 Markdown / HTML 分析报告 |
| 检索增强 | 数据目录中文别名加权检索，为理解与规划提供「可用数据」上下文 |
| 长短期记忆 | 会话内轨迹（`SessionMemory`）+ 跨会话经验沉淀（`memory.jsonl`），规划时召回历史经验 |
| 工作流编排 | 引擎以事件流（`emit`）驱动 5 阶段流水线，Web 层经 SSE 推送 |
| 多模态产物 | 自然语言输入，交付图文报告、专题图、栅格与表格等多模态结果 |

## 架构

```
                ┌───────────────── 浏览器前端 frontend/ ─────────────────┐
                │  对话界面 · 阶段导轨 · 规划时间线 · 工具卡片 · 产物画廊  │
                └───────▲──────────────────────────────┬───────────────┘
        SSE 事件流      │                              │  POST /api/task
   (session/plan/step/  │                              ▼
    progress/artifact)  │                 ┌──────────────────────────┐
                        │                 │  FastAPI 服务层 server.py │
                        │                 │  工作线程 ←→ asyncio 队列  │
                        │                 └────────────┬─────────────┘
                        │                              ▼
                        │              ┌────────────────────────────────┐
                        └──────────────┤  智能体编排引擎 agent.py        │
                                       │  理解→召回→规划→执行→交付        │
                                       └───┬───────┬───────┬────────┬───┘
                            catalog.py ────┘  llm.py │ memory.py│   │ tools/ (11 工具)
                            数据目录 RAG     大模型   │ 长短期记忆│   工具注册表
                                            (可离线降级)      │
                                                             ▼
                                    outputs/  figures · tables · rasters · reports
```

## 目录结构

```
.
├── app/
│   ├── config.py        # 全局配置（.env 读取、路径、LLM 参数）
│   ├── llm.py           # OpenAI 兼容大模型客户端（可用性探测 + 离线降级）
│   ├── catalog.py       # 数据目录 RAG（扫描数据根目录、中文别名检索、派生指数定义）
│   ├── memory.py        # 长短期记忆（会话轨迹 + memory.jsonl 持久化）
│   ├── agent.py         # 智能体编排引擎（任务理解/规划/ReAct/交付）
│   ├── server.py        # FastAPI 服务（SSE 流式、静态资源、会话接口）
│   └── tools/           # 工具集（11 个，统一注册）
├── frontend/            # 原生 Web 前端（index.html / styles.css / app.js）
├── scripts/
│   └── make_demo_data.py  # 生成合成演示数据（无真实数据也能跑通）
├── raw/                 # 原始遥感数据（不入库，见下方约定）
├── outputs/             # 运行产物（不入库）
├── data/                # 运行期状态：目录缓存、长期记忆（不入库）
├── requirements.txt
└── .env.example
```

## 快速开始

```bash
# 1) 安装依赖（Python 3.10+，3.12 已验证）
python -m pip install -r requirements.txt

# 2) 准备数据：生成一套合成演示数据（几十 MB，几秒钟）
python scripts/make_demo_data.py --out raw

#    或者：把自有遥感数据按「数据目录约定」放进 raw/

# 3) 配置（可选）。不配置 Key 时自动运行在离线规则模式，功能完整可用
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
# 编辑 .env，填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL

# 4) 启动
python -m app.server
```

浏览器打开 **http://127.0.0.1:8000** 即可对话。健康检查：`GET http://127.0.0.1:8000/api/health`。

### 双运行模式

- **在线模式**：配置 `LLM_API_KEY` 后启用。由大模型完成任务理解、规划、ReAct 追加调用与结论撰写。
- **离线模式**：未配置 Key 时启用，或在在线调用失败时**自动降级**。基于关键词规则完成任务理解与工具链规划，
  工具执行、产物生成与流式呈现完全一致，保证无网络 / 无额度环境下也能完整演示。

## 数据集（18 个）

智能体启动时扫描数据根目录，按「子目录 + 文件名模板」识别数据集。
内置 18 个数据集的别名、单位、时间范围定义见 `app/catalog.py` 的 `DATASET_SPECS`。

| key | 说明 | 类型 |
| --- | --- | --- |
| `ndvi` | NDVI 植被指数（年最大值合成，2000–2025） | 连续 |
| `lst` | 地表温度（年均值，2000–2025） | 连续 |
| `reflectance` | 地表反射率（7 波段中值合成，2000–2025） | 连续 |
| `elevation` / `slope` / `twi` | 高程 / 坡度 / 地形湿度指数 | 连续（静态） |
| `soil_moisture` | GLDAS 土壤水分 | 连续 |
| `water_occurrence` | JRC 地表水出现频率 | 连续（静态） |
| `landuse` | 土地利用/覆被（1km 一级类） | 分类 |
| `population` | GPW 人口密度 | 连续 |
| `built` | GHS-BUILT 建成区比例 | 连续 |
| `precip` / `temp` / `pet` / `deficit` / `aridity` | TerraClimate 气候要素 | 连续 |
| `cmip_precip` / `cmip_temp` | CMIP6 未来气候情景（SSP2-4.5 / SSP5-8.5） | 情景 |

派生指数（由反射率 / NDVI / LST 现算）：`NDVI` `NDWI` `MNDWI` `NDBI` `SAVI` `WET` `NDBSI` `RSEI`。

## 工具清单（11 个）

| 工具 | 分类 | 作用 |
| --- | --- | --- |
| `search_catalog` | 数据检索 | 按关键词检索可用数据集 |
| `describe_dataset` | 数据检索 | 查看某数据集的年份/单位/波段等元数据 |
| `compute_index` | 指数计算 | 计算光谱指数或 RSEI 遥感生态指数 |
| `raster_stats` | 统计分析 | 单期/多期描述统计（均值、分位、极值） |
| `time_series` | 统计分析 | 区域均值逐年时间序列 + 折线图 + CSV |
| `zonal_by_landuse` | 统计分析 | 按土地利用一级类分区统计（网格不一致自动重采样） |
| `trend_analysis` | 趋势分析 | 逐像元 Mann-Kendall 检验 + Sen's slope，9 级趋势分级 |
| `change_detection` | 趋势分析 | 两期差分，5 级变化分级 + 面积统计 |
| `scenario_compare` | 情景分析 | CMIP6 多情景区域统计对比（不做像元级差分） |
| `compose_map` | 制图 | 多时相栅格拼成多面板专题对比图 |
| `generate_report` | 结果交付 | 汇总任务轨迹生成 Markdown / HTML 报告 |

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/task` | 提交任务，返回 **SSE 事件流**（body：`{query, session_id?}`） |
| `GET` | `/api/session/{session_id}` | 获取会话状态（理解/计划/步骤/产物/消息） |
| `POST` | `/api/session/reset` | 重置会话并新建 |
| `GET` | `/api/meta` | 智能体元信息（模式、模型、数据集、工具、统计） |
| `GET` | `/api/memory?limit=20` | 长期记忆（倒序） |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/files/...` | 访问 `outputs/` 下的产物文件 |

### SSE 事件类型

`session` → `status` → `understanding` → `recall` → `plan` → `step_start` → `progress` → `tool_result`
→ `reflection`（在线模式）→ `final` → `done`；异常时为 `error`。
事件均为单行 JSON（`data: {...}`），队列空闲时发送 `: keep-alive` 心跳。

## 示例提问

- `分析 2000-2025 年 NDVI 的变化趋势与显著性`
- `对比 2000 年与 2020 年 NDVI 的变化情况`
- `绘制 2020 年 NDVI 的空间分布图`
- `评估 2020 年研究区的生态质量`（触发 RSEI 生态指数计算）
- `统计 2020 年不同土地利用类型的 NDVI 平均值`
- `给出 NDVI 区域平均的时间序列折线图`
- `未来不同情景下降水量将如何变化？`
- `汇总一份植被覆盖分析报告`

> 前端示例文案中的「研究区」由 `frontend/app.js` 顶部的 `STUDY_AREA` 常量控制，
> 改成你自己的区域名称即可。

## 接入自己的数据

1. **放置数据**：按 `app/catalog.py` 中 `DATASET_SPECS` 的 `subdir` 与 `template` 约定命名，
   例如 NDVI 为 `<数据根目录>/NDVI/NDVI_MVC_2020.tif`。
2. **改数据根目录**：默认是项目内的 `raw/`，可用环境变量 `GEO_DATA_ROOT` 指向任意位置。
3. **新增数据集**：在 `DATASET_SPECS` 里追加一条 `DatasetSpec`（key / 子目录 / 文件名模板 / 年份 / 单位 / 别名），
   并在 `ALIASES` 中补充中文别名，检索与规划即刻可用。
4. **改研究区名称**：修改 `frontend/app.js` 的 `STUDY_AREA` 常量。

数据要求：GeoTIFF，同一数据集内网格一致；分类数据建议用 `DatasetSpec.classes` 声明类别字典与 `nodata`。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容接口地址 |
| `LLM_API_KEY` | 空 | 留空则运行离线规则模式 |
| `LLM_MODEL` | `deepseek-flash` | 模型名 |
| `LLM_TEMPERATURE` | `0.2` | 采样温度 |
| `LLM_TIMEOUT` | `180` | 单次请求超时（秒） |
| `AGENT_MAX_STEPS` | `14` | 单任务最大工具调用步数 |
| `TREND_MAX_PIXELS` | `900000` | 趋势分析最大像元数，超出自动抽稀 |
| `GEO_DATA_ROOT` | `./raw` | 原始数据根目录 |
| `GEO_OUTPUT_DIR` | `./outputs` | 产物输出目录 |
| `GEO_STATE_DIR` | `./data` | 运行期状态目录 |

## 技术栈

Python 3.10+ · FastAPI（SSE 流式）· rasterio · NumPy / SciPy / scikit-learn · matplotlib(Agg) · 原生 HTML/CSS/JS

## 数据与声明

- 仓库**不包含任何遥感数据**。`scripts/make_demo_data.py` 生成的是**数学合成的假数据**，
  不映射任何真实地点，仅用于打通功能链路。
- 使用真实数据时，请自行确认数据源的许可与引用要求（如 MODIS / GLDAS / JRC / GPW / GHS /
  TerraClimate / CMIP6 等均有各自的使用条款）。
- `raw/`、`outputs/`、`data/`、`.env` 均已加入 `.gitignore`，不会被误提交。
