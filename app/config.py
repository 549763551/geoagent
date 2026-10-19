"""全局配置：统一从 .env 读取数据根目录、输出目录与 LLM 参数。

不硬编码任何密钥；未配置 Key 时智能体自动切换到离线规则模式，仍可完整演示。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv 可选
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """运行期配置。字段带默认值，可直接实例化用于测试。"""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    # 原始遥感数据根目录（内含 NDVI / LST / Reflectance / DEM / 中科院 等子目录）
    data_root: Path = field(default_factory=lambda: _env_path("GEO_DATA_ROOT", PROJECT_ROOT / "raw"))
    # 分析产物输出目录（图、CSV、GeoTIFF、报告）
    output_dir: Path = field(default_factory=lambda: _env_path("GEO_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    # 运行期状态目录（数据目录缓存、长期记忆）
    state_dir: Path = field(default_factory=lambda: _env_path("GEO_STATE_DIR", PROJECT_ROOT / "data"))

    # OpenAI 兼容大模型服务
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "").strip())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-flash"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 180))

    # 智能体循环控制
    max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 14))
    # 逐像元趋势分析的最大处理像元数（超过则自动抽稀，保证交互响应）
    trend_max_pixels: int = field(default_factory=lambda: _env_int("TREND_MAX_PIXELS", 900_000))

    @property
    def llm_enabled(self) -> bool:
        """是否启用大模型。未配置 Key 时返回 False，走离线规则规划。"""
        return bool(self.llm_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
