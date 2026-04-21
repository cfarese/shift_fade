## Central config for the app

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataPaths:
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    cache: Path = ROOT / "data" / "cache"

    def __post_init__(self):
        for p in (self.raw, self.processed, self.cache):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class NHLApiConfig:
    base_url: str = "https://api-web.nhle.com/v1"
    ## older stats endpoint still alive for some things the v1 api dropped
    stats_url: str = "https://api.nhle.com/stats/rest/en"
    request_timeout: int = 30
    max_retries: int = 3
    ## sleep between requests so we don't get rate limited
    rate_limit_sleep: float = 0.5


@dataclass
class AppConfig:
    paths: DataPaths = field(default_factory=DataPaths)
    nhl_api: NHLApiConfig = field(default_factory=NHLApiConfig)

    ## seasons to pull, format YYYYYYYY
    seasons: list[str] = field(default_factory=lambda: ["20222023", "20232024", "20242025", "20252026"])

    ## ridge regularization strength, tuned empirically
    ridge_alpha: float = 25.0

    ## stints shorter than this get thrown out as noise
    min_stint_seconds: int = 5

    ## bucket width in seconds for the decay curve
    shift_age_bucket: int = 5

    debug: bool = False
    log_level: str = "INFO"


cfg = AppConfig()
