from datetime import datetime, timedelta, timezone
from dataclasses import dataclass


def utc_now() -> datetime:
    """Get current UTC time (naive datetime for consistency)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class Config:
    """All pipeline settings. Edit these values to tune behavior."""

    # Date filters — markets must resolve within this window
    min_end_date: datetime = None
    max_end_date: datetime = None
    days_until_max_end: int = 30
    days_until_min_end: int = 1

    # Volume filters
    polymarket_min_volume: int = 5000
    kalshi_min_volume: int = 5000

    # Liquidity filters
    polymarket_min_liquidity: int = 5000
    kalshi_min_liquidity: int = 5000

    # Additional Kalshi filters
    kalshi_min_open_interest: int = 0
    kalshi_min_volume_24h: int = 0

    # Category filters (empty list = include all)
    include_categories: list = None
    exclude_categories: list = None

    # Caching
    cache_expiry_minutes: int = 240
    cache_dir: str = ".cache"

    # Candidate matching — date proximity
    max_date_difference_days: int = 3

    # Embedding model (sentence-transformers)
    # Using all-mpnet-base-v2 instead of gte-large-en-v1.5 to avoid custom embedding layer bugs
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    embedding_subprocess: bool = True

    # Multi-pass matching thresholds
    title_only_threshold: float = 0.88
    full_text_threshold: float = 0.85
    title_with_date_threshold: float = 0.75

    # Legacy single-threshold mode
    cosine_similarity_threshold: float = 0.70
    multi_pass_matching: bool = True

    # LLM verification (Ollama)
    llm_model: str = "qwen2.5:7b"

    # Output directory
    output_dir: str = "./data"

    def __post_init__(self):
        if self.include_categories is None:
            self.include_categories = []
        if self.exclude_categories is None:
            self.exclude_categories = []

        now = utc_now()
        if self.min_end_date is None:
            self.min_end_date = now + timedelta(days=self.days_until_min_end)
        if self.max_end_date is None:
            self.max_end_date = now + timedelta(days=self.days_until_max_end)
