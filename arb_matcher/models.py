from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class PolymarketMarket:
    condition_id: str
    title: str
    description: str
    outcomes: list
    outcome_prices: list
    end_date: Optional[datetime]
    volume: float
    liquidity: float
    slug: str
    tags: list

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"

    @property
    def yes_price(self) -> float:
        if self.outcome_prices and len(self.outcome_prices) > 0:
            try:
                return float(self.outcome_prices[0])
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    @property
    def no_price(self) -> float:
        if self.outcome_prices and len(self.outcome_prices) > 1:
            try:
                return float(self.outcome_prices[1])
            except (ValueError, TypeError):
                return 0.0
        return 0.0


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    rules_primary: str
    rules_secondary: str
    yes_price: float
    no_price: float
    end_date: Optional[datetime]
    volume: float
    volume_24h: float
    liquidity: float
    open_interest: float
    category: str

    @property
    def url(self) -> str:
        return f"https://kalshi.com/events/{self.event_ticker}"

    @property
    def rules(self) -> str:
        parts = [self.rules_primary or "", self.rules_secondary or ""]
        return " ".join(p for p in parts if p).strip()


@dataclass
class MatchCandidate:
    """A potential match identified by cosine similarity (pre-LLM)."""
    poly_market: PolymarketMarket
    kalshi_market: KalshiMarket
    cosine_similarity: float


@dataclass
class VerifiedMatch:
    """A confirmed match after LLM verification, with arb spread data."""
    poly_condition_id: str
    poly_title: str
    poly_description: str
    poly_url: str
    poly_yes_price: float
    poly_no_price: float
    poly_end_date: str
    poly_tags: str

    kalshi_ticker: str
    kalshi_title: str
    kalshi_rules: str
    kalshi_url: str
    kalshi_yes_price: float
    kalshi_no_price: float
    kalshi_end_date: str
    kalshi_category: str

    cosine_similarity: float
    llm_confidence: float
    llm_reasoning: str
    price_diff_pct: float
    match_date: str

    # Cross-venue spread columns
    pypn: float  # poly_yes + poly_no  (should be ~1.0)
    kykn: float  # kalshi_yes + kalshi_no  (should be ~1.0)
    pykn: float  # poly_yes + kalshi_no  (< 1.0 = arb: buy YES poly, NO kalshi)
    kypn: float  # kalshi_yes + poly_no  (< 1.0 = arb: buy YES kalshi, NO poly)
