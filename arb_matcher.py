#!/usr/bin/env python3
"""
Cross-Venue Arbitrage Market Matcher

Identifies matching prediction markets between Polymarket and Kalshi
for potential arbitrage opportunities.

Usage:
    python arb_matcher.py              # Normal run (uses cache if valid)
    python arb_matcher.py --resume     # Resume from candidates CSV (skip embedding)
    python arb_matcher.py --fresh      # Force fresh API fetch (ignore cache)
"""

import os
import sys
import json
import sqlite3
import time
import pickle
import subprocess
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama

# sentence_transformers and torch are imported lazily in compute_embeddings()
# to avoid reserving GPU memory when using --resume mode


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class Config:
    # Date filters
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

    # Candidate matching
    max_date_difference_days: int = 3

    # Embedding model
    embedding_model: str = "Alibaba-NLP/gte-large-en-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    embedding_subprocess: bool = True

    # Multi-pass matching thresholds
    title_only_threshold: float = 0.88
    full_text_threshold: float = 0.85
    title_with_date_threshold: float = 0.75

    # Legacy single threshold
    cosine_similarity_threshold: float = 0.70
    multi_pass_matching: bool = True

    # LLM
    llm_model: str = "qwen2.5:7b"

    # Output
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


CONFIG = Config()


# --- Data Classes ---

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
    poly_market: PolymarketMarket
    kalshi_market: KalshiMarket
    cosine_similarity: float


@dataclass
class VerifiedMatch:
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

    pypn: float  # poly_yes + poly_no (should be ~1.0)
    kykn: float  # kalshi_yes + kalshi_no (should be ~1.0)
    pykn: float  # poly_yes + kalshi_no (if < 1.0: arb opportunity)
    kypn: float  # kalshi_yes + poly_no (if < 1.0: arb opportunity)


# --- Embedding Text Construction ---

def build_title_text_polymarket(market: PolymarketMarket) -> str:
    return market.title


def build_title_text_kalshi(market: KalshiMarket) -> str:
    return market.title


def build_full_text_polymarket(market: PolymarketMarket) -> str:
    parts = [market.title, market.description or ""]
    if market.end_date:
        parts.append(f"Resolves: {market.end_date.strftime('%Y-%m-%d')}")
    return " ".join(p for p in parts if p).strip()


def build_full_text_kalshi(market: KalshiMarket) -> str:
    parts = [market.title, market.subtitle or "", market.rules or ""]
    if market.end_date:
        parts.append(f"Resolves: {market.end_date.strftime('%Y-%m-%d')}")
    return " ".join(p for p in parts if p).strip()


def build_embedding_text_polymarket(market: PolymarketMarket) -> str:
    return build_full_text_polymarket(market)


def build_embedding_text_kalshi(market: KalshiMarket) -> str:
    return build_full_text_kalshi(market)


# --- Ollama GPU Management ---

def reload_ollama_model(model: str) -> bool:
    print(f"Reloading Ollama model for full GPU allocation...")
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
        time.sleep(1)
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            options={"num_predict": 1},
        )
        print(f"Ollama model reloaded successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not reload Ollama model: {e}")
        return False


def stop_ollama_model(model: str) -> bool:
    print(f"Stopping Ollama to free GPU memory for embeddings...")
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
        time.sleep(2)
        print(f"Ollama model stopped")
        return True
    except Exception as e:
        print(f"Warning: Could not stop Ollama model: {e}")
        return False


# --- Market Caching ---

def get_cache_path(cache_dir: str, platform: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{platform}_markets.pkl")


def is_cache_valid(cache_path: str, expiry_minutes: int) -> bool:
    if not os.path.exists(cache_path) or expiry_minutes <= 0:
        return False
    cache_age_minutes = (time.time() - os.path.getmtime(cache_path)) / 60
    return cache_age_minutes < expiry_minutes


def save_to_cache(cache_path: str, data: list):
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)


def load_from_cache(cache_path: str) -> list:
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def get_cache_age_str(cache_path: str) -> str:
    if not os.path.exists(cache_path):
        return "no cache"
    minutes = int((time.time() - os.path.getmtime(cache_path)) / 60)
    if minutes < 1:
        return "just now"
    elif minutes < 60:
        return f"{minutes} min ago"
    else:
        return f"{minutes // 60}h {minutes % 60}m ago"


# --- API Fetching ---

def fetch_polymarket_markets(config: Config = None) -> list[PolymarketMarket]:
    print("Fetching Polymarket markets...")
    markets = []
    base_url = "https://gamma-api.polymarket.com/markets"
    offset = 0
    limit = 500
    session = requests.Session()

    while True:
        params = {
            "active": "true",
            "closed": "false",
            "archived": "false",
            "limit": limit,
            "offset": offset,
        }

        try:
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching Polymarket markets: {e}")
            break

        if not data:
            break

        for m in data:
            try:
                end_date = None
                end_date_str = m.get("endDate") or m.get("end_date_iso")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except ValueError:
                        pass

                outcomes = m.get("outcomes", [])
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except json.JSONDecodeError:
                        outcomes = []

                outcome_prices = m.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except json.JSONDecodeError:
                        outcome_prices = []

                tags = []
                if m.get("tags"):
                    if isinstance(m["tags"], list):
                        tags = [t.get("label", t) if isinstance(t, dict) else str(t) for t in m["tags"]]

                market = PolymarketMarket(
                    condition_id=m.get("conditionId", m.get("condition_id", "")),
                    title=m.get("question", m.get("title", "")),
                    description=m.get("description", ""),
                    outcomes=outcomes,
                    outcome_prices=outcome_prices,
                    end_date=end_date,
                    volume=float(m.get("volumeClob", 0) or m.get("volume", 0) or 0),
                    liquidity=float(m.get("liquidityClob", 0) or m.get("liquidity", 0) or 0),
                    slug=m.get("slug", ""),
                    tags=tags,
                )
                markets.append(market)
            except Exception as e:
                print(f"Error parsing Polymarket market: {e}")
                continue

        if len(data) < limit:
            break
        offset += limit
        print(f"  Fetched {len(markets)} markets so far...")

    session.close()
    print(f"Fetched {len(markets)} total Polymarket markets")
    return markets


def fetch_kalshi_markets(config: Config = None) -> list[KalshiMarket]:
    print("Fetching Kalshi markets...")
    markets = []
    base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    cursor = None
    session = requests.Session()

    while True:
        params = {"status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        try:
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching Kalshi markets: {e}")
            break

        market_list = data.get("markets", [])

        for m in market_list:
            try:
                end_date = None
                end_date_str = m.get("expiration_time") or m.get("close_time")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except ValueError:
                        pass

                yes_price = float(m.get("yes_bid", 0) or 0) / 100.0
                no_price = float(m.get("no_bid", 0) or 0) / 100.0

                market = KalshiMarket(
                    ticker=m.get("ticker", ""),
                    event_ticker=m.get("event_ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    rules_primary=m.get("rules_primary", ""),
                    rules_secondary=m.get("rules_secondary", ""),
                    yes_price=yes_price,
                    no_price=no_price,
                    end_date=end_date,
                    volume=float(m.get("volume", 0) or 0),
                    volume_24h=float(m.get("volume_24h", 0) or 0),
                    liquidity=float(m.get("liquidity", 0) or 0),
                    open_interest=float(m.get("open_interest", 0) or 0),
                    category=m.get("category", ""),
                )
                markets.append(market)
            except Exception as e:
                print(f"Error parsing Kalshi market: {e}")
                continue

        cursor = data.get("cursor")
        if not cursor or not market_list:
            break
        print(f"  Fetched {len(markets)} markets so far...")

    session.close()
    print(f"Fetched {len(markets)} total Kalshi markets")
    return markets


# --- Filtering ---

def filter_polymarket_markets(markets: list[PolymarketMarket], config: Config) -> list[PolymarketMarket]:
    filtered = []

    for m in markets:
        if m.end_date:
            if m.end_date < config.min_end_date or m.end_date > config.max_end_date:
                continue
        if m.volume < config.polymarket_min_volume:
            continue
        if m.liquidity < config.polymarket_min_liquidity:
            continue
        if config.include_categories:
            market_tags_lower = [t.lower() for t in m.tags]
            if not any(cat.lower() in market_tags_lower for cat in config.include_categories):
                continue
        if config.exclude_categories:
            market_tags_lower = [t.lower() for t in m.tags]
            if any(cat.lower() in market_tags_lower for cat in config.exclude_categories):
                continue
        filtered.append(m)

    print(f"Filtered Polymarket: {len(markets)} -> {len(filtered)} markets")

    tag_counts = {}
    for m in filtered:
        for tag in m.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if tag_counts:
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top Polymarket tags: {', '.join(f'{t}({c})' for t, c in top_tags)}")

    return filtered


def filter_kalshi_markets(markets: list[KalshiMarket], config: Config) -> list[KalshiMarket]:
    filtered = []

    for m in markets:
        if m.end_date:
            if m.end_date < config.min_end_date or m.end_date > config.max_end_date:
                continue
        if m.volume < config.kalshi_min_volume:
            continue
        if m.liquidity < config.kalshi_min_liquidity:
            continue
        if m.open_interest < config.kalshi_min_open_interest:
            continue
        if m.volume_24h < config.kalshi_min_volume_24h:
            continue
        if config.include_categories:
            if m.category.lower() not in [c.lower() for c in config.include_categories]:
                continue
        if config.exclude_categories:
            if m.category.lower() in [c.lower() for c in config.exclude_categories]:
                continue
        filtered.append(m)

    print(f"Filtered Kalshi: {len(markets)} -> {len(filtered)} markets")

    cat_counts = {}
    for m in filtered:
        cat = m.category if m.category else "(uncategorized)"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat_counts and not (len(cat_counts) == 1 and "(uncategorized)" in cat_counts):
        top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top Kalshi categories: {', '.join(f'{c}({n})' for c, n in top_cats)}")

    return filtered


# --- Embedding & Similarity ---

def compute_embeddings_subprocess(
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
    config: Config,
) -> dict:
    """Compute embeddings in a subprocess to ensure complete GPU memory release."""
    import tempfile

    print(f"Computing embeddings in subprocess (device={config.embedding_device})...")

    poly_titles = [build_title_text_polymarket(m) for m in poly_markets]
    kalshi_titles = [build_title_text_kalshi(m) for m in kalshi_markets]
    poly_full = [build_full_text_polymarket(m) for m in poly_markets]
    kalshi_full = [build_full_text_kalshi(m) for m in kalshi_markets]
    all_texts = poly_titles + kalshi_titles + poly_full + kalshi_full

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        texts_path = f.name
        pickle.dump(all_texts, f)

    embeddings_path = texts_path.replace('.pkl', '_emb.npy')

    script = f'''
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

with open("{texts_path}", "rb") as f:
    texts = pickle.load(f)

print(f"Loading model: {config.embedding_model}")
model = SentenceTransformer("{config.embedding_model}", trust_remote_code=True, device="{config.embedding_device}")

print(f"Computing {{len(texts)}} embeddings (batch_size={config.embedding_batch_size})...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size={config.embedding_batch_size})

np.save("{embeddings_path}", embeddings)
print("Done - subprocess exiting, GPU memory will be fully released")
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("Embedding subprocess failed")

    all_embeddings = np.load(embeddings_path)

    os.remove(texts_path)
    os.remove(embeddings_path)

    n_poly = len(poly_markets)
    n_kalshi = len(kalshi_markets)

    idx = 0
    poly_title_emb = all_embeddings[idx:idx + n_poly]; idx += n_poly
    kalshi_title_emb = all_embeddings[idx:idx + n_kalshi]; idx += n_kalshi
    poly_full_emb = all_embeddings[idx:idx + n_poly]; idx += n_poly
    kalshi_full_emb = all_embeddings[idx:idx + n_kalshi]; idx += n_kalshi

    return {
        "poly_title": poly_title_emb,
        "kalshi_title": kalshi_title_emb,
        "poly_full": poly_full_emb,
        "kalshi_full": kalshi_full_emb,
    }


def compute_embeddings(
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
    config: Config,
) -> dict:
    """Compute embeddings in-process. Batches all texts for a single encode call."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {config.embedding_model}")
    print(f"Embedding device: {config.embedding_device}")
    model = SentenceTransformer(config.embedding_model, trust_remote_code=True, device=config.embedding_device)

    print("Building embedding texts...")

    poly_titles = [build_title_text_polymarket(m) for m in poly_markets]
    kalshi_titles = [build_title_text_kalshi(m) for m in kalshi_markets]
    poly_full = [build_full_text_polymarket(m) for m in poly_markets]
    kalshi_full = [build_full_text_kalshi(m) for m in kalshi_markets]

    all_texts = poly_titles + kalshi_titles + poly_full + kalshi_full

    print(f"Computing embeddings for {len(all_texts)} texts (batch_size={config.embedding_batch_size})...")
    all_embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=config.embedding_batch_size)

    n_poly = len(poly_markets)
    n_kalshi = len(kalshi_markets)

    idx = 0
    poly_title_emb = all_embeddings[idx:idx + n_poly]; idx += n_poly
    kalshi_title_emb = all_embeddings[idx:idx + n_kalshi]; idx += n_kalshi
    poly_full_emb = all_embeddings[idx:idx + n_poly]; idx += n_poly
    kalshi_full_emb = all_embeddings[idx:idx + n_kalshi]; idx += n_kalshi

    print("Freeing memory...")
    del model
    import gc
    gc.collect()

    if config.embedding_device == "cuda":
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "poly_title": poly_title_emb,
        "kalshi_title": kalshi_title_emb,
        "poly_full": poly_full_emb,
        "kalshi_full": kalshi_full_emb,
    }


def find_candidates(
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
    embeddings: dict,
    config: Config,
) -> list[MatchCandidate]:
    """
    Find candidate pairs using multi-pass matching:
      1. Title similarity >= title_only_threshold
      2. Full text similarity >= full_text_threshold
      3. Title similarity >= title_with_date_threshold AND dates within max_date_difference_days
    """
    print(f"Computing similarity matrices ({len(poly_markets)} x {len(kalshi_markets)})...")

    def normalize(emb):
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    poly_title_norm = normalize(embeddings["poly_title"])
    kalshi_title_norm = normalize(embeddings["kalshi_title"])
    poly_full_norm = normalize(embeddings["poly_full"])
    kalshi_full_norm = normalize(embeddings["kalshi_full"])

    title_sim = np.dot(poly_title_norm, kalshi_title_norm.T)
    full_sim = np.dot(poly_full_norm, kalshi_full_norm.T)

    candidates = []
    seen_pairs = set()

    # Pre-compute date differences (vectorized)
    date_diffs = None
    if config.max_date_difference_days is not None:
        print(f"Computing date proximity matrix (max {config.max_date_difference_days} days)...")
        poly_dates = np.array([
            m.end_date.timestamp() if m.end_date else np.nan for m in poly_markets
        ])
        kalshi_dates = np.array([
            m.end_date.timestamp() if m.end_date else np.nan for m in kalshi_markets
        ])
        date_diffs = np.abs(poly_dates[:, np.newaxis] - kalshi_dates[np.newaxis, :]) / 86400
        date_diffs = np.nan_to_num(date_diffs, nan=float('inf'))

    if config.multi_pass_matching:
        print("Using multi-pass matching:")
        print(f"  Pass 1: Title similarity >= {config.title_only_threshold}")
        print(f"  Pass 2: Full text similarity >= {config.full_text_threshold}")
        print(f"  Pass 3: Title similarity >= {config.title_with_date_threshold} AND dates within {config.max_date_difference_days} days")

        # Pass 1: Title-only high threshold
        pass1_pairs = np.argwhere(title_sim >= config.title_only_threshold)
        print(f"  Pass 1 found: {len(pass1_pairs)} pairs")
        for poly_idx, kalshi_idx in pass1_pairs:
            pair_key = (poly_idx, kalshi_idx)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                candidates.append(MatchCandidate(
                    poly_market=poly_markets[poly_idx],
                    kalshi_market=kalshi_markets[kalshi_idx],
                    cosine_similarity=float(title_sim[poly_idx, kalshi_idx]),
                ))

        # Pass 2: Full text medium threshold
        pass2_pairs = np.argwhere(full_sim >= config.full_text_threshold)
        pass2_new = 0
        for poly_idx, kalshi_idx in pass2_pairs:
            pair_key = (poly_idx, kalshi_idx)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                pass2_new += 1
                candidates.append(MatchCandidate(
                    poly_market=poly_markets[poly_idx],
                    kalshi_market=kalshi_markets[kalshi_idx],
                    cosine_similarity=float(full_sim[poly_idx, kalshi_idx]),
                ))
        print(f"  Pass 2 found: {pass2_new} new pairs")

        # Pass 3: Title with date proximity
        if date_diffs is not None:
            title_date_mask = (title_sim >= config.title_with_date_threshold) & (date_diffs <= config.max_date_difference_days)
            pass3_pairs = np.argwhere(title_date_mask)
            pass3_new = 0
            for poly_idx, kalshi_idx in pass3_pairs:
                pair_key = (poly_idx, kalshi_idx)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pass3_new += 1
                    candidates.append(MatchCandidate(
                        poly_market=poly_markets[poly_idx],
                        kalshi_market=kalshi_markets[kalshi_idx],
                        cosine_similarity=float(title_sim[poly_idx, kalshi_idx]),
                    ))
            print(f"  Pass 3 found: {pass3_new} new pairs")

    else:
        # Legacy single-threshold matching
        print(f"Using single threshold: {config.cosine_similarity_threshold}")
        pairs_above_threshold = np.argwhere(full_sim >= config.cosine_similarity_threshold)

        if date_diffs is not None:
            filtered_pairs = []
            for poly_idx, kalshi_idx in pairs_above_threshold:
                if date_diffs[poly_idx, kalshi_idx] <= config.max_date_difference_days:
                    filtered_pairs.append((poly_idx, kalshi_idx))
            pairs_above_threshold = filtered_pairs
            print(f"After date filter: {len(pairs_above_threshold)} pairs")

        for poly_idx, kalshi_idx in pairs_above_threshold:
            candidates.append(MatchCandidate(
                poly_market=poly_markets[poly_idx],
                kalshi_market=kalshi_markets[kalshi_idx],
                cosine_similarity=float(full_sim[poly_idx, kalshi_idx]),
            ))

    print(f"Total candidates: {len(candidates)}")
    candidates.sort(key=lambda x: x.cosine_similarity, reverse=True)
    return candidates


# --- LLM Verification ---

LLM_VERIFICATION_PROMPT = """You are evaluating whether two prediction markets from different platforms are asking about the SAME underlying event/question and would resolve the same way.

MARKET A (Polymarket):
Title: {poly_title}
Description: {poly_description}
Resolution Date: {poly_end_date}

MARKET B (Kalshi):
Title: {kalshi_title}
Rules: {kalshi_rules}
Resolution Date: {kalshi_end_date}

Evaluate whether these markets:
1. Ask about the same underlying event/question
2. Would resolve YES/NO in the same way (not inverted)
3. Have compatible resolution criteria (minor date differences of a few days are acceptable)

Respond with a JSON object only, no other text:
{{
    "is_match": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""


def verify_with_llm(candidate: MatchCandidate, model: str) -> tuple[bool, float, str]:
    prompt = LLM_VERIFICATION_PROMPT.format(
        poly_title=candidate.poly_market.title,
        poly_description=candidate.poly_market.description or "N/A",
        poly_end_date=candidate.poly_market.end_date.strftime("%Y-%m-%d") if candidate.poly_market.end_date else "N/A",
        kalshi_title=candidate.kalshi_market.title,
        kalshi_rules=candidate.kalshi_market.rules or "N/A",
        kalshi_end_date=candidate.kalshi_market.end_date.strftime("%Y-%m-%d") if candidate.kalshi_market.end_date else "N/A",
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )

        response_text = response["message"]["content"].strip()

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return (
            result.get("is_match", False),
            float(result.get("confidence", 0.0)),
            result.get("reasoning", ""),
        )
    except Exception as e:
        print(f"LLM verification error: {e}")
        return False, 0.0, f"Error: {e}"


def create_verified_match(candidate: MatchCandidate, confidence: float, reasoning: str) -> VerifiedMatch:
    poly_yes = candidate.poly_market.yes_price
    poly_no = candidate.poly_market.no_price
    kalshi_yes = candidate.kalshi_market.yes_price
    kalshi_no = candidate.kalshi_market.no_price

    if poly_yes > 0 and kalshi_yes > 0:
        price_diff_pct = abs(poly_yes - kalshi_yes) / max(poly_yes, kalshi_yes) * 100
    else:
        price_diff_pct = 0.0

    poly_tags_str = ", ".join(candidate.poly_market.tags) if candidate.poly_market.tags else ""

    return VerifiedMatch(
        poly_condition_id=candidate.poly_market.condition_id,
        poly_title=candidate.poly_market.title,
        poly_description=candidate.poly_market.description or "",
        poly_url=candidate.poly_market.url,
        poly_yes_price=poly_yes,
        poly_no_price=poly_no,
        poly_end_date=candidate.poly_market.end_date.strftime("%Y-%m-%d") if candidate.poly_market.end_date else "",
        poly_tags=poly_tags_str,
        kalshi_ticker=candidate.kalshi_market.ticker,
        kalshi_title=candidate.kalshi_market.title,
        kalshi_rules=candidate.kalshi_market.rules or "",
        kalshi_url=candidate.kalshi_market.url,
        kalshi_yes_price=kalshi_yes,
        kalshi_no_price=kalshi_no,
        kalshi_end_date=candidate.kalshi_market.end_date.strftime("%Y-%m-%d") if candidate.kalshi_market.end_date else "",
        kalshi_category=candidate.kalshi_market.category,
        cosine_similarity=candidate.cosine_similarity,
        llm_confidence=confidence,
        llm_reasoning=reasoning,
        price_diff_pct=price_diff_pct,
        match_date=utc_now().strftime("%Y-%m-%d"),
        pypn=poly_yes + poly_no,
        kykn=kalshi_yes + kalshi_no,
        pykn=poly_yes + kalshi_no,
        kypn=kalshi_yes + poly_no,
    )


def verify_candidates_incremental(
    candidates: list[MatchCandidate],
    model: str,
    conn: sqlite3.Connection,
) -> list[VerifiedMatch]:
    """Verify candidates with LLM, saving progress incrementally (safe to Ctrl+C)."""
    print(f"Verifying {len(candidates)} candidates with LLM...")
    print("(Progress is saved after each check - safe to cancel with Ctrl+C)")
    print()

    processed_pairs = get_processed_pair_ids(conn)
    verified = []
    matches_this_run = 0
    skipped = 0

    for i, candidate in enumerate(candidates):
        pair_id = (candidate.poly_market.condition_id, candidate.kalshi_market.ticker)

        if pair_id in processed_pairs:
            skipped += 1
            continue

        print(f"  Verifying {i+1}/{len(candidates)}:")
        print(f"    Poly:   {candidate.poly_market.title[:70]}")
        print(f"    Kalshi: {candidate.kalshi_market.title[:70]}")

        is_match, confidence, reasoning = verify_with_llm(candidate, model)
        mark_candidate_processed(conn, pair_id[0], pair_id[1], is_match)

        if is_match:
            match = create_verified_match(candidate, confidence, reasoning)
            verified.append(match)
            save_single_match_to_db(conn, match)
            matches_this_run += 1
            print(f"    ✅ Match confirmed (confidence: {confidence:.2f}) [Total: {matches_this_run}]")
        else:
            print(f"    ❌ Not a match: {reasoning[:50]}... [Total: {matches_this_run}] ")

    print()
    if skipped > 0:
        print(f"Skipped {skipped} already-processed candidates")
    print(f"Verified {len(verified)} true matches this run")
    return verified


def save_single_match_to_db(conn: sqlite3.Connection, match: VerifiedMatch) -> bool:
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO verified_matches (
                poly_condition_id, kalshi_ticker, poly_title, poly_description,
                poly_url, poly_yes_price, poly_no_price, poly_end_date, poly_tags,
                kalshi_title, kalshi_rules, kalshi_url, kalshi_yes_price, kalshi_no_price,
                kalshi_end_date, kalshi_category, cosine_similarity, llm_confidence,
                llm_reasoning, price_diff_pct, match_date, pypn, kykn, pykn, kypn
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match.poly_condition_id, match.kalshi_ticker, match.poly_title,
            match.poly_description, match.poly_url, match.poly_yes_price,
            match.poly_no_price, match.poly_end_date, match.poly_tags,
            match.kalshi_title, match.kalshi_rules, match.kalshi_url,
            match.kalshi_yes_price, match.kalshi_no_price, match.kalshi_end_date,
            match.kalshi_category, match.cosine_similarity, match.llm_confidence,
            match.llm_reasoning, match.price_diff_pct, match.match_date,
            match.pypn, match.kykn, match.pykn, match.kypn,
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Error saving match: {e}")
        return False


# --- State Management (SQLite) ---

def init_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verified_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            poly_condition_id TEXT,
            kalshi_ticker TEXT,
            poly_title TEXT,
            poly_description TEXT,
            poly_url TEXT,
            poly_yes_price REAL,
            poly_no_price REAL,
            poly_end_date TEXT,
            poly_tags TEXT,
            kalshi_title TEXT,
            kalshi_rules TEXT,
            kalshi_url TEXT,
            kalshi_yes_price REAL,
            kalshi_no_price REAL,
            kalshi_end_date TEXT,
            kalshi_category TEXT,
            cosine_similarity REAL,
            llm_confidence REAL,
            llm_reasoning TEXT,
            price_diff_pct REAL,
            match_date TEXT,
            pypn REAL,
            kykn REAL,
            pykn REAL,
            kypn REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(poly_condition_id, kalshi_ticker)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            poly_condition_id TEXT,
            kalshi_ticker TEXT,
            is_match INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(poly_condition_id, kalshi_ticker)
        )
    """)

    conn.commit()
    return conn


def get_seen_pair_ids(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    cursor = conn.cursor()
    cursor.execute("SELECT poly_condition_id, kalshi_ticker FROM verified_matches")
    return set(cursor.fetchall())


def get_processed_pair_ids(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    cursor = conn.cursor()
    cursor.execute("SELECT poly_condition_id, kalshi_ticker FROM processed_candidates")
    return set(cursor.fetchall())


def mark_candidate_processed(conn: sqlite3.Connection, poly_condition_id: str, kalshi_ticker: str, is_match: bool):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO processed_candidates (poly_condition_id, kalshi_ticker, is_match)
            VALUES (?, ?, ?)
        """, (poly_condition_id, kalshi_ticker, 1 if is_match else 0))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error marking candidate processed: {e}")


def save_matches_to_db(conn: sqlite3.Connection, matches: list[VerifiedMatch]) -> int:
    cursor = conn.cursor()
    new_count = 0

    for match in matches:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO verified_matches (
                    poly_condition_id, kalshi_ticker, poly_title, poly_description,
                    poly_url, poly_yes_price, poly_end_date, kalshi_title, kalshi_rules,
                    kalshi_url, kalshi_yes_price, kalshi_end_date, cosine_similarity,
                    llm_confidence, llm_reasoning, price_diff_pct, match_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match.poly_condition_id, match.kalshi_ticker, match.poly_title,
                match.poly_description, match.poly_url, match.poly_yes_price,
                match.poly_end_date, match.kalshi_title, match.kalshi_rules,
                match.kalshi_url, match.kalshi_yes_price, match.kalshi_end_date,
                match.cosine_similarity, match.llm_confidence, match.llm_reasoning,
                match.price_diff_pct, match.match_date,
            ))
            if cursor.rowcount > 0:
                new_count += 1
        except sqlite3.Error as e:
            print(f"Error saving match: {e}")

    conn.commit()
    return new_count


def get_all_matches_from_db(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM verified_matches ORDER BY match_date DESC")
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# --- CSV Output ---

def save_candidates_csv(candidates: list[MatchCandidate], filepath: str):
    rows = []
    for c in candidates:
        poly_tags_str = ", ".join(c.poly_market.tags) if c.poly_market.tags else ""
        rows.append({
            "poly_condition_id": c.poly_market.condition_id,
            "poly_title": c.poly_market.title,
            "poly_tags": poly_tags_str,
            "poly_url": c.poly_market.url,
            "kalshi_ticker": c.kalshi_market.ticker,
            "kalshi_title": c.kalshi_market.title,
            "kalshi_category": c.kalshi_market.category,
            "kalshi_url": c.kalshi_market.url,
            "cosine_similarity": c.cosine_similarity,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(rows)} candidates to {filepath}")


def save_matches_csv(matches: list[VerifiedMatch], filepath: str):
    rows = [asdict(m) for m in matches]
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(rows)} matches to {filepath}")


def save_all_matches_csv(matches: list[dict], filepath: str):
    df = pd.DataFrame(matches)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(matches)} total matches to {filepath}")


# --- Resume Helper ---

def load_candidates_from_csv(
    filepath: str,
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
) -> list[MatchCandidate]:
    df = pd.read_csv(filepath)
    poly_lookup = {m.condition_id: m for m in poly_markets}
    kalshi_lookup = {m.ticker: m for m in kalshi_markets}

    candidates = []
    for _, row in df.iterrows():
        poly_id = row["poly_condition_id"]
        kalshi_ticker = row["kalshi_ticker"]
        if poly_id in poly_lookup and kalshi_ticker in kalshi_lookup:
            candidates.append(MatchCandidate(
                poly_market=poly_lookup[poly_id],
                kalshi_market=kalshi_lookup[kalshi_ticker],
                cosine_similarity=row["cosine_similarity"],
            ))
    return candidates


# --- Main Pipeline ---

def main():
    resume_mode = "--resume" in sys.argv
    fresh_mode = "--fresh" in sys.argv

    print("=" * 60)
    print("Cross-Venue Arbitrage Market Matcher")
    print("=" * 60)
    print(f"Date range: {CONFIG.min_end_date.date()} to {CONFIG.max_end_date.date()}")
    print(f"Volume filters - Poly: {CONFIG.polymarket_min_volume}, Kalshi: {CONFIG.kalshi_min_volume}")
    print(f"Liquidity filters - Poly: {CONFIG.polymarket_min_liquidity}, Kalshi: {CONFIG.kalshi_min_liquidity}")
    if CONFIG.include_categories:
        print(f"Include categories: {CONFIG.include_categories}")
    if CONFIG.exclude_categories:
        print(f"Exclude categories: {CONFIG.exclude_categories}")
    print()
    print(f"Embedding model: {CONFIG.embedding_model}")
    print(f"Embedding device: {CONFIG.embedding_device}" + (" (subprocess)" if CONFIG.embedding_subprocess else ""))
    if CONFIG.multi_pass_matching:
        print("Matching mode: Multi-pass")
        print(f"  Title-only threshold: {CONFIG.title_only_threshold}")
        print(f"  Full-text threshold: {CONFIG.full_text_threshold}")
        print(f"  Title+date threshold: {CONFIG.title_with_date_threshold} (within {CONFIG.max_date_difference_days} days)")
    else:
        print(f"Matching mode: Single threshold ({CONFIG.cosine_similarity_threshold})")
        if CONFIG.max_date_difference_days:
            print(f"  Date proximity filter: {CONFIG.max_date_difference_days} days")
    print(f"LLM model: {CONFIG.llm_model}")
    print(f"Cache expiry: {CONFIG.cache_expiry_minutes} minutes")
    if resume_mode:
        print("MODE: Resume from candidates CSV")
    if fresh_mode:
        print("MODE: Fresh fetch (ignoring cache)")
    print()

    os.makedirs(CONFIG.output_dir, exist_ok=True)

    db_path = os.path.join(CONFIG.output_dir, "state.db")
    conn = init_database(db_path)

    verified_count = len(get_seen_pair_ids(conn))
    processed_count = len(get_processed_pair_ids(conn))
    print(f"Previously verified matches: {verified_count}")
    print(f"Previously processed candidates: {processed_count}")
    print()

    now = utc_now()
    datetime_str = now.strftime("%Y-%m-%d_%H%M%S")
    candidates_filepath = os.path.join(CONFIG.output_dir, f"arb_candidates_{datetime_str}.csv")

    poly_cache_path = get_cache_path(CONFIG.cache_dir, "polymarket")
    kalshi_cache_path = get_cache_path(CONFIG.cache_dir, "kalshi")

    poly_cache_valid = not fresh_mode and is_cache_valid(poly_cache_path, CONFIG.cache_expiry_minutes)
    kalshi_cache_valid = not fresh_mode and is_cache_valid(kalshi_cache_path, CONFIG.cache_expiry_minutes)

    fetch_start = time.time()

    if not poly_cache_valid and not kalshi_cache_valid:
        print("Fetching markets from both platforms in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            poly_future = executor.submit(fetch_polymarket_markets)
            kalshi_future = executor.submit(fetch_kalshi_markets)
            poly_markets = poly_future.result()
            kalshi_markets = kalshi_future.result()

        if poly_markets:
            save_to_cache(poly_cache_path, poly_markets)
        if kalshi_markets:
            save_to_cache(kalshi_cache_path, kalshi_markets)
    else:
        if poly_cache_valid:
            print(f"Loading Polymarket markets from cache ({get_cache_age_str(poly_cache_path)})...")
            poly_markets = load_from_cache(poly_cache_path)
            print(f"Loaded {len(poly_markets)} Polymarket markets from cache")
        else:
            print("Fetching Polymarket markets from API...")
            poly_markets = fetch_polymarket_markets()
            if poly_markets:
                save_to_cache(poly_cache_path, poly_markets)

        if kalshi_cache_valid:
            print(f"Loading Kalshi markets from cache ({get_cache_age_str(kalshi_cache_path)})...")
            kalshi_markets = load_from_cache(kalshi_cache_path)
            print(f"Loaded {len(kalshi_markets)} Kalshi markets from cache")
        else:
            print("Fetching Kalshi markets from API...")
            kalshi_markets = fetch_kalshi_markets()
            if kalshi_markets:
                save_to_cache(kalshi_cache_path, kalshi_markets)

    fetch_time = time.time() - fetch_start
    print(f"Markets loaded in {fetch_time:.1f}s")
    print()

    if not poly_markets or not kalshi_markets:
        print("Error: Failed to fetch markets from one or both platforms")
        return

    poly_filtered = filter_polymarket_markets(poly_markets, CONFIG)
    kalshi_filtered = filter_kalshi_markets(kalshi_markets, CONFIG)
    print()

    if not poly_filtered or not kalshi_filtered:
        print("No markets remaining after filtering")
        return

    if resume_mode and os.path.exists(candidates_filepath):
        print(f"Loading candidates from {candidates_filepath}...")
        candidates = load_candidates_from_csv(candidates_filepath, poly_filtered, kalshi_filtered)
        print(f"Loaded {len(candidates)} candidates")
    else:
        embed_start = time.time()

        if CONFIG.embedding_device == "cuda" and CONFIG.embedding_subprocess:
            stop_ollama_model(CONFIG.llm_model)
            embeddings = compute_embeddings_subprocess(poly_filtered, kalshi_filtered, CONFIG)
            reload_ollama_model(CONFIG.llm_model)
        else:
            if CONFIG.embedding_device == "cuda":
                stop_ollama_model(CONFIG.llm_model)
            embeddings = compute_embeddings(poly_filtered, kalshi_filtered, CONFIG)
            if CONFIG.embedding_device == "cuda":
                reload_ollama_model(CONFIG.llm_model)

        embed_time = time.time() - embed_start
        print(f"Embeddings completed in {embed_time:.1f}s")
        print()

        candidate_start = time.time()
        candidates = find_candidates(poly_filtered, kalshi_filtered, embeddings, CONFIG)
        candidate_time = time.time() - candidate_start
        print(f"Candidate finding completed in {candidate_time:.1f}s")
        print()

        save_candidates_csv(candidates, candidates_filepath)

    print()

    if not candidates:
        print("No candidates found above similarity threshold")
        return

    verified_matches = verify_candidates_incremental(candidates, CONFIG.llm_model, conn)

    all_matches = get_all_matches_from_db(conn)
    all_matches_filepath = os.path.join(CONFIG.output_dir, "arb_matches_all.csv")
    save_all_matches_csv(all_matches, all_matches_filepath)

    if verified_matches:
        new_matches_filepath = os.path.join(CONFIG.output_dir, f"arb_matches_{datetime_str}.csv")
        save_matches_csv(verified_matches, new_matches_filepath)

    conn.close()

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Matches found this run: {len(verified_matches)}")
    print(f"  Total matches all-time: {len(all_matches)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
