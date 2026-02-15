import os
import json
import time
import pickle
import logging
from datetime import datetime

import requests

from .models import PolymarketMarket, KalshiMarket

logger = logging.getLogger(__name__)


# --- Caching ---

def get_cache_path(cache_dir: str, platform: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{platform}_markets.pkl")


def is_cache_valid(cache_path: str, expiry_minutes: int) -> bool:
    if not os.path.exists(cache_path) or expiry_minutes <= 0:
        return False
    cache_age_minutes = (time.time() - os.path.getmtime(cache_path)) / 60
    return cache_age_minutes < expiry_minutes


def save_to_cache(cache_path: str, data: list):
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_from_cache(cache_path: str) -> list:
    """Load markets from cache, handling old cache format migration."""
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (AttributeError, ModuleNotFoundError) as e:
        # Cache was saved with old structure - delete it and return empty list
        logger.warning(
            "Cache file has incompatible format (old structure). Deleting cache: %s", e
        )
        try:
            import os
            os.remove(cache_path)
        except Exception:
            pass
        return []


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


# --- Polymarket ---

def fetch_polymarket_markets() -> list[PolymarketMarket]:
    """Fetch all active markets from the Polymarket Gamma API."""
    logger.info("Fetching Polymarket markets...")
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
            logger.error("Error fetching Polymarket markets: %s", e)
            break

        if not data:
            break

        for m in data:
            try:
                end_date = None
                end_date_str = m.get("endDate") or m.get("end_date_iso")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(
                            end_date_str.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
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
                        tags = [
                            t.get("label", t) if isinstance(t, dict) else str(t)
                            for t in m["tags"]
                        ]

                market = PolymarketMarket(
                    condition_id=m.get("conditionId", m.get("condition_id", "")),
                    title=m.get("question", m.get("title", "")),
                    description=m.get("description", ""),
                    outcomes=outcomes,
                    outcome_prices=outcome_prices,
                    end_date=end_date,
                    volume=float(m.get("volumeClob", 0) or m.get("volume", 0) or 0),
                    liquidity=float(
                        m.get("liquidityClob", 0) or m.get("liquidity", 0) or 0
                    ),
                    slug=m.get("slug", ""),
                    tags=tags,
                )
                markets.append(market)
            except Exception as e:
                logger.warning("Error parsing Polymarket market: %s", e)
                continue

        if len(data) < limit:
            break
        offset += limit
        logger.info("  Fetched %d markets so far...", len(markets))

    session.close()
    logger.info("Fetched %d total Polymarket markets", len(markets))
    return markets


# --- Kalshi ---

def fetch_kalshi_markets() -> list[KalshiMarket]:
    """Fetch all active markets from the Kalshi trading API."""
    logger.info("Fetching Kalshi markets...")
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
            logger.error("Error fetching Kalshi markets: %s", e)
            break

        market_list = data.get("markets", [])

        for m in market_list:
            try:
                end_date = None
                end_date_str = m.get("expiration_time") or m.get("close_time")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(
                            end_date_str.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
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
                logger.warning("Error parsing Kalshi market: %s", e)
                continue

        cursor = data.get("cursor")
        if not cursor or not market_list:
            break
        logger.info("  Fetched %d markets so far...", len(markets))

    session.close()
    logger.info("Fetched %d total Kalshi markets", len(markets))
    return markets
