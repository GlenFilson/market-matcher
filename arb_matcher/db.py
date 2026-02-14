"""SQLite state management and CSV output."""

import sqlite3
import logging

import pandas as pd
from dataclasses import asdict

from .models import PolymarketMarket, KalshiMarket, MatchCandidate, VerifiedMatch

logger = logging.getLogger(__name__)


# --- Database init ---

def init_database(db_path: str) -> sqlite3.Connection:
    """Create tables if they don't exist and return a connection."""
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


# --- Queries ---

def get_seen_pair_ids(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Return set of (poly_condition_id, kalshi_ticker) for verified matches."""
    cursor = conn.cursor()
    cursor.execute("SELECT poly_condition_id, kalshi_ticker FROM verified_matches")
    return set(cursor.fetchall())


def get_processed_pair_ids(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Return set of (poly_condition_id, kalshi_ticker) for all processed candidates."""
    cursor = conn.cursor()
    cursor.execute("SELECT poly_condition_id, kalshi_ticker FROM processed_candidates")
    return set(cursor.fetchall())


def mark_candidate_processed(
    conn: sqlite3.Connection,
    poly_condition_id: str,
    kalshi_ticker: str,
    is_match: bool,
):
    """Record that a candidate has been processed (match or not)."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO processed_candidates
                (poly_condition_id, kalshi_ticker, is_match)
            VALUES (?, ?, ?)
            """,
            (poly_condition_id, kalshi_ticker, 1 if is_match else 0),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error("Error marking candidate processed: %s", e)


def save_single_match_to_db(conn: sqlite3.Connection, match: VerifiedMatch) -> bool:
    """Persist a single verified match immediately."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO verified_matches (
                poly_condition_id, kalshi_ticker, poly_title, poly_description,
                poly_url, poly_yes_price, poly_no_price, poly_end_date, poly_tags,
                kalshi_title, kalshi_rules, kalshi_url, kalshi_yes_price, kalshi_no_price,
                kalshi_end_date, kalshi_category, cosine_similarity, llm_confidence,
                llm_reasoning, price_diff_pct, match_date, pypn, kykn, pykn, kypn
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match.poly_condition_id, match.kalshi_ticker, match.poly_title,
                match.poly_description, match.poly_url, match.poly_yes_price,
                match.poly_no_price, match.poly_end_date, match.poly_tags,
                match.kalshi_title, match.kalshi_rules, match.kalshi_url,
                match.kalshi_yes_price, match.kalshi_no_price, match.kalshi_end_date,
                match.kalshi_category, match.cosine_similarity, match.llm_confidence,
                match.llm_reasoning, match.price_diff_pct, match.match_date,
                match.pypn, match.kykn, match.pykn, match.kypn,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Error saving match: %s", e)
        return False


def get_all_matches_from_db(conn: sqlite3.Connection) -> list[dict]:
    """Retrieve all verified matches as a list of dicts."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM verified_matches ORDER BY match_date DESC")
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# --- CSV I/O ---

def save_candidates_csv(candidates: list[MatchCandidate], filepath: str):
    """Write pre-LLM candidate pairs to CSV."""
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
    pd.DataFrame(rows).to_csv(filepath, index=False)
    logger.info("Saved %d candidates to %s", len(rows), filepath)


def save_matches_csv(matches: list[VerifiedMatch], filepath: str):
    """Write LLM-verified matches to CSV."""
    rows = [asdict(m) for m in matches]
    pd.DataFrame(rows).to_csv(filepath, index=False)
    logger.info("Saved %d matches to %s", len(rows), filepath)


def save_all_matches_csv(matches: list[dict], filepath: str):
    """Write cumulative match history to CSV."""
    pd.DataFrame(matches).to_csv(filepath, index=False)
    logger.info("Saved %d total matches to %s", len(matches), filepath)


def load_candidates_from_csv(
    filepath: str,
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
) -> list[MatchCandidate]:
    """Reload candidate pairs from a previously saved CSV."""
    df = pd.read_csv(filepath)
    poly_lookup = {m.condition_id: m for m in poly_markets}
    kalshi_lookup = {m.ticker: m for m in kalshi_markets}

    candidates = []
    for _, row in df.iterrows():
        poly_id = row["poly_condition_id"]
        kalshi_ticker = row["kalshi_ticker"]
        if poly_id in poly_lookup and kalshi_ticker in kalshi_lookup:
            candidates.append(
                MatchCandidate(
                    poly_market=poly_lookup[poly_id],
                    kalshi_market=kalshi_lookup[kalshi_ticker],
                    cosine_similarity=row["cosine_similarity"],
                )
            )
    return candidates
