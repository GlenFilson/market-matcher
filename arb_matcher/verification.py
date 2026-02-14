"""LLM-based match verification and Ollama GPU management."""

import json
import time
import sqlite3
import subprocess
import logging

import ollama

from .config import Config, utc_now
from .models import PolymarketMarket, KalshiMarket, MatchCandidate, VerifiedMatch
from .db import get_processed_pair_ids, mark_candidate_processed, save_single_match_to_db

logger = logging.getLogger(__name__)


# --- Ollama GPU management ---

def stop_ollama_model(model: str) -> bool:
    """Stop Ollama model to free GPU memory for embedding computation."""
    logger.info("Stopping Ollama to free GPU memory for embeddings...")
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
        time.sleep(2)
        logger.info("Ollama model stopped")
        return True
    except Exception as e:
        logger.warning("Could not stop Ollama model: %s", e)
        return False


def reload_ollama_model(model: str) -> bool:
    """Reload Ollama model to reclaim full GPU allocation after embeddings."""
    logger.info("Reloading Ollama model for full GPU allocation...")
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
        time.sleep(1)
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            options={"num_predict": 1},
        )
        logger.info("Ollama model reloaded successfully")
        return True
    except Exception as e:
        logger.warning("Could not reload Ollama model: %s", e)
        return False


# --- LLM verification ---

_VERIFICATION_PROMPT = """You are evaluating whether two prediction markets from different platforms are asking about the SAME underlying event/question and would resolve the same way.

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


def _verify_single(candidate: MatchCandidate, model: str) -> tuple[bool, float, str]:
    """Send a single candidate pair to the LLM for verification."""
    prompt = _VERIFICATION_PROMPT.format(
        poly_title=candidate.poly_market.title,
        poly_description=candidate.poly_market.description or "N/A",
        poly_end_date=(
            candidate.poly_market.end_date.strftime("%Y-%m-%d")
            if candidate.poly_market.end_date
            else "N/A"
        ),
        kalshi_title=candidate.kalshi_market.title,
        kalshi_rules=candidate.kalshi_market.rules or "N/A",
        kalshi_end_date=(
            candidate.kalshi_market.end_date.strftime("%Y-%m-%d")
            if candidate.kalshi_market.end_date
            else "N/A"
        ),
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )

        text = response["message"]["content"].strip()

        # Handle markdown-wrapped JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        return (
            result.get("is_match", False),
            float(result.get("confidence", 0.0)),
            result.get("reasoning", ""),
        )
    except Exception as e:
        logger.error("LLM verification error: %s", e)
        return False, 0.0, f"Error: {e}"


def _create_verified_match(
    candidate: MatchCandidate, confidence: float, reasoning: str
) -> VerifiedMatch:
    """Build a VerifiedMatch with pre-computed arb spread columns."""
    poly_yes = candidate.poly_market.yes_price
    poly_no = candidate.poly_market.no_price
    kalshi_yes = candidate.kalshi_market.yes_price
    kalshi_no = candidate.kalshi_market.no_price

    if poly_yes > 0 and kalshi_yes > 0:
        price_diff_pct = abs(poly_yes - kalshi_yes) / max(poly_yes, kalshi_yes) * 100
    else:
        price_diff_pct = 0.0

    poly_tags_str = (
        ", ".join(candidate.poly_market.tags) if candidate.poly_market.tags else ""
    )

    return VerifiedMatch(
        poly_condition_id=candidate.poly_market.condition_id,
        poly_title=candidate.poly_market.title,
        poly_description=candidate.poly_market.description or "",
        poly_url=candidate.poly_market.url,
        poly_yes_price=poly_yes,
        poly_no_price=poly_no,
        poly_end_date=(
            candidate.poly_market.end_date.strftime("%Y-%m-%d")
            if candidate.poly_market.end_date
            else ""
        ),
        poly_tags=poly_tags_str,
        kalshi_ticker=candidate.kalshi_market.ticker,
        kalshi_title=candidate.kalshi_market.title,
        kalshi_rules=candidate.kalshi_market.rules or "",
        kalshi_url=candidate.kalshi_market.url,
        kalshi_yes_price=kalshi_yes,
        kalshi_no_price=kalshi_no,
        kalshi_end_date=(
            candidate.kalshi_market.end_date.strftime("%Y-%m-%d")
            if candidate.kalshi_market.end_date
            else ""
        ),
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


def verify_candidates(
    candidates: list[MatchCandidate],
    model: str,
    conn: sqlite3.Connection,
) -> list[VerifiedMatch]:
    """
    Verify all candidates with the LLM, saving progress incrementally.

    Already-processed candidates are skipped (safe to Ctrl+C and resume).
    """
    logger.info("Verifying %d candidates with LLM...", len(candidates))
    logger.info("(Progress saved after each — safe to cancel with Ctrl+C)")

    processed_pairs = get_processed_pair_ids(conn)
    verified: list[VerifiedMatch] = []
    matches_this_run = 0
    skipped = 0

    for i, candidate in enumerate(candidates):
        pair_id = (candidate.poly_market.condition_id, candidate.kalshi_market.ticker)

        if pair_id in processed_pairs:
            skipped += 1
            continue

        logger.info(
            "  Verifying %d/%d:", i + 1, len(candidates),
        )
        logger.info("    Poly:   %s", candidate.poly_market.title[:70])
        logger.info("    Kalshi: %s", candidate.kalshi_market.title[:70])

        is_match, confidence, reasoning = _verify_single(candidate, model)
        mark_candidate_processed(conn, pair_id[0], pair_id[1], is_match)

        if is_match:
            match = _create_verified_match(candidate, confidence, reasoning)
            verified.append(match)
            save_single_match_to_db(conn, match)
            matches_this_run += 1
            logger.info(
                "    ✅ Match confirmed (confidence: %.2f) [Total: %d]",
                confidence,
                matches_this_run,
            )
        else:
            logger.info(
                "    ❌ Not a match: %s... [Total: %d]",
                reasoning[:50],
                matches_this_run,
            )

    if skipped > 0:
        logger.info("Skipped %d already-processed candidates", skipped)
    logger.info("Verified %d true matches this run", len(verified))
    return verified
