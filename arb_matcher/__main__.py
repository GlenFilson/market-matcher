"""
Cross-Venue Arbitrage Market Matcher

Usage:
    python -m arb_matcher              # Normal run (uses cache if valid)
    python -m arb_matcher --resume     # Resume from candidates CSV (skip embedding)
    python -m arb_matcher --fresh      # Force fresh API fetch (ignore cache)
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from .config import Config, utc_now
from .api import (
    fetch_polymarket_markets,
    fetch_kalshi_markets,
    get_cache_path,
    is_cache_valid,
    save_to_cache,
    load_from_cache,
    get_cache_age_str,
)
from .matching import (
    filter_polymarket_markets,
    filter_kalshi_markets,
    compute_embeddings,
    compute_embeddings_subprocess,
    find_candidates,
)
from .verification import (
    stop_ollama_model,
    reload_ollama_model,
    verify_candidates,
)
from .db import (
    init_database,
    get_seen_pair_ids,
    get_processed_pair_ids,
    get_all_matches_from_db,
    save_candidates_csv,
    save_matches_csv,
    save_all_matches_csv,
    load_candidates_from_csv,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    config = Config()

    resume_mode = "--resume" in sys.argv
    fresh_mode = "--fresh" in sys.argv

    logger.info("=" * 60)
    logger.info("Cross-Venue Arbitrage Market Matcher")
    logger.info("=" * 60)
    logger.info("Date range: %s to %s", config.min_end_date.date(), config.max_end_date.date())
    logger.info(
        "Volume filters — Poly: %d, Kalshi: %d",
        config.polymarket_min_volume,
        config.kalshi_min_volume,
    )
    logger.info(
        "Liquidity filters — Poly: %d, Kalshi: %d",
        config.polymarket_min_liquidity,
        config.kalshi_min_liquidity,
    )
    if config.include_categories:
        logger.info("Include categories: %s", config.include_categories)
    if config.exclude_categories:
        logger.info("Exclude categories: %s", config.exclude_categories)
    logger.info("")
    logger.info("Embedding model: %s", config.embedding_model)
    logger.info(
        "Embedding device: %s%s",
        config.embedding_device,
        " (subprocess)" if config.embedding_subprocess else "",
    )
    if config.multi_pass_matching:
        logger.info("Matching mode: Multi-pass")
        logger.info("  Title-only threshold: %s", config.title_only_threshold)
        logger.info("  Full-text threshold: %s", config.full_text_threshold)
        logger.info(
            "  Title+date threshold: %s (within %d days)",
            config.title_with_date_threshold,
            config.max_date_difference_days,
        )
    else:
        logger.info(
            "Matching mode: Single threshold (%s)", config.cosine_similarity_threshold
        )
        if config.max_date_difference_days:
            logger.info("  Date proximity filter: %d days", config.max_date_difference_days)
    logger.info("LLM model: %s", config.llm_model)
    logger.info("Cache expiry: %d minutes", config.cache_expiry_minutes)
    if resume_mode:
        logger.info("MODE: Resume from candidates CSV")
    if fresh_mode:
        logger.info("MODE: Fresh fetch (ignoring cache)")
    logger.info("")

    os.makedirs(config.output_dir, exist_ok=True)

    db_path = os.path.join(config.output_dir, "state.db")
    conn = init_database(db_path)

    verified_count = len(get_seen_pair_ids(conn))
    processed_count = len(get_processed_pair_ids(conn))
    logger.info("Previously verified matches: %d", verified_count)
    logger.info("Previously processed candidates: %d", processed_count)
    logger.info("")

    now = utc_now()
    datetime_str = now.strftime("%Y-%m-%d_%H%M%S")
    candidates_filepath = os.path.join(
        config.output_dir, f"arb_candidates_{datetime_str}.csv"
    )

    # --- Fetch or load from cache ---

    poly_cache_path = get_cache_path(config.cache_dir, "polymarket")
    kalshi_cache_path = get_cache_path(config.cache_dir, "kalshi")

    poly_cache_valid = not fresh_mode and is_cache_valid(
        poly_cache_path, config.cache_expiry_minutes
    )
    kalshi_cache_valid = not fresh_mode and is_cache_valid(
        kalshi_cache_path, config.cache_expiry_minutes
    )

    fetch_start = time.time()

    if not poly_cache_valid and not kalshi_cache_valid:
        logger.info("Fetching markets from both platforms in parallel...")
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
            logger.info(
                "Loading Polymarket from cache (%s)...",
                get_cache_age_str(poly_cache_path),
            )
            poly_markets = load_from_cache(poly_cache_path)
            logger.info("Loaded %d Polymarket markets from cache", len(poly_markets))
        else:
            logger.info("Fetching Polymarket markets from API...")
            poly_markets = fetch_polymarket_markets()
            if poly_markets:
                save_to_cache(poly_cache_path, poly_markets)

        if kalshi_cache_valid:
            logger.info(
                "Loading Kalshi from cache (%s)...",
                get_cache_age_str(kalshi_cache_path),
            )
            kalshi_markets = load_from_cache(kalshi_cache_path)
            logger.info("Loaded %d Kalshi markets from cache", len(kalshi_markets))
        else:
            logger.info("Fetching Kalshi markets from API...")
            kalshi_markets = fetch_kalshi_markets()
            if kalshi_markets:
                save_to_cache(kalshi_cache_path, kalshi_markets)

    logger.info("Markets loaded in %.1fs", time.time() - fetch_start)
    logger.info("")

    if not poly_markets or not kalshi_markets:
        logger.error("Failed to fetch markets from one or both platforms")
        return

    # --- Filter ---

    poly_filtered = filter_polymarket_markets(poly_markets, config)
    kalshi_filtered = filter_kalshi_markets(kalshi_markets, config)
    logger.info("")

    if not poly_filtered or not kalshi_filtered:
        logger.warning("No markets remaining after filtering")
        return

    # --- Embed & match (or resume) ---

    if resume_mode and os.path.exists(candidates_filepath):
        logger.info("Loading candidates from %s...", candidates_filepath)
        candidates = load_candidates_from_csv(
            candidates_filepath, poly_filtered, kalshi_filtered
        )
        logger.info("Loaded %d candidates", len(candidates))
    else:
        embed_start = time.time()

        if config.embedding_device == "cuda" and config.embedding_subprocess:
            stop_ollama_model(config.llm_model)
            embeddings = compute_embeddings_subprocess(
                poly_filtered, kalshi_filtered, config
            )
            reload_ollama_model(config.llm_model)
        else:
            if config.embedding_device == "cuda":
                stop_ollama_model(config.llm_model)
            embeddings = compute_embeddings(poly_filtered, kalshi_filtered, config)
            if config.embedding_device == "cuda":
                reload_ollama_model(config.llm_model)

        logger.info("Embeddings completed in %.1fs", time.time() - embed_start)
        logger.info("")

        candidate_start = time.time()
        candidates = find_candidates(poly_filtered, kalshi_filtered, embeddings, config)
        logger.info(
            "Candidate finding completed in %.1fs", time.time() - candidate_start
        )
        logger.info("")

        save_candidates_csv(candidates, candidates_filepath)

    logger.info("")

    if not candidates:
        logger.warning("No candidates found above similarity threshold")
        return

    # --- Verify with LLM ---

    verified_matches = verify_candidates(candidates, config.llm_model, conn)

    # --- Output ---

    all_matches = get_all_matches_from_db(conn)
    all_matches_filepath = os.path.join(config.output_dir, "arb_matches_all.csv")
    save_all_matches_csv(all_matches, all_matches_filepath)

    if verified_matches:
        new_matches_filepath = os.path.join(
            config.output_dir, f"arb_matches_{datetime_str}.csv"
        )
        save_matches_csv(verified_matches, new_matches_filepath)

    conn.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Matches found this run: %d", len(verified_matches))
    logger.info("  Total matches all-time: %d", len(all_matches))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
