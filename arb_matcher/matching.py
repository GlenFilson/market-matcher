"""Market filtering, embedding computation, and candidate pair discovery."""

import os
import sys
import pickle
import subprocess
import logging

import numpy as np

from .config import Config
from .models import PolymarketMarket, KalshiMarket, MatchCandidate

logger = logging.getLogger(__name__)


# --- Filtering ---

def filter_polymarket_markets(
    markets: list[PolymarketMarket], config: Config
) -> list[PolymarketMarket]:
    """Apply date, volume, liquidity, and category filters to Polymarket markets."""
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
            if not any(
                cat.lower() in market_tags_lower for cat in config.include_categories
            ):
                continue
        if config.exclude_categories:
            market_tags_lower = [t.lower() for t in m.tags]
            if any(
                cat.lower() in market_tags_lower for cat in config.exclude_categories
            ):
                continue
        filtered.append(m)

    logger.info("Filtered Polymarket: %d -> %d markets", len(markets), len(filtered))

    tag_counts: dict[str, int] = {}
    for m in filtered:
        for tag in m.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if tag_counts:
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(
            "  Top Polymarket tags: %s",
            ", ".join(f"{t}({c})" for t, c in top_tags),
        )

    return filtered


def filter_kalshi_markets(
    markets: list[KalshiMarket], config: Config
) -> list[KalshiMarket]:
    """Apply date, volume, liquidity, and category filters to Kalshi markets."""
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

    logger.info("Filtered Kalshi: %d -> %d markets", len(markets), len(filtered))

    cat_counts: dict[str, int] = {}
    for m in filtered:
        cat = m.category if m.category else "(uncategorized)"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat_counts and not (len(cat_counts) == 1 and "(uncategorized)" in cat_counts):
        top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(
            "  Top Kalshi categories: %s",
            ", ".join(f"{c}({n})" for c, n in top_cats),
        )

    return filtered


# --- Embedding text construction ---

# Hard character cap to prevent tokenizer from producing sequences longer than
# the model's position-embedding table.  500 chars ≈ ~125 tokens, extremely conservative
# to avoid CUDA index-out-of-bounds errors with custom model implementations.
_MAX_TEXT_CHARS = 500


def _title_text_polymarket(market: PolymarketMarket) -> str:
    return market.title[:_MAX_TEXT_CHARS]


def _title_text_kalshi(market: KalshiMarket) -> str:
    return market.title[:_MAX_TEXT_CHARS]


def _full_text_polymarket(market: PolymarketMarket) -> str:
    parts = [market.title, market.description or ""]
    if market.end_date:
        parts.append(f"Resolves: {market.end_date.strftime('%Y-%m-%d')}")
    return " ".join(p for p in parts if p).strip()[:_MAX_TEXT_CHARS]


def _full_text_kalshi(market: KalshiMarket) -> str:
    parts = [market.title, market.subtitle or "", market.rules or ""]
    if market.end_date:
        parts.append(f"Resolves: {market.end_date.strftime('%Y-%m-%d')}")
    return " ".join(p for p in parts if p).strip()[:_MAX_TEXT_CHARS]


# --- Embedding computation ---

def compute_embeddings_subprocess(
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
    config: Config,
) -> dict:
    """Compute embeddings in a subprocess to ensure complete GPU memory release."""
    import tempfile

    logger.info(
        "Computing embeddings in subprocess (device=%s)...", config.embedding_device
    )

    poly_titles = [_title_text_polymarket(m) for m in poly_markets]
    kalshi_titles = [_title_text_kalshi(m) for m in kalshi_markets]
    poly_full = [_full_text_polymarket(m) for m in poly_markets]
    kalshi_full = [_full_text_kalshi(m) for m in kalshi_markets]
    all_texts = poly_titles + kalshi_titles + poly_full + kalshi_full

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        texts_path = f.name
        pickle.dump(all_texts, f)

    embeddings_path = texts_path.replace(".pkl", "_emb.npy")

    script = f'''
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

with open("{texts_path}", "rb") as f:
    texts = pickle.load(f)

print(f"Loading model: {config.embedding_model}")
# Standard models don't need trust_remote_code=True
model = SentenceTransformer(
    "{config.embedding_model}", device="{config.embedding_device}"
)

# Standard models handle truncation properly, use model's max_seq_length
safe_max = model.max_seq_length
print(f"Using model's max_seq_length: {{safe_max}}")

# Standard models handle truncation properly - just use the texts as-is
print(f"Computing embeddings for {{len(texts)}} texts...")

print(f"Computing embeddings (batch_size={config.embedding_batch_size}, max_seq={{safe_max}})...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size={config.embedding_batch_size},
    convert_to_numpy=True,
    normalize_embeddings=False,
)

np.save("{embeddings_path}", embeddings)
print("Done - subprocess exiting, GPU memory will be fully released")
'''

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=False, text=True
    )

    if result.returncode != 0:
        raise RuntimeError("Embedding subprocess failed")

    all_embeddings = np.load(embeddings_path)

    os.remove(texts_path)
    os.remove(embeddings_path)

    return _split_embeddings(all_embeddings, len(poly_markets), len(kalshi_markets))


def compute_embeddings(
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
    config: Config,
) -> dict:
    """Compute embeddings in-process. Batches all texts for a single encode() call."""
    # Lazy import to avoid reserving GPU memory when using --resume
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", config.embedding_model)
    logger.info("Embedding device: %s", config.embedding_device)
    model = SentenceTransformer(
        config.embedding_model, device=config.embedding_device
    )
    
    # Standard models handle truncation properly
    safe_max = model.max_seq_length
    logger.info("Using model's max_seq_length: %d", safe_max)
    
    model.max_seq_length = safe_max
    if hasattr(model.tokenizer, "model_max_length"):
        model.tokenizer.model_max_length = safe_max

    logger.info("Building embedding texts...")

    poly_titles = [_title_text_polymarket(m) for m in poly_markets]
    kalshi_titles = [_title_text_kalshi(m) for m in kalshi_markets]
    poly_full = [_full_text_polymarket(m) for m in poly_markets]
    kalshi_full = [_full_text_kalshi(m) for m in kalshi_markets]

    all_texts = poly_titles + kalshi_titles + poly_full + kalshi_full

    # Standard models handle truncation properly - no manual truncation needed
    logger.info("Computing embeddings for %d texts...", len(all_texts))

    logger.info(
        "Computing embeddings for %d texts (batch_size=%d, max_seq=%d)...",
        len(all_texts),
        config.embedding_batch_size,
        safe_max,
    )
    all_embeddings = model.encode(
        all_texts,
        show_progress_bar=True,
        batch_size=config.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    logger.info("Freeing embedding model memory...")
    del model
    import gc

    gc.collect()

    if config.embedding_device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _split_embeddings(all_embeddings, len(poly_markets), len(kalshi_markets))


def _split_embeddings(all_embeddings, n_poly: int, n_kalshi: int) -> dict:
    """Split a flat embedding array back into title/full components."""
    idx = 0
    poly_title = all_embeddings[idx : idx + n_poly]; idx += n_poly
    kalshi_title = all_embeddings[idx : idx + n_kalshi]; idx += n_kalshi
    poly_full = all_embeddings[idx : idx + n_poly]; idx += n_poly
    kalshi_full = all_embeddings[idx : idx + n_kalshi]; idx += n_kalshi

    return {
        "poly_title": poly_title,
        "kalshi_title": kalshi_title,
        "poly_full": poly_full,
        "kalshi_full": kalshi_full,
    }


# --- Candidate discovery ---

def find_candidates(
    poly_markets: list[PolymarketMarket],
    kalshi_markets: list[KalshiMarket],
    embeddings: dict,
    config: Config,
) -> list[MatchCandidate]:
    """
    Multi-pass cosine similarity search across all Polymarket × Kalshi pairs.

    A candidate is included if ANY of these conditions are met:
      1. Title similarity >= title_only_threshold
      2. Full-text similarity >= full_text_threshold
      3. Title similarity >= title_with_date_threshold AND dates within max_date_difference_days
    """
    logger.info(
        "Computing similarity matrices (%d x %d)...",
        len(poly_markets),
        len(kalshi_markets),
    )

    def normalize(emb):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        # Avoid division by zero (shouldn't happen with real embeddings, but be safe)
        norms = np.where(norms == 0, 1.0, norms)
        return emb / norms

    poly_title_norm = normalize(embeddings["poly_title"])
    kalshi_title_norm = normalize(embeddings["kalshi_title"])
    poly_full_norm = normalize(embeddings["poly_full"])
    kalshi_full_norm = normalize(embeddings["kalshi_full"])

    title_sim = np.dot(poly_title_norm, kalshi_title_norm.T)
    full_sim = np.dot(poly_full_norm, kalshi_full_norm.T)

    candidates: list[MatchCandidate] = []
    seen_pairs: set[tuple[int, int]] = set()

    # Vectorized date-difference matrix
    date_diffs = None
    if config.max_date_difference_days is not None:
        logger.info(
            "Computing date proximity matrix (max %d days)...",
            config.max_date_difference_days,
        )
        poly_dates = np.array(
            [m.end_date.timestamp() if m.end_date else np.nan for m in poly_markets]
        )
        kalshi_dates = np.array(
            [m.end_date.timestamp() if m.end_date else np.nan for m in kalshi_markets]
        )
        date_diffs = (
            np.abs(poly_dates[:, np.newaxis] - kalshi_dates[np.newaxis, :]) / 86400
        )
        date_diffs = np.nan_to_num(date_diffs, nan=float("inf"))

    if config.multi_pass_matching:
        logger.info("Using multi-pass matching:")
        logger.info("  Pass 1: Title similarity >= %s", config.title_only_threshold)
        logger.info("  Pass 2: Full text similarity >= %s", config.full_text_threshold)
        logger.info(
            "  Pass 3: Title similarity >= %s AND dates within %d days",
            config.title_with_date_threshold,
            config.max_date_difference_days,
        )

        # Pass 1: Title-only (high threshold)
        pass1_pairs = np.argwhere(title_sim >= config.title_only_threshold)
        logger.info("  Pass 1 found: %d pairs", len(pass1_pairs))
        for poly_idx, kalshi_idx in pass1_pairs:
            key = (poly_idx, kalshi_idx)
            if key not in seen_pairs:
                seen_pairs.add(key)
                candidates.append(
                    MatchCandidate(
                        poly_market=poly_markets[poly_idx],
                        kalshi_market=kalshi_markets[kalshi_idx],
                        cosine_similarity=float(title_sim[poly_idx, kalshi_idx]),
                    )
                )

        # Pass 2: Full text (medium threshold)
        pass2_pairs = np.argwhere(full_sim >= config.full_text_threshold)
        pass2_new = 0
        for poly_idx, kalshi_idx in pass2_pairs:
            key = (poly_idx, kalshi_idx)
            if key not in seen_pairs:
                seen_pairs.add(key)
                pass2_new += 1
                candidates.append(
                    MatchCandidate(
                        poly_market=poly_markets[poly_idx],
                        kalshi_market=kalshi_markets[kalshi_idx],
                        cosine_similarity=float(full_sim[poly_idx, kalshi_idx]),
                    )
                )
        logger.info("  Pass 2 found: %d new pairs", pass2_new)

        # Pass 3: Title + date proximity
        if date_diffs is not None:
            mask = (title_sim >= config.title_with_date_threshold) & (
                date_diffs <= config.max_date_difference_days
            )
            pass3_pairs = np.argwhere(mask)
            pass3_new = 0
            for poly_idx, kalshi_idx in pass3_pairs:
                key = (poly_idx, kalshi_idx)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pass3_new += 1
                    candidates.append(
                        MatchCandidate(
                            poly_market=poly_markets[poly_idx],
                            kalshi_market=kalshi_markets[kalshi_idx],
                            cosine_similarity=float(title_sim[poly_idx, kalshi_idx]),
                        )
                    )
            logger.info("  Pass 3 found: %d new pairs", pass3_new)

    else:
        # Legacy single-threshold mode
        logger.info("Using single threshold: %s", config.cosine_similarity_threshold)
        pairs = np.argwhere(full_sim >= config.cosine_similarity_threshold)

        if date_diffs is not None:
            pairs = [
                (pi, ki)
                for pi, ki in pairs
                if date_diffs[pi, ki] <= config.max_date_difference_days
            ]
            logger.info("After date filter: %d pairs", len(pairs))

        for poly_idx, kalshi_idx in pairs:
            candidates.append(
                MatchCandidate(
                    poly_market=poly_markets[poly_idx],
                    kalshi_market=kalshi_markets[kalshi_idx],
                    cosine_similarity=float(full_sim[poly_idx, kalshi_idx]),
                )
            )

    logger.info("Total candidates: %d", len(candidates))
    candidates.sort(key=lambda c: c.cosine_similarity, reverse=True)
    return candidates
