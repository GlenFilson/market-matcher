# Cross-Venue Arbitrage Market Matcher

Identifies matching prediction markets between Polymarket and Kalshi for potential arbitrage opportunities.

## Pipeline Overview

```
1. Fetch active markets from both platforms (with caching)
2. Apply filters (date, volume, liquidity, categories)
3. Generate embeddings (local, sentence-transformers)
4. Find candidate pairs via multi-pass similarity matching
5. Save candidates CSV (for pipeline tuning)
6. LLM verification of candidates (local, Ollama)
7. Dedupe against previously identified pairs
8. Output CSVs
```

## Setup

### 1. Install Python Dependencies

```bash
cd arb_matcher
pip install -r requirements.txt
```

### 2. Install Ollama (Local LLM)

**Windows:** Download from https://ollama.ai/download

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

### 3. Download the LLM Model

```bash
ollama pull qwen2.5:7b
```

## Usage

### Command Line Flags

| Flag | Description |
|------|-------------|
| (none) | Normal run - uses cache if valid |
| `--resume` | Resume LLM verification from candidates CSV |
| `--fresh` | Force fresh API fetch, ignore cache |

```bash
python arb_matcher.py              # Normal run
python arb_matcher.py --resume     # Resume from where you left off
python arb_matcher.py --fresh      # Force fresh API fetch
```

## Configuration

Edit the `CONFIG` section in `arb_matcher.py`:

```python
# Cache settings
cache_expiry_minutes: int = 60  # How long to use cached market data

# Multi-pass matching thresholds
title_only_threshold: float = 0.88
full_text_threshold: float = 0.85
title_with_date_threshold: float = 0.75
max_date_difference_days: int = 3

# LLM model
llm_model: str = "qwen2.5:7b"
```

## Output Files

| File | Description |
|------|-------------|
| `arb_candidates_*.csv` | Pre-LLM candidate pairs |
| `arb_matches_*.csv` | LLM-verified matches |
| `arb_matches_all.csv` | Cumulative all matches |
| `state.db` | SQLite match history |
| `.cache/` | Cached market data |

## Arbitrage Columns

| Column | Meaning |
|--------|---------|
| `pykn` | Poly YES + Kalshi NO - if < 1.0: buy YES poly, NO kalshi |
| `kypn` | Kalshi YES + Poly NO - if < 1.0: buy YES kalshi, NO poly |

## Resetting

```bash
rm state.db              # Clear match history
rm -rf .cache            # Force fresh fetch
rm arb_candidates_*.csv  # Re-run embeddings
```

## Performance

| Stage | Fresh | Cached |
|-------|-------|--------|
| API fetch | ~6 min | ~2 sec |
| Embeddings | ~40 sec | ~40 sec |
| LLM (~700 candidates) | ~25 min | ~25 min |
| **Total** | **~35 min** | **~30 min** |
