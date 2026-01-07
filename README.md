# Cross-Venue Arbitrage Market Matcher

Identifies matching prediction markets between Polymarket and Kalshi for potential arbitrage opportunities.

## Pipeline Overview

```
1. Fetch active markets from both platforms (public APIs)
2. Apply filters (date, volume, liquidity, categories)
3. Generate embeddings (local, sentence-transformers)
4. Find candidate pairs via cosine similarity
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

**Windows:**

- Download from https://ollama.ai/download
- Run the installer

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
ollama pull llama3.1:8b
```

This downloads the Llama 3.1 8B model (~4.7GB). First run will take a few minutes.

### 4. Start Ollama (if not running)

```bash
ollama serve
```

Leave this running in a terminal, or it may run as a service automatically.

## Usage

### Basic Run

```bash
python arb_matcher.py
```

### Configuration

Edit the `CONFIG` section at the top of `arb_matcher.py`:

```python
@dataclass
class Config:
    # Date filters (markets must resolve within this window)
    min_end_date: datetime = None  # Set to None for "now + days_until_min_end"
    max_end_date: datetime = None  # Set to None for "now + days_until_max_end"
    days_until_max_end: int = 90   # Default: 90 days
    days_until_min_end: int = 1    # Default: 1 day (exclude markets closing today)

    # Volume filters
    polymarket_min_volume: int = 5000
    kalshi_min_volume: int = 5000

    # Liquidity filters
    polymarket_min_liquidity: int = 5000
    kalshi_min_liquidity: int = 5000

    # Category filters (empty list = include all)
    include_categories: list = None  # e.g., ["politics", "crypto"]
    exclude_categories: list = None  # e.g., ["weather"]

    # Matching thresholds
    cosine_similarity_threshold: float = 0.70  # Lower = more candidates

    # LLM settings
    llm_model: str = "llama3.1:8b"
```

## Output Files

| File                            | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| `arb_candidates_YYYY-MM-DD.csv` | All pairs above cosine similarity threshold (pre-LLM) |
| `arb_matches_YYYY-MM-DD.csv`    | LLM-verified matches from this run only               |
| `arb_matches_all.csv`           | Cumulative all-time verified matches                  |
| `state.db`                      | SQLite database storing all match history             |

## Output Fields

| Field               | Description                         |
| ------------------- | ----------------------------------- |
| `poly_condition_id` | Polymarket market ID                |
| `poly_title`        | Polymarket market question          |
| `poly_description`  | Polymarket resolution criteria      |
| `poly_url`          | Link to Polymarket market           |
| `poly_yes_price`    | Current YES price (0-1)             |
| `poly_end_date`     | Resolution date                     |
| `kalshi_ticker`     | Kalshi market ticker                |
| `kalshi_title`      | Kalshi market title                 |
| `kalshi_rules`      | Kalshi resolution rules             |
| `kalshi_url`        | Link to Kalshi market               |
| `kalshi_yes_price`  | Current YES price (0-1)             |
| `kalshi_end_date`   | Resolution date                     |
| `cosine_similarity` | Embedding similarity score          |
| `llm_confidence`    | LLM's confidence in the match (0-1) |
| `llm_reasoning`     | LLM's explanation                   |
| `price_diff_pct`    | Price difference percentage         |
| `match_date`        | When the match was identified       |

## Customizing Embedding Text

To change what information is used for semantic matching, edit these functions in `arb_matcher.py`:

```python
def build_embedding_text_polymarket(market: PolymarketMarket) -> str:
    """
    MODIFY THIS FUNCTION to change what information is used for matching.
    Currently includes: title, description, end_date
    """
    ...

def build_embedding_text_kalshi(market: KalshiMarket) -> str:
    """
    MODIFY THIS FUNCTION to change what information is used for matching.
    Currently includes: title, subtitle, rules, end_date
    """
    ...
```

## Resetting State

To re-check all markets (clear "previously seen" state):

```bash
rm state.db
```

Or delete specific entries using any SQLite tool:

```bash
sqlite3 state.db "DELETE FROM verified_matches WHERE poly_condition_id = 'xxx'"
```

## Tuning the Pipeline

### If you're getting too many false positives:

- Increase `cosine_similarity_threshold` (e.g., 0.75 or 0.80)

### If you're missing matches:

- Decrease `cosine_similarity_threshold` (e.g., 0.65)
- Check `arb_candidates_YYYY-MM-DD.csv` to see what pairs the embeddings are finding

### If LLM is too slow:

- Try a smaller model: `ollama pull llama3.2:3b` and set `llm_model: str = "llama3.2:3b"`
- Or use a faster quantization: `ollama pull llama3.1:8b-q4_0`

### If LLM quality is poor:

- Try a larger model: `ollama pull llama3.1:70b` (requires ~40GB RAM)
- Or use an API-based model (modify `verify_with_llm` function)

## Performance Estimates (RTX 3070 + Ryzen 7 7700x)

| Stage                  | ~8000 markets total   | Notes                        |
| ---------------------- | --------------------- | ---------------------------- |
| API fetching           | 30-60 seconds         | Network dependent            |
| Embedding generation   | 10-30 seconds         | Very fast on GPU             |
| Similarity computation | 1-5 seconds           | Pure numpy                   |
| LLM verification       | 1-3 sec per candidate | ~5-15 min for 300 candidates |
| **Total**              | **~10-20 minutes**    | Depends on candidate count   |

## Troubleshooting

### "Connection refused" from Ollama

Make sure Ollama is running:

```bash
ollama serve
```

### "Model not found"

Pull the model first:

```bash
ollama pull llama3.1:8b
```

### API rate limiting

The tool fetches markets in batches. If you hit rate limits, add delays in the fetch functions or reduce fetch frequency.

### Out of memory

- Close other applications
- Use a smaller LLM model
- Process in smaller batches (modify the code to limit candidates per run)
