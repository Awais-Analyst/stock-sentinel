"""
data_pipeline.py — Fetch, clean, merge, and cache stock + news + social data.

Robustness features (2026):
  • yfinance: auto_adjust=True, exponential-backoff retry (3 attempts)
  • News: NewsAPI primary → NewsData.io fallback → cached sentiment fallback
  • X scraping: minimal (n≤15), random UA, sleep(5-10); graceful failure
  • CSV caching: load cached first; re-fetch only when missing or stale
"""

import os
import time
import random
import logging
import requests
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

from utils import DATA_DIR, NEWS_API_KEY, NEWSDATA_API_KEY, NEWS_SOURCE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CACHING
# ─────────────────────────────────────────────

def _cache_path(symbol: str, tag: str = "ohlcv") -> str:
    return os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}_{tag}.csv")


def cache_data(df: pd.DataFrame, symbol: str, tag: str = "ohlcv") -> None:
    path = _cache_path(symbol, tag)
    df.to_csv(path)
    log.info(f"Cached {tag} data → {path}")


def load_cached(symbol: str, tag: str = "ohlcv") -> pd.DataFrame | None:
    path = _cache_path(symbol, tag)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log.info(f"Loaded cached {tag} data for {symbol} ({len(df)} rows)")
        return df
    return None


# ─────────────────────────────────────────────
# STOCK DATA (yfinance)
# ─────────────────────────────────────────────

def get_stock_data(symbol: str, start: str, end: str,
                   force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with exponential-backoff retry.
    Cached data is ALWAYS filtered by the requested date range so that
    changing the date in the sidebar actually updates the chart.
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    if not force_refresh:
        cached = load_cached(symbol, "ohlcv")
        if cached is not None:
            filtered = cached[
                (cached.index >= start_ts) & (cached.index <= end_ts)
            ]
            if not filtered.empty:
                log.info(f"Cache hit for {symbol} [{start} to {end}]: {len(filtered)} rows")
                return filtered
            log.info(f"Cache miss for {symbol} in [{start} to {end}] — re-fetching from yfinance")

    for attempt in range(1, 4):
        try:
            df = yf.download(
                symbol, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if df.empty:
                raise ValueError(f"Empty DataFrame returned for {symbol}")

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df.dropna(subset=["Close"], inplace=True)
            cache_data(df, symbol, "ohlcv")
            log.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as exc:
            wait = 2 ** attempt
            log.warning(f"yfinance attempt {attempt}/3 failed for {symbol}: {exc} — retrying in {wait}s")
            time.sleep(wait)

    log.error(f"All yfinance attempts failed for {symbol}. Falling back to cache.")
    cached = load_cached(symbol, "ohlcv")
    if cached is not None:
        return cached[(cached.index >= start_ts) & (cached.index <= end_ts)]
    return pd.DataFrame()


# ── Ticker → company name mapping for better news search ──────────────────
# AAPL → "Apple" so search "Apple stock" not "AAPL stock" (news never writes tickers)
_TICKER_TO_NAME: dict = {
    "AAPL":    "Apple",
    "MSFT":    "Microsoft",
    "TSLA":    "Tesla",
    "GOOGL":   "Google",
    "GOOG":    "Google",
    "AMZN":    "Amazon",
    "META":    "Meta Facebook",
    "NVDA":    "Nvidia",
    "NFLX":    "Netflix",
    "BRK.B":   "Berkshire Hathaway",
    "JPM":     "JPMorgan",
    "V":       "Visa",
    "MA":      "Mastercard",
    "JNJ":     "Johnson Johnson",
    "WMT":     "Walmart",
    "DIS":     "Disney",
    "BABA":    "Alibaba",
    "TSM":     "TSMC Taiwan Semiconductor",
    "SAMSUNG": "Samsung",
    # Pakistani stocks
    "NBP.KA":   "National Bank Pakistan NBP",
    "MCB.KA":   "MCB Bank Pakistan",
    "HBL.KA":   "Habib Bank HBL Pakistan",
    "UBL.KA":   "United Bank UBL Pakistan",
    "OGDC.KA":  "Oil Gas Development Pakistan OGDC",
    "PSO.KA":   "Pakistan State Oil PSO",
    "ENGRO.KA": "Engro Corporation Pakistan",
    "LUCK.KA":  "Lucky Cement Pakistan",
    "HUBC.KA":  "Hub Power Pakistan",
    "PPL.KA":   "Pakistan Petroleum PPL",
}

def _build_search_query(symbol: str) -> str:
    """
    Build a news search query that uses the company name, not the ticker.
    'AAPL' → 'Apple stock'  (not 'AAPL stock' which returns 0 results)
    """
    name = _TICKER_TO_NAME.get(symbol.upper())
    if name:
        return f'"{name}" stock OR "{name}" shares'
    # Unknown ticker: strip exchange suffix and search both name and ticker
    base = symbol.split(".")[0]
    return f'"{base}" stock OR "{base}" shares'


# ─────────────────────────────────────────────
# NEWS DATA
# ─────────────────────────────────────────────

def _fetch_newsapi(symbol: str, api_key: str, days_back: int = 28) -> list[dict]:
    """NewsAPI.org headlines for the past `days_back` days.
    Searches by company name, not ticker symbol.
    NOTE: Developer plan strictly limits to 1 month. Using 28 days avoids
    errors in shorter months like February."""
    from_date  = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    search_q   = _build_search_query(symbol)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        search_q,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 100,
        "apiKey":   api_key,
    }
    log.info(f"NewsAPI query for {symbol}: {search_q}")
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")
    articles = data.get("articles", [])
    log.info(f"NewsAPI returned {len(articles)} raw articles for {symbol}")
    return [
        {
            "date":   (a.get("publishedAt") or "")[:10],
            "text":   f"{a.get('title', '')} {a.get('description') or ''}".strip(),
            "source": "newsapi",
        }
        for a in articles if a.get("title")
    ]



def _fetch_newsdata(symbol: str, api_key: str) -> list[dict]:
    """
    NewsData.io headlines — fallback source.
    Searches without category restriction to maximise results.
    Tries the latest endpoint; falls back to archive endpoint if needed.
    """
    # Strip .KA / exchange suffix for cleaner search (e.g. "NBP.KA" → "NBP")
    clean_sym = symbol.split(".")[0]
    search_q  = f"{clean_sym} stock OR {clean_sym} shares"

    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey":   api_key,
        "q":        search_q,
        "language": "en",
        # No 'category' filter — it was removing valid financial articles
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        # 402 = payment required (free plan limit hit)
        raise RuntimeError(f"NewsData.io HTTP {e.response.status_code}: {e}")
    except Exception as e:
        raise RuntimeError(f"NewsData.io request failed: {e}")

    if data.get("status") != "success":
        raise RuntimeError(f"NewsData.io API error: {data.get('message', data.get('results', 'unknown'))}")

    results = data.get("results") or []
    articles = []
    for r in results:
        title = r.get("title") or ""
        desc  = r.get("description") or r.get("content") or ""
        date  = (r.get("pubDate") or r.get("published_at") or "")[:10]
        if title and date:
            articles.append({
                "date":   date,
                "text":   f"{title} {desc}".strip(),
                "source": "newsdata",
            })
    log.info(f"NewsData.io returned {len(articles)} articles for {symbol}")
    return articles


def get_news(symbol: str,
             newsapi_key: str = NEWS_API_KEY,
             newsdata_key: str = NEWSDATA_API_KEY,
             source: str = NEWS_SOURCE) -> list[dict]:
    """
    Fetch news with dual-source resilience.
    source='auto'  → tries NewsAPI; if it returns 0 articles OR fails,
                     also tries NewsData.io; combines results.
    source='newsapi'  → only NewsAPI
    source='newsdata' → only NewsData.io
    """
    all_articles: list[dict] = []

    # ── NewsAPI ──────────────────────────────────────────────────────────
    if source in ("newsapi", "auto"):
        if newsapi_key and newsapi_key not in ("", "YOUR_NEWSAPI_KEY"):
            try:
                articles = _fetch_newsapi(symbol, newsapi_key)
                log.info(f"NewsAPI returned {len(articles)} articles for '{symbol}'")
                all_articles.extend(articles)
            except Exception as e:
                log.warning(f"NewsAPI failed for '{symbol}': {e}")

    # ── NewsData.io ───────────────────────────────────────────────────────
    # Try if: (a) explicitly selected, OR (b) auto mode and NewsAPI gave 0
    if source == "newsdata" or (source == "auto" and len(all_articles) == 0):
        if newsdata_key and newsdata_key not in ("", "YOUR_NEWSDATA_KEY"):
            try:
                articles = _fetch_newsdata(symbol, newsdata_key)
                log.info(f"NewsData.io returned {len(articles)} articles for '{symbol}'")
                all_articles.extend(articles)
            except Exception as e:
                log.warning(f"NewsData.io failed for '{symbol}': {e}")

    if not all_articles:
        log.warning(f"All news sources returned 0 articles for '{symbol}'.")
    return all_articles



# ─────────────────────────────────────────────
# X / SOCIAL SCRAPING (minimal + graceful)
# ─────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]


def get_x_posts(symbol: str, n: int = 15) -> list[dict]:
    """
    Scrape a minimal sample of public X search results.
    n is capped at 15. On any failure returns [] gracefully.
    NOTE: X increasingly requires login for search; this is best-effort.
    """
    n = min(n, 15)
    try:
        query = f"{symbol} stock"
        url = f"https://x.com/search?q={requests.utils.quote(query)}&src=typed_query&f=live"
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        sleep_sec = random.uniform(5, 10)
        log.info(f"X scrape: sleeping {sleep_sec:.1f}s before request")
        time.sleep(sleep_sec)

        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code in (429, 403, 401):
            raise RuntimeError(f"X returned HTTP {resp.status_code} — bot detection / rate limit")

        soup = BeautifulSoup(resp.text, "lxml")
        # X is JS-heavy; look for any visible text nodes as a best-effort parse
        posts = []
        for tag in soup.find_all(["article", "div"], class_=lambda c: c and "tweet" in c.lower()):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 20:
                posts.append({"date": datetime.utcnow().strftime("%Y-%m-%d"),
                               "text": text[:280], "source": "x"})
                if len(posts) >= n:
                    break

        if not posts:
            log.warning("X scrape returned 0 posts (JS-heavy page or blocked). Using news only.")
        else:
            log.info(f"X scrape collected {len(posts)} posts for {symbol}")
        return posts

    except Exception as exc:
        log.warning(f"X scrape failed — using news only. Reason: {exc}")
        return []


# ─────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns, moving averages, and basic tech features."""
    df = df.copy()
    df["log_return"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["ma_10"]        = df["Close"].rolling(10).mean()
    df["ma_20"]        = df["Close"].rolling(20).mean()
    df["ma_50"]        = df["Close"].rolling(50).mean()
    df["volatility_5"] = df["log_return"].rolling(5).std()
    df["daily_pct"]    = df["Close"].pct_change()
    return df


def _aggregate_texts_by_date(items: list[dict]) -> dict[str, list[str]]:
    """Group article/post texts by date string."""
    result: dict[str, list[str]] = {}
    for item in items:
        date = (item.get("date") or "")[:10]
        if date:
            result.setdefault(date, []).append(item.get("text", ""))
    return result


# ─────────────────────────────────────────────
# MAIN MERGE FUNCTION
# ─────────────────────────────────────────────

def build_dataset(symbol: str, start: str, end: str,
                  newsapi_key: str = NEWS_API_KEY,
                  newsdata_key: str = NEWSDATA_API_KEY,
                  force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Build the full merged dataset:
      • OHLCV + technical features (from yfinance)
      • news texts per date (for sentiment)
      • X post texts per date (for sentiment)

    Returns (df, texts_by_date) where texts_by_date maps date_str → list[str].
    """
    # 1. Stock prices
    df = get_stock_data(symbol, start, end, force_refresh=force_refresh)
    if df.empty:
        log.error(f"No price data for {symbol}. Cannot build dataset.")
        return df, {}

    df = _add_features(df)

    # 2. News
    news = get_news(symbol, newsapi_key=newsapi_key, newsdata_key=newsdata_key)

    # 3. Social (best-effort)
    x_posts = get_x_posts(symbol, n=15)

    # 4. Merge text sources by date
    all_texts = news + x_posts
    texts_by_date = _aggregate_texts_by_date(all_texts)

    # 5. Add sentiment placeholder column (filled by sentiment.py)
    df["sentiment"] = 0.0
    for date_str, texts in texts_by_date.items():
        try:
            idx = pd.Timestamp(date_str)
            if idx in df.index:
                df.at[idx, "sentiment"] = np.nan  # sentinel for "has text, score later"
        except Exception:
            pass

    df.ffill(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    cache_data(df, symbol, "dataset")
    log.info(f"Dataset built: {len(df)} rows, {len(texts_by_date)} dated text buckets")
    return df, texts_by_date


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("─── data_pipeline.py self-test ───")
    sym = "AAPL"
    df, texts = build_dataset(sym, "2024-01-01", "2024-06-30",
                               newsapi_key="YOUR_NEWSAPI_KEY",  # will skip to fallback
                               newsdata_key="YOUR_NEWSDATA_KEY")
    if not df.empty:
        print(f"Shape : {df.shape}")
        print(df[["Close", "log_return", "ma_20"]].tail(5))
        print(f"Text buckets with data: {len(texts)}")
        print("✅ data_pipeline.py self-test passed.")
    else:
        print("⚠️  No data returned (check API keys or network).")
