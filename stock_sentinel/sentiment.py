"""
sentiment.py — Text preprocessing, VADER scoring, word cloud, and correlation.
"""

import re
import string
import logging
from collections import Counter

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

# Download NLTK stopwords lazily
try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))

# Finance-domain stopwords to add
_FINANCE_STOPS = {
    "stock", "share", "market", "company", "firm",
    "year", "quarter", "said", "says", "would", "could",
}
_STOPWORDS |= _FINANCE_STOPS

_ANALYZER = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip URLs, mentions, hashtags, punctuation, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                       # @mentions
    text = re.sub(r"#\w+", " ", text)                       # #hashtags
    text = re.sub(r"[^a-z\s]", " ", text)                   # non-alpha
    text = re.sub(r"\s+", " ", text).strip()                 # extra spaces
    return text


def _tokenize(text: str) -> list[str]:
    return [w for w in clean_text(text).split() if w not in _STOPWORDS and len(w) > 2]


# ─────────────────────────────────────────────
# VADER SCORING
# ─────────────────────────────────────────────

def score_sentiment(texts: list[str]) -> float:
    """
    Compute mean VADER compound score for a list of texts.
    Returns 0.0 for empty lists.
    """
    if not texts:
        return 0.0
    scores = [_ANALYZER.polarity_scores(t)["compound"] for t in texts if t.strip()]
    return float(np.mean(scores)) if scores else 0.0


def score_sentiment_detailed(texts: list[str]) -> dict:
    """Full VADER breakdown: pos, neg, neu, compound (mean over all texts)."""
    if not texts:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
    keys = ["compound", "pos", "neg", "neu"]
    agg = {k: [] for k in keys}
    for t in texts:
        if t.strip():
            scores = _ANALYZER.polarity_scores(t)
            for k in keys:
                agg[k].append(scores[k])
    return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}


def add_sentiment_to_df(df: pd.DataFrame,
                         texts_by_date: dict[str, list[str]]) -> pd.DataFrame:
    """
    Add a 'sentiment' column to price DataFrame.
    texts_by_date: {date_str → [raw text, ...]}
    Dates with no text get the previous day's sentiment (ffill).
    """
    df = df.copy()
    df["sentiment"] = 0.0

    for date_str, texts in texts_by_date.items():
        cleaned = [clean_text(t) for t in texts]
        score   = score_sentiment(cleaned)
        try:
            idx = pd.Timestamp(date_str)
            if idx in df.index:
                df.at[idx, "sentiment"] = score
        except Exception:
            pass

    df["sentiment"] = df["sentiment"].replace(0.0, np.nan)
    df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
    log.info(f"Sentiment column added. Range: [{df['sentiment'].min():.3f}, {df['sentiment'].max():.3f}]")
    return df


# ─────────────────────────────────────────────
# WORD CLOUD & KEYWORDS
# ─────────────────────────────────────────────

def get_top_keywords(texts: list[str], n: int = 20) -> list[tuple[str, int]]:
    """Top-n keywords (after stopword removal) across all texts."""
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(_tokenize(t))
    return Counter(all_tokens).most_common(n)


def generate_wordcloud(texts: list[str], width: int = 800, height: int = 400):
    """
    Generate a word cloud PIL Image from texts.
    Returns None if wordcloud lib is not installed or texts are empty.
    """
    try:
        from wordcloud import WordCloud
        combined = " ".join([clean_text(t) for t in texts])
        if not combined.strip():
            return None
        wc = WordCloud(
            width=width, height=height,
            background_color="#0e1117",
            colormap="YlGn",
            stopwords=_STOPWORDS,
            max_words=150,
            prefer_horizontal=0.9,
        ).generate(combined)
        return wc.to_image()
    except ImportError:
        log.warning("wordcloud not installed — skipping word cloud generation.")
        return None
    except Exception as exc:
        log.warning(f"Word cloud generation failed: {exc}")
        return None


# ─────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────

def sentiment_price_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation between sentiment and price-related columns.
    Looks for: sentiment, Close, log_return, daily_pct columns.
    """
    cols = [c for c in ["sentiment", "Close", "log_return", "daily_pct", "Volume"]
            if c in df.columns]
    corr = df[cols].corr()
    log.info(f"Sentiment vs. Close correlation: {corr.loc['sentiment', 'Close']:.4f}")
    return corr


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("─── sentiment.py self-test ───")

    samples = [
        "Apple stock surges 5% on record earnings, investors thrilled",
        "Market crash fears as Fed raises rates again, stocks plunge",
        "AAPL neutral trading session with mixed signals from analysts",
        "Breaking: company faces massive fraud allegations — stock tanks",
        "Strong buy signal on Apple, revenue beats expectations",
    ]

    cleaned = [clean_text(t) for t in samples]
    print("Cleaned texts:")
    for c in cleaned:
        print(f"  {c}")

    score = score_sentiment(samples)
    print(f"\nMean compound score: {score:.4f}")

    detailed = score_sentiment_detailed(samples)
    print(f"Detailed: {detailed}")

    keywords = get_top_keywords(samples, n=10)
    print(f"\nTop keywords: {keywords}")

    wc_img = generate_wordcloud(samples)
    print(f"Word cloud generated: {wc_img is not None}")

    # Correlation test with dummy DF
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df_test = pd.DataFrame({
        "Close": [150, 148, 155, 153, 157],
        "log_return": [0, -0.013, 0.046, -0.013, 0.026],
        "daily_pct":  [0, -0.013, 0.047, -0.013, 0.026],
        "sentiment":  [0.3, -0.5, 0.6, -0.2, 0.4],
    }, index=idx)
    corr = sentiment_price_correlation(df_test)
    print(f"\nCorrelation matrix:\n{corr.round(3)}")

    print("\n✅ sentiment.py self-test passed.")
