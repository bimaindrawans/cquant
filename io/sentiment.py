# io/sentiment.py

import os
import time
import requests
import pandas as pd

# Fear & Greed Index (no API key required)
FNG_URL = "https://api.alternative.me/fng/?limit=1"

# CryptoPanic (requires free API key; optional)
CP_URL_TEMPLATE = "https://cryptopanic.com/api/v1/posts/?auth_token={token}&kind=news"

def fear_greed_score() -> float:
    """
    Fetch the Fear & Greed Index.
    Returns a float in [0,1], where 0 = extreme fear, 1 = extreme greed.
    """
    try:
        resp = requests.get(FNG_URL, timeout=5)
        data = resp.json().get("data", [])
        if not data:
            return 0.5
        # value is 0–100
        return float(data[0].get("value", 50)) / 100.0
    except Exception:
        return 0.5  # fallback neutral

def crypto_panic_score(token: Optional[str] = None, limit: int = 20) -> float:
    """
    Fetch latest CryptoPanic headlines and compute a sentiment score.
    token: your CryptoPanic API key (set via env var CRYPTOPANIC_TOKEN or pass in).
    limit: max number of posts to fetch.
    
    Returns average sentiment in [-1, 1]:
      +1 for positive, –1 for negative, 0 for neutral.
    """
    api_token = token or os.getenv("CRYPTOPANIC_TOKEN", "")
    url = CP_URL_TEMPLATE.format(token=api_token)
    try:
        resp = requests.get(f"{url}&public=true&kind=news&limit={limit}", timeout=5)
        results = resp.json().get("results", [])
        if not results:
            return 0.0
        score = 0
        count = 0
        for post in results:
            s = post.get("sentiment", {}).get("type", "neutral")
            if s == "positive":
                score += 1
            elif s == "negative":
                score -= 1
            count += 1
        return score / max(1, count)
    except Exception:
        return 0.0  # fallback neutral

def aggregate_sentiment(limit: int = 20) -> pd.DataFrame:
    """
    Combine multiple sentiment sources into a single DataFrame.
    Useful for backtesting or feature-engineering pipelines.
    
    Output columns:
      - timestamp (pd.Timestamp)
      - fear_greed (float)
      - crypto_panic (float)
    """
    now = pd.Timestamp.utcnow().floor("T")
    fg = fear_greed_score()
    cp = crypto_panic_score(limit=limit)
    df = pd.DataFrame({
        "timestamp": [now],
        "fear_greed": [fg],
        "crypto_panic": [cp]
    })
    return df.set_index("timestamp")
