"""
update_tracker.py

Searches recent news for stories about Trump praising a stock/company,
asks Claude to pull out a structured entry if one genuinely exists,
and appends new entries to data/entries.json (skipping duplicates).

Run manually:   python update_tracker.py
Run on a schedule: see .github/workflows/update.yml
"""

import json
import os
import re
import time
import urllib.parse
from pathlib import Path

import requests
from anthropic import Anthropic

DATA_FILE = Path(__file__).parent / "data" / "entries.json"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Search terms aimed at the specific behavior we're tracking.
SEARCH_QUERIES = [
    '"Trump" "great company" stock',
    '"Trump" "great stock"',
    '"Trump" "time to buy" stock',
    '"Trump" Truth Social praised stock',
    '"Trump" praised stock purchase disclosure',
]

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def load_existing():
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return []


def save_entries(entries):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(entries, indent=2))


def search_gdelt(query, max_records=15):
    """Free, keyless news search across the last 24 hours."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "timespan": "1d",
        "format": "json",
    }
    try:
        r = requests.get(GDELT_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        print(f"  search failed for {query!r}: {e}")
        return []


def fetch_article_text(url, max_chars=6000):
    """Best-effort plain-text grab. Skipped if it fails — we just rely
    on the title/snippet GDELT already gave us in that case."""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        text = re.sub("<[^<]+?>", " ", r.text)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception:
        return ""


EXTRACTION_PROMPT = """You are screening a news article for one specific pattern:
President Trump publicly praising a SPECIFIC stock, ticker, or company by name,
in a quote attributed to him (Truth Social post, speech, or TV/interview remark).

General economic boosterism ("the economy is great") does NOT count.
Praise of a vague policy area does NOT count.
It must be a specific, named, publicly traded company or its product.

Article title: {title}
Article URL: {url}
Article text (may be partial or a snippet): {text}

If this article describes a genuine instance of that pattern, respond with ONLY
a JSON object, no other text, in this exact shape:
{{
  "match": true,
  "date": "YYYY-MM-DD",
  "company": "Company name",
  "ticker": "TICKER",
  "platform": "Truth Social" | "Speech" | "TV interview",
  "quote": "the exact or closely paraphrased quote",
  "context": "1-2 sentence factual context, including any nearby stock trade if mentioned",
  "flag": "short phrase noting trade timing if relevant, else empty string"
}}

If it does NOT match, respond with ONLY:
{{"match": false}}
"""


def classify(article):
    title = article.get("title", "")
    url = article.get("url", "")
    text = fetch_article_text(url) or article.get("seendate", "")

    prompt = EXTRACTION_PROMPT.format(title=title, url=url, text=text)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not parsed.get("match"):
        return None
    parsed["source"] = article.get("domain", "")
    parsed["url"] = url
    parsed["flagType"] = "rust" if parsed.get("flag") else "teal"
    return parsed


def main():
    existing = load_existing()
    existing_urls = {e["url"] for e in existing}
    new_entries = []

    seen_urls = set()
    for q in SEARCH_QUERIES:
        for article in search_gdelt(q):
            url = article.get("url")
            if not url or url in existing_urls or url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"Checking: {article.get('title')}")
            result = classify(article)
            time.sleep(1)  # be polite to both APIs

            if result:
                print(f"  -> match: {result['company']}")
                new_entries.append(result)
            else:
                print("  -> no match")

    if new_entries:
        combined = existing + new_entries
        combined.sort(key=lambda e: e["date"], reverse=True)
        save_entries(combined)
        print(f"\nAdded {len(new_entries)} new entr{'y' if len(new_entries)==1 else 'ies'}.")
    else:
        print("\nNo new entries found this run.")


if __name__ == "__main__":
    main()
