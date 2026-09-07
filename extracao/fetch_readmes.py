"""Fetch README content from GitHub API and enrich repos.parquet.

Usage:
    python extracao/fetch_readmes.py --input data/raw/repos.parquet
    python extracao/fetch_readmes.py --input data/raw/repos.parquet --token ghp_...
    python extracao/fetch_readmes.py --input data/raw/repos.parquet --sample 500

Rate limits:
    Without token: 60 req/hour  (~1 min for 60 repos)
    With token:    5000 req/hour (~6 min for 500 repos)

Get a free token at: https://github.com/settings/tokens
(No scopes needed — public repo read is available without any scope)
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.seeds import GLOBAL_SEED  # noqa: F401

GITHUB_API = "https://api.github.com"
README_CACHE = Path("data/raw/readme_cache.parquet")
MAX_README_CHARS = 2000


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_readme(session: requests.Session, repo_id: str) -> str:
    """Return first MAX_README_CHARS chars of README, or '' if unavailable."""
    url = f"{GITHUB_API}/repos/{repo_id}/readme"
    resp = session.get(url, timeout=15)
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
    return content[:MAX_README_CHARS]


def fetch_readmes(
    input_path: Path,
    token: str | None,
    sample: int | None,
    output_path: Path,
) -> None:
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} repos from {input_path}")

    # Work only on the sample if specified (same sample as gold_standard.py uses)
    if sample:
        target = df.sample(n=min(sample, len(df)), random_state=GLOBAL_SEED).copy()
        print(f"Fetching READMEs for sample of {len(target):,} repos")
    else:
        target = df.copy()
        print(f"Fetching READMEs for all {len(target):,} repos")

    # Resume from cache
    cached_ids: set[str] = set()
    if README_CACHE.exists():
        cached = pd.read_parquet(README_CACHE)
        cached_ids = set(cached["repo_id"])
        print(f"Cache hit: {len(cached_ids):,} repos already fetched")

    session = requests.Session()
    session.headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        print("WARNING: No token provided. Rate limit: 60 req/hour. Use --token for 5000/hour.")

    results: list[dict] = []
    errors = 0

    to_fetch = target[~target["repo_id"].isin(cached_ids)]
    print(f"Repos to fetch: {len(to_fetch):,}")

    for i, (_, row) in enumerate(to_fetch.iterrows()):
        repo_id = row["repo_id"]
        try:
            readme = _fetch_readme(session, repo_id)
            results.append({"repo_id": repo_id, "readme_text": readme})
        except Exception as e:
            results.append({"repo_id": repo_id, "readme_text": ""})
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1:,}/{len(to_fetch):,} fetched (errors: {errors})", end="\r")
            # Save progress every 50
            if (i + 1) % 50 == 0:
                _save_cache(results, cached_ids, README_CACHE)

        # Respect rate limits: ~0.7s between requests stays under 5000/hour
        time.sleep(0.8 if not token else 0.72)

    _save_cache(results, cached_ids, README_CACHE)

    # Merge readme_text back into main parquet
    cache_df = pd.read_parquet(README_CACHE)
    readme_map = dict(zip(cache_df["repo_id"], cache_df["readme_text"]))
    df["readme_text"] = df["repo_id"].map(readme_map).fillna("")
    df.to_parquet(input_path, index=False, compression="snappy")

    filled = (df["readme_text"] != "").sum()
    print(f"\n\nDone!")
    print(f"  READMEs fetched: {len(results):,} (errors: {errors})")
    print(f"  repos.parquet updated: {filled:,}/{len(df):,} repos have readme_text")
    print(f"\nNext step: python preparacao/build_features.py")


def _save_cache(new_results: list[dict], existing_ids: set[str], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_results)
    if cache_path.exists() and existing_ids:
        existing = pd.read_parquet(cache_path)
        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates("repo_id")
        combined.to_parquet(cache_path, index=False)
    else:
        new_df.to_parquet(cache_path, index=False)


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="Fetch README content from GitHub API")
    parser.add_argument("--input", type=Path, default=Path("data/raw/repos.parquet"))
    parser.add_argument("--token", type=str, default=None, help="GitHub personal access token (fallback: GITHUB_TOKEN env var)")
    parser.add_argument("--sample", type=int, default=None, help="Fetch only N repos (default: all)")
    parser.add_argument("--all", dest="all_repos", action="store_true", help="Fetch READMEs for all repos (default behavior)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")

    fetch_readmes(
        input_path=args.input,
        token=token,
        sample=args.sample,
        output_path=README_CACHE,
    )


if __name__ == "__main__":
    main()
