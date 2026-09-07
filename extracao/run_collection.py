"""CLI script to collect repositories from SEART GHS and save to Parquet.

Usage:
    python extracao/run_collection.py --max-repos 15000
    python extracao/run_collection.py --max-repos 50000 --language Python
    python extracao/run_collection.py --max-repos 15000 --resume  # resume interrupted run
    python extracao/run_collection.py --max-repos 15000 --no-verify-ssl  # Windows SSL workaround
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED  # noqa: F401 — side effect: seeds applied
from extracao.seart_ghs import RepoRecord, fetch_repositories

RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT = RAW_DIR / "repos.parquet"
CHECKPOINT_FILE = RAW_DIR / "collection_checkpoint.json"


def record_to_dict(repo: RepoRecord, collected_at: str) -> dict:
    return {
        "repo_id": repo.repo_id,
        "name": repo.name,
        "description": repo.description,
        "language": repo.language,
        "stars": repo.stars,
        "forks": repo.forks,
        "watchers": repo.watchers,
        "open_issues": repo.open_issues,
        "size_kb": repo.size_kb,
        "has_readme": repo.has_readme,
        "has_license": repo.has_license,
        "has_wiki": repo.has_wiki,
        "commits": repo.commits,
        "contributors": repo.contributors,
        "created_at": repo.created_at,
        "updated_at": repo.updated_at,
        "readme_text": repo.readme_text,
        "collected_at": collected_at,
    }


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


def load_checkpoint() -> dict | None:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return None


def compute_dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def run_collection(
    max_repos: int,
    language: str | None,
    min_stars: int,
    output: Path,
    resume: bool,
    verify_ssl: bool = True,
    batch_size: int = 1000,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    existing_records: list[dict] = []
    if resume and output.exists():
        checkpoint = load_checkpoint()
        if checkpoint:
            print(f"Resuming from checkpoint: {checkpoint['collected']} repos already saved")
            existing_df = pd.read_parquet(output)
            existing_records = existing_df.to_dict("records")
            print(f"  Loaded {len(existing_records)} records from {output}")
        else:
            print("No checkpoint found — starting fresh")

    existing_ids = {r["repo_id"] for r in existing_records}
    records = list(existing_records)
    batch: list[dict] = []

    print(f"Target: {max_repos} repositories (min_stars={min_stars}, language={language or 'any'})")
    print(f"Output: {output}")

    if not verify_ssl:
        print("WARNING: SSL verification disabled (--no-verify-ssl). Use only in trusted networks.")

    for repo in fetch_repositories(
        min_stars=min_stars,
        language=language,
        has_readme=True,
        max_repos=max_repos,
        verify_ssl=verify_ssl,
    ):
        if repo.repo_id in existing_ids:
            continue

        batch.append(record_to_dict(repo, collected_at))
        existing_ids.add(repo.repo_id)

        if len(batch) % 100 == 0:
            progress = len(records) + len(batch)
            print(f"  Collected: {progress:,} / {max_repos:,}", end="\r")

        if len(batch) >= batch_size:
            records.extend(batch)
            _flush_to_parquet(records, output)
            save_checkpoint({"collected": len(records), "timestamp": collected_at})
            batch = []

        if len(records) + len(batch) >= max_repos:
            break

    if batch:
        records.extend(batch)

    _flush_to_parquet(records, output)

    df = pd.read_parquet(output)
    dataset_hash = compute_dataset_hash(df)

    print(f"\n\nCollection complete!")
    print(f"  Total repos:   {len(df):,}")
    print(f"  Languages:     {df['language'].nunique()} unique")
    print(f"  Has README:    {df['has_readme'].sum():,}")
    print(f"  Output:        {output}")
    print(f"  Dataset hash:  {dataset_hash}")
    print(f"\nNext step: python extracao/gold_standard.py --input {output}")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def _flush_to_parquet(records: list[dict], output: Path) -> None:
    df = pd.DataFrame(records)
    df["has_readme"] = df["has_readme"].astype(bool)
    df["has_license"] = df["has_license"].astype(bool)
    df["has_wiki"] = df["has_wiki"].astype(bool)
    df.to_parquet(output, index=False, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect repositories from SEART GHS")
    parser.add_argument("--max-repos", type=int, default=15_000, help="Max repos to collect (default: 15000)")
    parser.add_argument("--language", type=str, default=None, help="Filter by primary language")
    parser.add_argument("--min-stars", type=int, default=10, help="Minimum stars (default: 10)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output Parquet path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--no-verify-ssl", dest="no_verify_ssl", action="store_true",
                        help="Disable SSL certificate verification (workaround for corporate proxies on Windows)")
    args = parser.parse_args()

    run_collection(
        max_repos=args.max_repos,
        language=args.language,
        min_stars=args.min_stars,
        output=args.output,
        resume=args.resume,
        verify_ssl=not args.no_verify_ssl,
    )


if __name__ == "__main__":
    main()
