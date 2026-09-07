"""Limpeza e preparação do dataset para modelagem.

Merge repos + labels, deduplicação, filtragem por texto mínimo,
e marcação de elegibilidade de split (train_val_test vs test_only_ood).

Usage:
    python preparacao/clean.py
    python preparacao/clean.py --labels data/raw/labels.parquet
    python preparacao/clean.py --labels data/raw/gold_standard.csv  # modo dev com 500 repos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAIN_LABELS = {"application", "helper", "extender"}
OOD_LABELS = {"other", "incerto"}
MIN_TEXT_CHARS = 3  # repo com name < 3 chars e sem readme é descartado

OUTPUT = Path("data/processed/dataset_clean.parquet")


def load_labels(labels_path: Path) -> pd.DataFrame:
    if labels_path.suffix == ".csv":
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_parquet(labels_path)
    return df[["repo_id", "label", "confidence"]].copy()


def compute_age_days(df: pd.DataFrame) -> pd.Series:
    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    collected = pd.to_datetime(df["collected_at"], utc=True, errors="coerce")
    return (collected - created).dt.days.fillna(0).astype(int)


def make_text(name: str, readme: str) -> str:
    name = str(name or "").strip()
    readme = str(readme or "").strip()
    return (name + " " + readme).strip()


def assign_eligibility(label: str) -> str:
    if label in TRAIN_LABELS:
        return "train_val_test"
    return "test_only_ood"


def clean(repos_path: Path, labels_path: Path, output_path: Path) -> pd.DataFrame:
    repos = pd.read_parquet(repos_path)
    labels = load_labels(labels_path)

    print(f"Repos carregados: {len(repos):,}")
    print(f"Labels carregados: {len(labels):,}")

    df = repos.merge(labels, on="repo_id", how="inner")
    print(f"Após merge: {len(df):,}")

    # Deduplicar por repo_id (manter primeira ocorrência)
    before = len(df)
    df = df.drop_duplicates(subset=["repo_id"], keep="first")
    print(f"Após deduplicação: {len(df):,} ({before - len(df)} duplicatas removidas)")

    # Remover repos sem texto utilizável
    df["text"] = df.apply(lambda r: make_text(r["name"], r["readme_text"]), axis=1)
    has_text = df["text"].str.len() >= MIN_TEXT_CHARS
    before = len(df)
    df = df[has_text].copy()
    print(f"Após filtro de texto mínimo: {len(df):,} ({before - len(df)} removidos)")

    # Coluna de elegibilidade de split
    df["split_eligibility"] = df["label"].map(assign_eligibility)
    df["age_days"] = compute_age_days(df)

    print(f"\nDistribuição de labels:")
    print(df["label"].value_counts().to_string())
    print(f"\nElegibilidade:")
    print(df["split_eligibility"].value_counts().to_string())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSalvo em {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpeza do dataset para modelagem")
    parser.add_argument("--repos", type=Path, default=Path("data/raw/repos.parquet"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/raw/labels.parquet"),
        help="Labels do dataset inteiro (labels.parquet) ou gold_standard.csv para dev",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not args.labels.exists():
        alt = Path("data/raw/gold_standard.csv")
        print(f"AVISO: {args.labels} não encontrado. Usando {alt} (modo dev — 500 repos).")
        args.labels = alt

    clean(args.repos, args.labels, args.output)


if __name__ == "__main__":
    main()
