"""Split estratificado 70/15/15 com índices fixos.

- train/val/test_indist: repos com label in {application, helper, extender}
- test_ood: repos com label in {other, incerto}

Splits salvos em data/processed/splits.json com repo_ids por subset.
Reprodutível a partir de GLOBAL_SEED.

Usage:
    python preparacao/split.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED

INPUT = Path("data/processed/dataset_clean.parquet")
OUTPUT = Path("data/processed/splits.json")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test_indist = remainder (0.15)


def split_dataset(df: pd.DataFrame, seed: int = GLOBAL_SEED) -> dict[str, list[str]]:
    in_dist = df[df["split_eligibility"] == "train_val_test"].copy()
    ood = df[df["split_eligibility"] == "test_only_ood"].copy()

    # 70 / 30 split (stratified)
    train_ids, temp_ids = train_test_split(
        in_dist["repo_id"].tolist(),
        test_size=1 - TRAIN_RATIO,
        stratify=in_dist["label"].tolist(),
        random_state=seed,
    )

    # 30 → 50/50 (val=15%, test_indist=15%)
    temp_df = in_dist[in_dist["repo_id"].isin(temp_ids)]
    val_ids, test_indist_ids = train_test_split(
        temp_df["repo_id"].tolist(),
        test_size=0.5,
        stratify=temp_df["label"].tolist(),
        random_state=seed,
    )

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test_indist": sorted(test_indist_ids),
        "test_ood": sorted(ood["repo_id"].tolist()),
        "seed": seed,
    }


def report_distribution(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    output_path: Path | None = None,
) -> dict:
    id_to_label = dict(zip(df["repo_id"], df["label"]))
    distribution: dict[str, dict] = {}

    for subset in ("train", "val", "test_indist", "test_ood"):
        ids = splits[subset]
        labels = [id_to_label[i] for i in ids if i in id_to_label]
        counts = pd.Series(labels).value_counts().to_dict()
        distribution[subset] = {"n": len(ids), "classes": counts}
        print(f"\n{subset} ({len(ids)} repos):")
        print(pd.Series(counts).to_string())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(distribution, indent=2, ensure_ascii=False))
        print(f"\nDistribuição salva em {output_path}")

    return distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Split estratificado 70/15/15")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Dataset carregado: {len(df):,} repos")

    splits = split_dataset(df, seed=args.seed)

    print(f"\nSplits gerados (seed={args.seed}):")
    print(f"  train:       {len(splits['train']):,}")
    print(f"  val:         {len(splits['val']):,}")
    print(f"  test_indist: {len(splits['test_indist']):,}")
    print(f"  test_ood:    {len(splits['test_ood']):,}")

    dist_path = args.output.parent / "split_distribution.json"
    report_distribution(df, splits, output_path=dist_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    print(f"\nSplits salvos em {args.output}")


if __name__ == "__main__":
    main()
