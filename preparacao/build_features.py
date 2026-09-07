"""Orchestration module for feature engineering pipeline.

Calls the doc2vec feature builder, saves label encodings (y_*.npy) for
supervised splits, and metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED

OUTPUT_DIR = Path("data/processed")
LABEL_CLASSES = ("application", "helper", "extender")
LABEL_TO_INT = {cls: i for i, cls in enumerate(LABEL_CLASSES)}


def _save_labels(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {rid: i for i, rid in enumerate(df["repo_id"])}

    for split_name in ("train", "val", "test_indist"):
        if split_name not in splits:
            continue
        ids = splits[split_name]
        idx = [id_to_idx[i] for i in ids if i in id_to_idx]
        labels = df.iloc[idx]["label"].map(LABEL_TO_INT).values.astype(np.int32)
        path = output_dir / f"y_{split_name}.npy"
        np.save(path, labels)
        print(f"  {split_name}: {labels.shape} -> {path}")

    label_encoder_path = output_dir / "label_encoder.json"
    label_encoder_path.write_text(json.dumps(LABEL_TO_INT, indent=2))
    print(f"Label encoder salvo: {label_encoder_path}")


def _get_builder(feature_name: str):
    if feature_name == "doc2vec":
        from preparacao.features_doc2vec import build_doc2vec
        return build_doc2vec
    if feature_name == "sbert":
        from preparacao.features_sbert import build_sbert
        return build_sbert
    if feature_name == "codebert":
        from preparacao.features_codebert import build_codebert
        return build_codebert
    raise ValueError(f"Unknown feature set: {feature_name}")


# Builders que aceitam device= (encoders de transformer — SBERT, CodeBERT).
# Doc2Vec (gensim) não aceita e é ignorado nesse repasse.
_ACCEPTS_DEVICE = {"sbert", "codebert"}


def _build_requested_features(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    output_dir: Path,
    feature_sets: list[str],
    device: str | None = None,
) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for feature_name in feature_sets:
        print(f"\n[{feature_name.upper()}] Building features...")
        builder = _get_builder(feature_name)
        kwargs = {"device": device} if feature_name in _ACCEPTS_DEVICE else {}
        results = builder(df, splits, output_dir, **kwargs)
        metadata[feature_name] = {
            split: {"shape": list(X.shape), "dtype": str(X.dtype)}
            for split, X in results.items()
            if split != "seed"
        }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_clean.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["doc2vec", "sbert", "codebert"],
        default=["doc2vec"],
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device para builders de transformer (sbert/codebert); ignorado por doc2vec.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    splits = json.loads(args.splits.read_text())

    print("[LABELS] Saving label encodings...")
    _save_labels(df, splits, args.output_dir)

    print(f"\n[FEATURES] Building {len(args.feature_sets)} feature set(s)...")
    features_meta = _build_requested_features(
        df, splits, args.output_dir, args.feature_sets, device=args.device
    )

    # Merge com o meta existente em disco: rodar com um subconjunto de
    # --feature-sets (ex.: só sbert/codebert) nao deve apagar o registro de
    # feature sets gerados em execucoes anteriores (ex.: doc2vec) cujos .npy
    # continuam validos em data/processed/.
    meta_path = args.output_dir / "features_meta.json"
    all_features = {}
    if meta_path.exists():
        try:
            all_features = json.loads(meta_path.read_text()).get("features", {})
        except (json.JSONDecodeError, OSError):
            all_features = {}
    all_features.update(features_meta)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "seed": GLOBAL_SEED,
        "feature_sets": sorted(all_features.keys()),
        "features": all_features,
        "label_classes": list(LABEL_CLASSES),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMetadata salvo: {meta_path}")
    print("Pipeline completo.")


if __name__ == "__main__":
    main()
