"""Feature set SBERT: `all-MiniLM-L6-v2` via sentence-transformers.

Diferente do Doc2Vec, o SBERT é um encoder pré-treinado e congelado — não há
etapa de "treino" sobre o corpus deste projeto. Por isso o raciocínio de
no-leakage é diferente do Doc2Vec: não existe risco de vazamento de
val/test para o "modelo" porque o encoder nunca é ajustado (fitted) aos
dados — a mesma transformação determinística é aplicada igualmente a todos
os splits, incluindo `test_ood`.

Outputs (data/processed/):
  - X_sbert_{split}.npy  para cada split (incluindo test_ood)

Os .npy gerados NÃO são commitados (ver CLAUDE.md) — encoder determinista,
regenerar é barato.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from preparacao.text_utils import make_combined_texts, resolve_device

OUTPUT_DIR = Path("data/processed")
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
DEFAULT_BATCH_SIZE = 64


def build_sbert(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {rid: i for i, rid in enumerate(df["repo_id"])}
    texts = make_combined_texts(df)

    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_device(device)
    print(f"  Device: {resolved_device}")
    model = SentenceTransformer(SBERT_MODEL_NAME, device=resolved_device)

    results: dict[str, np.ndarray] = {}
    for split_name, ids in splits.items():
        if split_name == "seed":
            continue
        idx = [id_to_idx[i] for i in ids if i in id_to_idx]
        split_texts = [texts[i] for i in idx]
        X = model.encode(
            split_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(split_texts) > 1000,
        ).astype(np.float32)
        path = output_dir / f"X_sbert_{split_name}.npy"
        np.save(path, X)
        results[split_name] = X
        print(f"  {split_name}: {X.shape} -> {path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_clean.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    splits = json.loads(args.splits.read_text())
    build_sbert(df, splits, args.output_dir, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
