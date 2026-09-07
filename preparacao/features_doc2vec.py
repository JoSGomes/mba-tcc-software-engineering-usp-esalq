"""Feature set Doc2Vec: replicando configuração PROMISE'24.

Parâmetros alinhados com baseline:
  vector_size=100, min_count=2, epochs=40, seed=GLOBAL_SEED, workers=1

Modelo treinado SOMENTE em train. Val/test inferidos via infer_vector().

Outputs (data/processed/):
  - doc2vec_model.bin
  - X_doc2vec_{split}.npy  para cada split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from preparacao.text_utils import README_MAX_CHARS, make_combined_texts

OUTPUT_DIR = Path("data/processed")
VECTOR_SIZE = 100
EPOCHS = 40
MIN_COUNT = 2


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_doc2vec(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {rid: i for i, rid in enumerate(df["repo_id"])}
    texts = make_combined_texts(df)

    train_idx = [id_to_idx[i] for i in splits["train"] if i in id_to_idx]
    train_corpus = [
        TaggedDocument(words=_tokenize(texts[i]), tags=[str(i)])
        for i in train_idx
    ]

    model = Doc2Vec(
        vector_size=VECTOR_SIZE,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        seed=GLOBAL_SEED,
        workers=1,  # determinismo: worker único
        dm=1,  # PV-DM (distributed memory) — padrão gensim
    )
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model_path = output_dir / "doc2vec_model.bin"
    model.save(str(model_path))
    print(f"Modelo salvo: {model_path}")

    results: dict[str, np.ndarray] = {}
    for split_name, ids in splits.items():
        if split_name == "seed":
            continue
        idx = [id_to_idx[i] for i in ids if i in id_to_idx]
        X = np.vstack([
            model.infer_vector(_tokenize(texts[i]), epochs=EPOCHS)
            for i in idx
        ])
        path = output_dir / f"X_doc2vec_{split_name}.npy"
        np.save(path, X)
        results[split_name] = X
        print(f"  {split_name}: {X.shape} -> {path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset_clean.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    splits = json.loads(args.splits.read_text())
    build_doc2vec(df, splits, args.output_dir)


if __name__ == "__main__":
    main()
