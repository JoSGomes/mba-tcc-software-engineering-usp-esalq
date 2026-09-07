"""Utilitários de texto compartilhados pelos feature builders.

Extraído de `features_doc2vec.py` para reuso por
`features_sbert.py` e `features_codebert.py`: os 3 embeddings usam
exatamente o mesmo texto combinado como entrada, garantindo comparação
justa entre eles.
"""

from __future__ import annotations

import pandas as pd

README_MAX_CHARS = 2_000


def make_combined_texts(df: pd.DataFrame, max_chars: int = README_MAX_CHARS) -> list[str]:
    """Combina nome + README (truncado) em uma única string por repositório."""

    def combine(row: pd.Series) -> str:
        name = str(row["name"] or "").strip()
        readme = str(row["readme_text"] or "").strip()[:max_chars]
        return (name + " " + readme).strip()

    return df.apply(combine, axis=1).tolist()


def resolve_device(device: str | None) -> str:
    """Resolve "auto"/None para "cuda" (se disponível) ou "cpu". Usado pelos
    builders de transformer (SBERT, CodeBERT) — Doc2Vec (gensim) não usa."""
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
