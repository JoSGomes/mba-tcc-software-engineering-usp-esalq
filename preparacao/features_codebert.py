"""Feature set CodeBERT: `microsoft/codebert-base` via transformers.

Mesmo raciocínio de no-leakage do SBERT (`features_sbert.py`): encoder
pré-treinado e congelado, sem fit sobre o corpus deste projeto — aplicado
igualmente a todos os splits, incluindo `test_ood`.

Pooling: mean pooling ponderado pela attention mask (não CLS) — RoBERTa/
CodeBERT não tem objetivo de pré-treino tipo NSP, então o token CLS não é
especificamente treinado como representação de sentença (Reimers &
Gurevych, 2019, mostram mean pooling superando CLS em encoders RoBERTa).

Limitação a documentar na discussão do TCC (não é bloqueador técnico):
CodeBERT é pré-treinado em CodeSearchNet (código + docstring, majoritariamente
inglês) — não em prosa livre de README. Divergência de domínio esperada;
comparar o desempenho de CodeBERT com SBERT/Doc2Vec é o próprio
teste empírico dessa hipótese.

Outputs (data/processed/):
  - X_codebert_{split}.npy  para cada split (incluindo test_ood)

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
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
VECTOR_SIZE = 768
MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 16


def _mean_pool(last_hidden_state, attention_mask):
    """Mean pooling ponderado pela attention mask (ignora tokens de padding)."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_texts(texts, tokenizer, model, device, batch_size):
    import torch

    all_vecs = []
    i = 0
    current_batch_size = max(1, batch_size)
    while i < len(texts):
        batch = texts[i : i + current_batch_size]
        try:
            inputs = tokenizer(
                batch, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_vecs.append(pooled.cpu().numpy())
            i += current_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    return np.concatenate(all_vecs, axis=0).astype(np.float32)


def build_codebert(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {rid: i for i, rid in enumerate(df["repo_id"])}
    texts = make_combined_texts(df)

    from transformers import AutoModel, AutoTokenizer

    resolved_device = resolve_device(device)
    print(f"  Device: {resolved_device}")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(CODEBERT_MODEL_NAME).to(resolved_device)
    model.eval()  # dropout ativo por padrão quebraria determinismo silenciosamente

    results: dict[str, np.ndarray] = {}
    for split_name, ids in splits.items():
        if split_name == "seed":
            continue
        idx = [id_to_idx[i] for i in ids if i in id_to_idx]
        split_texts = [texts[i] for i in idx]
        X = _encode_texts(split_texts, tokenizer, model, resolved_device, batch_size)
        path = output_dir / f"X_codebert_{split_name}.npy"
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
    build_codebert(df, splits, args.output_dir, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
