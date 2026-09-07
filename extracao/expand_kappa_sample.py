"""Amostra repositórios adicionais para expandir a validação manual do kappa
(de 60 para ~100 repos revisados).

Amostra estratificada proporcionalmente à distribuição de classes já presente
em `data/raw/kappa_review.csv` (os 60 revisados até agora), excluindo:
  - repositórios já revisados (evita duplicar trabalho manual)
  - a classe "incerto" (Haiku se absteve — não entra no kappa, ver
    `extracao/compute_kappa.py::load_gold`)

Outputs:
  - data/raw/kappa_review_batch2.csv       schema igual a kappa_review.csv
                                            (manual_label vazio, a preencher)
  - data/processed/kappa_review_input.json 100 repos (60 já revisados + 40
                                            novos) com readme_text truncado
                                            — input para a ferramenta de
                                            revisão interativa (a construir
                                            separadamente)

A revisão manual em si (preencher manual_label) e o recálculo do kappa
(`extracao/compute_kappa.py --review data/raw/kappa_review_batch2.csv`, com
merge posterior dos dois CSVs) ficam para depois que o usuário terminar de
revisar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from extracao.compute_kappa import CLASSES as VALID_CLASSES

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
README_INPUT_MAX_CHARS = 2_000


def compute_stratified_counts(review_df: pd.DataFrame, n: int) -> dict[str, int]:
    """Conta quantos repos amostrar por classe, proporcional à distribuição
    de `label` em `review_df`, somando exatamente `n` (método do maior
    resto — evita viés de arredondamento)."""
    counts = review_df["label"].value_counts()
    counts = counts[counts.index.isin(VALID_CLASSES)]
    if counts.empty:
        raise ValueError("review_df não contém nenhuma classe válida em VALID_CLASSES")

    proportions = counts / counts.sum()
    raw = proportions * n
    floors = raw.apply(int)
    remainder = n - int(floors.sum())

    fracs = (raw - floors).sort_values(ascending=False)
    for cls in fracs.index[:remainder]:
        floors[cls] += 1

    return floors.to_dict()


def sample_new_repos(
    gold_df: pd.DataFrame,
    review_df: pd.DataFrame,
    n: int = 40,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """Amostra `n` repos de `gold_df`, estratificado proporcionalmente à
    distribuição de `review_df` (kappa_review.csv), excluindo 'incerto' e
    repos já em `review_df`."""
    already_reviewed = set(review_df["repo_id"])
    eligible = gold_df[
        gold_df["label"].isin(VALID_CLASSES) & ~gold_df["repo_id"].isin(already_reviewed)
    ]

    target_counts = compute_stratified_counts(review_df, n)

    parts = []
    for cls, count in target_counts.items():
        pool = eligible[eligible["label"] == cls]
        if len(pool) < count:
            raise ValueError(
                f"Pool insuficiente para classe '{cls}': precisa {count}, "
                f"disponível {len(pool)} (excluindo já revisados)"
            )
        parts.append(pool.sample(n=count, random_state=seed))

    result = pd.concat(parts, ignore_index=True)
    # embaralha a ordem final — evita blocos por classe na revisão manual
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_batch2_csv(sampled: pd.DataFrame) -> pd.DataFrame:
    """Mesmo schema de kappa_review.csv, manual_label vazio a preencher."""
    out = sampled[["repo_id", "name", "label", "confidence", "reasoning"]].copy()
    out["manual_label"] = ""
    return out


def build_review_input(
    review_df: pd.DataFrame,
    batch2_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    max_chars: int = README_INPUT_MAX_CHARS,
) -> list[dict]:
    """Combina os 60 já revisados + os N novos, junta com readme_text
    (truncado) do dataset limpo — pronto para a ferramenta de revisão."""
    combined = pd.concat([review_df, batch2_df], ignore_index=True)
    readme_by_id = dict(zip(dataset_df["repo_id"], dataset_df["readme_text"]))

    records = []
    for _, row in combined.iterrows():
        readme = str(readme_by_id.get(row["repo_id"], "") or "")[:max_chars]
        records.append(
            {
                "repo_id": row["repo_id"],
                "name": row["name"],
                "readme_text": readme,
                "label": row["label"],
                "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else None,
                "manual_label": (row.get("manual_label") or "").strip()
                if pd.notna(row.get("manual_label"))
                else "",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Expande a amostra de validação manual do kappa")
    parser.add_argument("--gold", type=Path, default=RAW_DIR / "gold_standard.csv")
    parser.add_argument("--reviewed", type=Path, default=RAW_DIR / "kappa_review.csv")
    parser.add_argument("--dataset", type=Path, default=PROCESSED_DIR / "dataset_clean.parquet")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--out-csv", type=Path, default=RAW_DIR / "kappa_review_batch2.csv")
    parser.add_argument("--out-json", type=Path, default=PROCESSED_DIR / "kappa_review_input.json")
    args = parser.parse_args()

    gold_df = pd.read_csv(args.gold)
    review_df = pd.read_csv(args.reviewed)
    dataset_df = pd.read_parquet(args.dataset)

    print(f"gold_standard: {len(gold_df)} repos | já revisados: {len(review_df)} repos")

    sampled = sample_new_repos(gold_df, review_df, n=args.n, seed=args.seed)
    print(f"Amostrados {len(sampled)} repos novos:")
    print(sampled["label"].value_counts().to_string())

    batch2 = build_batch2_csv(sampled)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    batch2.to_csv(args.out_csv, index=False)
    print(f"\nSalvo -> {args.out_csv}")

    review_input = build_review_input(review_df, batch2, dataset_df)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(review_input, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Salvo -> {args.out_json} ({len(review_input)} repos: "
          f"{len(review_df)} já revisados + {len(batch2)} novos)")


if __name__ == "__main__":
    main()
