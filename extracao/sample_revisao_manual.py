"""Sorteio das duas amostras da revisão manual dos rótulos.

Dois grupos de 30 repositórios, classificados pelo pesquisador sem acesso ao
rótulo do modelo:

  - baixa_confianca: supervisionados (application/helper/extender) rotulados
    em lote (500 caracteres), com confiança do modelo < 0,8, fora dos 100 já
    revisados na validação do kappa. 10 por classe, para que "extender" tenha
    casos suficientes.
  - classes_excluidas: repositórios do split test_ood, estratificados entre
    "other"/incerto e entre rejeitados/aceitos pelo modelo final
    (confiança < tau = 0,5), com a confiança de experiments/ood_confidences.csv
    (gerado por avaliacao/ood_analysis.py).

A amostra revisada, com o rótulo manual, está em
data/raw/revisao_manual_amostra.csv. Este script nunca a sobrescreve: se o
arquivo existir, apenas confere se o sorteio reproduz os mesmos 60
repositórios; caso contrário, grava a amostra com o rótulo manual vazio.

Uso: python extracao/sample_revisao_manual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED

RAW_DIR = Path("data/raw")
SAMPLE_PATH = RAW_DIR / "revisao_manual_amostra.csv"
OOD_CONF_PATH = Path("experiments/ood_confidences.csv")
SUPERVISED = ["application", "helper", "extender"]
CONF_MAX = 0.8
PER_CLASS = 10
EXCLUDED_STRATA = {("other", True): 8, ("other", False): 7, ("incerto", True): 8, ("incerto", False): 7}
COLS = ["repo_id", "name", "grupo", "estrato", "label", "confidence", "label_source",
        "ood_confidence", "ood_pred", "ood_rejected", "manual_label"]


def sample_low_confidence(labels: pd.DataFrame, reviewed_ids: set[str], seed: int = GLOBAL_SEED) -> pd.DataFrame:
    pool = labels[
        labels["label"].isin(SUPERVISED)
        & (labels["label_source"] == "haiku_auto_batch")
        & (labels["confidence"] < CONF_MAX)
        & ~labels["repo_id"].isin(reviewed_ids)
    ]
    parts = []
    for cls in SUPERVISED:
        sub = pool[pool["label"] == cls]
        if len(sub) < PER_CLASS:
            raise ValueError(f"pool insuficiente para {cls}: {len(sub)}")
        parts.append(sub.sample(n=PER_CLASS, random_state=seed).assign(estrato=f"{cls}, conf < {CONF_MAX}"))
    print(f"baixa_confianca: pool de {len(pool)} repos ({pool['label'].value_counts().to_dict()})")
    return pd.concat(parts).assign(grupo="baixa_confianca")


def sample_excluded(conf: pd.DataFrame, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    parts = []
    for (label, rejected), n in EXCLUDED_STRATA.items():
        sub = conf[(conf["label"] == label) & (conf["ood_rejected"] == rejected)]
        if len(sub) < n:
            raise ValueError(f"pool insuficiente para {label}/{rejected}: {len(sub)}")
        nome = "rejeitado" if rejected else "aceito"
        parts.append(sub.sample(n=n, random_state=seed).assign(estrato=f"{label}, {nome}"))
    print("classes_excluidas: pools", conf.groupby(["label", "ood_rejected"]).size().to_dict())
    return pd.concat(parts).assign(grupo="classes_excluidas")


def main() -> None:
    labels = pd.read_parquet(RAW_DIR / "labels.parquet")
    reviewed = pd.read_csv(RAW_DIR / "kappa_review_combined.csv")
    conf = pd.read_csv(OOD_CONF_PATH)

    amostra = pd.concat([
        sample_low_confidence(labels, set(reviewed["repo_id"])),
        sample_excluded(conf),
    ], ignore_index=True)
    amostra = amostra.sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)
    amostra["name"] = amostra["repo_id"].str.split("/").str[-1]
    amostra["manual_label"] = ""

    if SAMPLE_PATH.exists():
        existing = pd.read_csv(SAMPLE_PATH)
        same = list(existing["repo_id"]) == list(amostra["repo_id"])
        print(f"{SAMPLE_PATH} já existe (amostra revisada) — não sobrescrito. "
              f"Sorteio reproduz os mesmos 60 repositórios: {'sim' if same else 'NÃO'}")
        if not same:
            sys.exit(1)
        return

    amostra[COLS].to_csv(SAMPLE_PATH, index=False)
    print(f"Salvo -> {SAMPLE_PATH} ({len(amostra)} repos, manual_label vazio)")


if __name__ == "__main__":
    main()
