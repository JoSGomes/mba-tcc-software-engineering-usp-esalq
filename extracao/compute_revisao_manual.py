"""Métricas da revisão manual dos rótulos (amostra de baixa confiança e
amostra das classes excluídas do treino) e verificação de calibração do
modelo final.

Entradas:
  - data/raw/revisao_manual_amostra.csv     60 repos com rótulo do Haiku e
                                              rótulo manual final
  - data/raw/kappa_review_combined.csv       100 repos de alta confiança
  - data/raw/labels.parquet                  procedimento de rotulagem
  - experiments/nested_cv_full_selection_oof_probas.csv
                                              probabilidades out-of-fold
  - experiments/ood_confidences.csv          confiança no conjunto excluído

Saída: experiments/revisao_manual_result.json

Uso: python extracao/compute_revisao_manual.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/raw")
EXP_DIR = Path("experiments")
RESULT_PATH = EXP_DIR / "revisao_manual_result.json"
SUPERVISED = ["application", "helper", "extender"]
ALL4 = SUPERVISED + ["other"]
TAU = 0.5
N_BINS = 10
FEATURE = "sbert"


def agreement(df: pd.DataFrame, a: str, b: str, labels: list[str]) -> dict:
    cm = confusion_matrix(df[a], df[b], labels=labels)
    return {
        "n": int(len(df)),
        "concordancia": int((df[a] == df[b]).sum()),
        "kappa": float(cohen_kappa_score(df[a], df[b], labels=labels)),
        "labels": labels,
        "matriz_confusao (linhas=haiku, colunas=manual)": cm.tolist(),
    }


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict]]:
    """Erro de calibração esperado: média ponderada de |acurácia - confiança| por faixa."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        total += m.mean() * abs(acc - cf)
        bins.append({"faixa": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
                     "confianca_media": cf, "acuracia": acc})
    return float(total), bins


def main() -> None:
    labels = pd.read_parquet(RAW_DIR / "labels.parquet")[["repo_id", "label_source"]]
    result: dict = {}

    # ── kappa original (100 de alta confiança), por procedimento ───────────
    k = pd.read_csv(RAW_DIR / "kappa_review_combined.csv").merge(labels, on="repo_id")
    result["kappa_100"] = {
        "total": agreement(k, "label", "manual_label", ALL4),
        "individual (1.000 caracteres)": agreement(k[k.label_source == "haiku_auto"], "label", "manual_label", ALL4),
        "lote (500 caracteres)": agreement(k[k.label_source == "haiku_auto_batch"], "label", "manual_label", ALL4),
        "confianca_min": float(k.confidence.min()),
    }

    a = pd.read_csv(RAW_DIR / "revisao_manual_amostra.csv")

    # ── supervisionados de baixa confiança rotulados em lote ──────────────
    p3 = a[a.grupo == "baixa_confianca"]
    res_p3 = agreement(p3, "label", "manual_label", ALL4)
    res_p3["por_classe_haiku"] = {
        c: {"n": int((p3.label == c).sum()), "concordancia": int(((p3.label == c) & (p3.manual_label == c)).sum())}
        for c in SUPERVISED
    }
    sup = pd.read_parquet(RAW_DIR / "labels.parquet")
    sup = sup[sup.label.isin(SUPERVISED)]
    res_p3["populacao"] = {"supervisionados": int(len(sup)),
                           "confianca_menor_0_8": int((sup.confidence < 0.8).sum())}
    result["baixa_confianca"] = res_p3

    # ── classes excluídas do treino ───────────────────────────────────────
    p4 = a[a.grupo == "classes_excluidas"]
    res_p4 = {}
    for lab in ["incerto", "other"]:
        sub = p4[p4.label == lab]
        res_p4[f"haiku_{lab}"] = {"n": int(len(sub)),
                                  "manual": sub.manual_label.value_counts().reindex(ALL4, fill_value=0).to_dict()}
    for rej, nome in [(True, "rejeitados"), (False, "aceitos")]:
        sub = p4[p4.ood_rejected == rej]
        res_p4[nome] = {"n": int(len(sub)), "manual_other": int((sub.manual_label == "other").sum()),
                        "manual_uma_das_3": int(sub.manual_label.isin(SUPERVISED).sum())}
    res_p4["nota"] = "amostra estratificada (8/7 por estrato), não proporcional à população"
    conf = pd.read_csv(EXP_DIR / "ood_confidences.csv")
    res_p4["populacao"] = conf.groupby(["label", "ood_rejected"]).size().rename("n").reset_index().to_dict("records")
    result["classes_excluidas"] = res_p4

    # ── calibração e falsos rejeitados (probabilidades out-of-fold) ───────
    oof = pd.read_csv(EXP_DIR / "nested_cv_full_selection_oof_probas.csv")
    oof = oof[oof.feature == FEATURE]
    proba = oof[["p_application", "p_helper", "p_extender"]].to_numpy()
    conf_k = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == oof.y_true.to_numpy())
    e, bins = ece(conf_k, correct)
    by_class = {}
    for idx, c in enumerate(SUPERVISED):
        m = oof.y_true.to_numpy() == idx
        by_class[c] = {"n": int(m.sum()), "falsos_rejeitados": float((conf_k[m] < TAU).mean())}
    result["calibracao_oof"] = {
        "feature": FEATURE, "n": int(len(oof)), "ece": e, "n_bins": N_BINS, "bins": bins,
        "acuracia": float(correct.mean()), "confianca_media": float(conf_k.mean()),
        "tau": TAU,
        "falsos_rejeitados (classes conhecidas com confianca < tau)": float((conf_k < TAU).mean()),
        "falsos_rejeitados_por_classe": by_class,
        "rejeicao_classes_excluidas": float((conf.ood_confidence < TAU).mean()),
        "confianca_media_classes_excluidas": float(conf.ood_confidence.mean()),
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
