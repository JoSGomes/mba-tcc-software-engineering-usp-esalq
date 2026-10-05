"""Comparacao estatistica entre configuracoes a partir dos escores por fold
do nested CV (Tabelas 7 e 10 do TCC).

Os 10 folds externos compartilham ~89% dos dados de treino entre si, entao
nao sao 10 amostras independentes; testes que pressupoem independencia
(como o de Wilcoxon) superestimam a evidencia nesse cenario.

Este script nao roda nenhum modelo: le os escores por fold ja salvos e
calcula, para cada par de configuracoes:
  - descritivo pareado: diferenca media, desvio padrao das diferencas e
    numero de folds em que cada configuracao venceu;
  - teste t com reamostragem corrigido (Nadeau e Bengio, 2003), que infla a
    variancia pelo fator (1/k + n_teste/n_treino) para compensar a
    sobreposicao dos conjuntos de treino entre os folds, com IC de 95%.

Saida: experiments/comparacoes_estatisticas.csv

Uso:
    python modelagem/comparacoes_estatisticas.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from modelagem.common import FULL_SELECTION_CV_PATH

OUT_PATH = Path("experiments/comparacoes_estatisticas.csv")

# n_teste / n_treino de cada fold externo (10 folds sobre 12.690 repositorios)
N_TOTAL = 12_690
N_OUTER = 10

ANALYSES = {
    # Tabela 7 do TCC: estrategias de ajuste do RF sobre o Doc2Vec (modelagem/baseline.py)
    "rf_doc2vec": {
        "path": Path("experiments/nested_cv_scores.csv"),
        "configs": {"baseline": "RF padrão", "gridsearch": "RF + Grid Search",
                    "smote": "RF + SMOTE + Grid Search"},
    },
    # Tabela 10 do TCC: representacoes, com selecao completa dentro do fold
    "representacoes": {
        "path": FULL_SELECTION_CV_PATH,
        "configs": {"doc2vec": "Doc2Vec", "sbert": "SBERT", "codebert": "CodeBERT"},
    },
}


def corrected_resampled_ttest(d: np.ndarray, n_test: int, n_train: int, alpha: float = 0.05):
    """Teste t com reamostragem corrigido (Nadeau e Bengio, 2003) sobre as
    diferencas por fold `d`. Retorna (t, p bicaudal, IC inferior, IC superior)."""
    k = len(d)
    var = np.var(d, ddof=1) * (1 / k + n_test / n_train)
    se = np.sqrt(var)
    mean = d.mean()
    if se == 0:
        return (np.inf if mean else 0.0), (0.0 if mean else 1.0), mean, mean
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    half = stats.t.ppf(1 - alpha / 2, df=k - 1) * se
    return float(t), float(p), float(mean - half), float(mean + half)


def compare(scores: pd.DataFrame, configs: dict[str, str], metric: str = "f1_macro") -> list[dict]:
    n_test = N_TOTAL // N_OUTER
    n_train = N_TOTAL - n_test
    rows = []
    for a, b in combinations(configs, 2):
        xa = scores[f"{metric}_{a}"].to_numpy()
        xb = scores[f"{metric}_{b}"].to_numpy()
        d = xa - xb
        t, p, lo, hi = corrected_resampled_ttest(d, n_test, n_train)
        rows.append({
            "comparacao": f"{configs[a]} vs {configs[b]}",
            "metrica": metric,
            "media_a": xa.mean(), "media_b": xb.mean(),
            "dif_media": d.mean(), "dp_dif": d.std(ddof=1),
            "folds_a_melhor": int((d > 0).sum()), "folds_b_melhor": int((d < 0).sum()),
            "t_corrigido": t, "p_corrigido": p, "ic95_inf": lo, "ic95_sup": hi,
        })
    return rows


def main() -> None:
    rows = []
    for name, cfg in ANALYSES.items():
        scores = pd.read_csv(cfg["path"])
        for r in compare(scores, cfg["configs"]):
            rows.append({"analise": name, **r})
    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        cols = ["analise", "comparacao", "dif_media", "folds_a_melhor", "folds_b_melhor",
                "ic95_inf", "ic95_sup", "p_corrigido"]
        print(df[cols].round(4).to_string(index=False))
    print(f"\nSalvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
