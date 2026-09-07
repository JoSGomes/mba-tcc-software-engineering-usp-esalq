"""Funcoes e constantes compartilhadas entre modelagem/baseline.py (Doc2Vec+RF)
e modelagem/compare_models.py (comparacao entre representacoes e classificadores).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon as wilcoxon_test
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

DATA_DIR      = Path("data/processed")
REGISTRY_PATH = Path("experiments/registry.csv")
FIGURES_DIR   = Path("experiments/figures")
LABEL_CLASSES = ["application", "helper", "extender"]

# Comparacao entre representacoes (nested CV) — nomes de arquivo distintos dos
# do baseline Doc2Vec+RF: nunca sobrescrevem nested_cv_scores.csv/
# nested_cv_summary.csv/wilcoxon_results.csv
NESTED_EMBEDDINGS_CV_PATH        = Path("experiments/nested_cv_embeddings_scores.csv")
NESTED_EMBEDDINGS_SUMMARY_PATH   = Path("experiments/nested_cv_embeddings_summary.csv")
WILCOXON_EMBEDDINGS_PATH         = Path("experiments/wilcoxon_embeddings_results.csv")
CONFUSION_MATRICES_JSON_PATH     = Path("experiments/nested_cv_confusion_matrices.json")
CONFUSION_COMPARISON_FIGURE_PATH = Path("experiments/figures/cm_nested_embeddings_comparison.tiff")

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "class_weight": [None, "balanced"],
}


def load_split(
    split: str, feature: str = "doc2vec", data_dir: Path = DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(data_dir / f"X_{feature}_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_application": float(f1_per[0]),
        "f1_helper": float(f1_per[1]),
        "f1_extender": float(f1_per[2]),
    }


def register_experiment(row: dict, registry_path: Path = REGISTRY_PATH) -> None:
    df = pd.read_csv(registry_path)
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(registry_path, index=False)


def save_confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_id: str,
    title: str,
    labels: list[str] = LABEL_CLASSES,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Verdadeiro", fontsize=11)
    plt.tight_layout()
    out = figures_dir / f"cm_{experiment_id}.tiff"
    fig.savefig(out, dpi=300, format="tiff")
    plt.close(fig)
    print(f"  Figura salva: {out}")


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta (tamanho do efeito nao-parametrico) para amostras emparelhadas."""
    diffs = np.array(a) - np.array(b)
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / len(diffs))


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147: return "negligivel"
    if d < 0.330: return "pequeno"
    if d < 0.474: return "medio"
    return "grande"


def run_nested_cv_configs(
    configs: dict[str, dict],
    y_all: np.ndarray,
    n_outer: int = 10,
    n_inner: int = 5,
    cv_scores_path: Path = NESTED_EMBEDDINGS_CV_PATH,
    wilcoxon_path: Path = WILCOXON_EMBEDDINGS_PATH,
    summary_path: Path = NESTED_EMBEDDINGS_SUMMARY_PATH,
    confusion_matrices_path: Path = CONFUSION_MATRICES_JSON_PATH,
    confusion_figure_path: Path = CONFUSION_COMPARISON_FIGURE_PATH,
    labels: list[str] = LABEL_CLASSES,
    seed: int = 42,
) -> None:
    """Nested CV generalizado comparando N configs (ex.: representacoes) — versao
    generica de modelagem/baseline.py::run_nested_cv.

    Cada config traz seu proprio X_all, um `pipeline_factory` (callable sem
    argumentos que retorna um ImbPipeline nao-ajustado) e um `param_grid`
    (chaves sem prefixo — prefixadas aqui com "clf__").

    O StratifiedKFold externo e calculado UMA UNICA VEZ a partir de y_all (os
    indices de fold nao dependem de X) e aplicado a cada config — garante o
    pareamento por fold exigido pelo Wilcoxon signed-rank test mesmo com
    arrays X_all distintos por config (doc2vec/sbert/codebert sao
    linha-a-linha alinhados por construcao: mesmos repos, mesma ordem, cada
    um com seu proprio X_{feature}_{split}.npy mas y_{split}.npy compartilhado).

    O StratifiedKFold interno seleciona hiperparametros via GridSearchCV
    dentro de cada fold externo, sem ver os dados de teste do fold — mesma
    metodologia de baseline.py::run_nested_cv.

    configs: {nome: {"X_all": np.ndarray, "pipeline_factory": Callable[[], Any],
                      "param_grid": dict}}

    Salva scores por fold, resumo (media +/- dp), Wilcoxon + Cliff's delta
    entre cada par de configs, e matriz de confusao agregada por config (soma
    dos n_outer folds externos — mais rigorosa que a matriz de um unico
    split) em contagem absoluta e normalizada por linha (%).
    """
    names = list(configs.keys())
    n_classes = len(labels)

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    # Folds calculados uma unica vez a partir de y_all (comum a todas as
    # configs) — aplicados ao X_all de cada config individualmente.
    dummy_X = np.zeros((len(y_all), 1))
    fold_indices = list(outer_cv.split(dummy_X, y_all))

    fold_records: list[dict] = []
    confusion_sums = {name: np.zeros((n_classes, n_classes), dtype=np.int64) for name in names}

    for fold, (tr_idx, te_idx) in enumerate(fold_indices):
        print(f"\n  Fold {fold + 1}/{n_outer}")
        record: dict = {"fold": fold}

        for name in names:
            cfg = configs[name]
            X_all = cfg["X_all"]
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]

            pipeline = cfg["pipeline_factory"]()
            param_grid_prefixed = {f"clf__{k}": v for k, v in cfg["param_grid"].items()}

            grid = GridSearchCV(
                pipeline, param_grid_prefixed, cv=inner_cv, scoring="f1_macro",
                n_jobs=-1, refit=True, verbose=0,
            )
            grid.fit(X_tr, y_tr)
            y_pred = grid.predict(X_te)
            m = compute_metrics(y_te, y_pred)

            confusion_sums[name] += confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))

            best_params = {k.replace("clf__", ""): v for k, v in grid.best_params_.items()}
            record[f"f1_macro_{name}"]       = m["f1_macro"]
            record[f"accuracy_{name}"]        = m["accuracy"]
            record[f"f1_application_{name}"]  = m["f1_application"]
            record[f"f1_helper_{name}"]       = m["f1_helper"]
            record[f"f1_extender_{name}"]     = m["f1_extender"]
            record[f"best_params_{name}"]     = json.dumps(best_params)
            record[f"best_cv_score_{name}"]   = float(grid.best_score_)

            print(f"    [{name}] F1-macro={m['f1_macro']:.4f}  F1-ext={m['f1_extender']:.4f}  "
                  f"best_CV={grid.best_score_:.4f}")

        fold_records.append(record)

    cv_scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(fold_records)
    scores_df.to_csv(cv_scores_path, index=False)
    print(f"\n  Scores por fold salvos em {cv_scores_path}")

    # ── Resumo media +/- std (todas as metricas, todas as configs) ─────────
    metric_cols = ["f1_macro", "accuracy", "f1_application", "f1_helper", "f1_extender"]
    print(f"\n  Media +/- desvio padrao ({n_outer} folds):")
    summary_rows = []
    for name in names:
        row = {"config": name}
        for mc in metric_cols:
            arr = scores_df[f"{mc}_{name}"].to_numpy()
            row[f"{mc}_mean"] = arr.mean()
            row[f"{mc}_std"]  = arr.std()
            row[f"{mc}_min"]  = arr.min()
            row[f"{mc}_max"]  = arr.max()
        arr = scores_df[f"best_cv_score_{name}"].to_numpy()
        row["best_cv_score_mean"] = arr.mean()
        row["best_cv_score_std"]  = arr.std()
        summary_rows.append(row)
        print(f"    {name:15s}: F1-macro={row['f1_macro_mean']:.4f} +/- {row['f1_macro_std']:.4f}"
              f"  F1-ext={row['f1_extender_mean']:.4f} +/- {row['f1_extender_std']:.4f}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Resumo salvo em {summary_path}")

    # ── Wilcoxon + Cliff's delta entre cada par de configs ──────────────────
    print("\n  Wilcoxon signed-rank test + Cliff's delta (bicaudal, alpha = 0.05):")
    rows = []
    for a, b in combinations(names, 2):
        arr_a = scores_df[f"f1_macro_{a}"].tolist()
        arr_b = scores_df[f"f1_macro_{b}"].tolist()
        try:
            stat, p = wilcoxon_test(arr_a, arr_b, alternative="two-sided")
        except ValueError:
            stat, p = 0.0, 1.0
            print(f"    [aviso] {a} vs {b}: diferenca zero em todos os folds — p=1.0")
        sig   = "sim" if p < 0.05 else "nao"
        delta = cliffs_delta(arr_a, arr_b)
        mag   = cliffs_magnitude(delta)
        mean_diff = float(np.mean(np.array(arr_a) - np.array(arr_b)))
        label = f"{a} vs {b}"
        print(f"    {label}")
        print(f"      W={stat:.4f}  p={p:.4f}  sig={sig}"
              f"  delta={delta:.4f} ({mag})  mean_diff={mean_diff:+.4f}")
        rows.append({
            "comparacao":            label,
            "statistic":             stat,
            "p_value":               p,
            "significativo_005":     sig,
            "cliffs_delta":          delta,
            "effect_size_magnitude": mag,
            "mean_diff":             mean_diff,
            "mean_a":                float(np.mean(arr_a)),
            "mean_b":                float(np.mean(arr_b)),
        })

    pd.DataFrame(rows).to_csv(wilcoxon_path, index=False)
    print(f"\n  Resultados Wilcoxon salvos em {wilcoxon_path}")

    # ── Matriz de confusao agregada (contagem + normalizada por linha) ─────
    matrices_raw = {name: confusion_sums[name].tolist() for name in names}
    matrices_norm = {}
    for name in names:
        cm = confusion_sums[name].astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # evita divisao por zero (classe ausente no fold)
        matrices_norm[name] = (cm / row_sums).tolist()

    confusion_matrices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(confusion_matrices_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": labels,
            "n_outer_folds": n_outer,
            "raw": matrices_raw,
            "normalized": matrices_norm,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Matrizes de confusao salvas em {confusion_matrices_path}")

    save_confusion_comparison_figure(
        matrices_raw={name: np.array(matrices_raw[name]) for name in names},
        matrices_norm={name: np.array(matrices_norm[name]) for name in names},
        labels=labels,
        config_names=names,
        out_path=confusion_figure_path,
    )


def save_confusion_comparison_figure(
    matrices_raw: dict[str, np.ndarray],
    matrices_norm: dict[str, np.ndarray],
    labels: list[str],
    config_names: list[str],
    out_path: Path,
) -> None:
    """Figura 2 x N: linha 1 = matriz de confusao agregada (contagem
    absoluta, soma dos folds externos do nested CV), linha 2 = normalizada
    por linha (recall por classe, %) — uma coluna por config (ex.:
    Doc2Vec/SBERT/CodeBERT), comparacao lado a lado."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(config_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(config_names):
        sns.heatmap(
            matrices_raw[name], annot=True, fmt=".0f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=axes[0, col], linewidths=0.5, cbar=False,
        )
        axes[0, col].set_title(name, fontsize=12)
        axes[0, col].set_xlabel("Predito")
        axes[0, col].set_ylabel("Verdadeiro" if col == 0 else "")

        sns.heatmap(
            matrices_norm[name], annot=True, fmt=".1%", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=axes[1, col], linewidths=0.5, cbar=False, vmin=0, vmax=1,
        )
        axes[1, col].set_xlabel("Predito")
        axes[1, col].set_ylabel("Verdadeiro (%)" if col == 0 else "")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="tiff")
    plt.close(fig)
    print(f"  Figura comparativa salva: {out_path}")
