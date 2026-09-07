"""Doc2Vec + Random Forest: baseline, GridSearchCV e SMOTE+GridSearchCV.

Experimentos (em ordem):
  1. RF baseline (params padrao) — sem tuning, sem SMOTE
  2. GridSearchCV sem SMOTE (CV=5 em train, scoring=f1_macro)
  3. SMOTE + GridSearchCV (SMOTE aplicado dentro de cada fold do CV via Pipeline)
  4. Nested 10x5 CV + Wilcoxon signed-rank test (comparacao estatistica entre os 3 experimentos)

Os experimentos 1-3 sao pulados se ja existirem no registry.csv.
O nested CV (passo 4) e pulado se experiments/nested_cv_scores.csv ja existir.
Para re-executar tudo, apagar as entradas do registry.csv e os arquivos de nested CV.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import wilcoxon as wilcoxon_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import (
    FIGURES_DIR,
    LABEL_CLASSES,
    REGISTRY_PATH,
)
from modelagem.common import RF_PARAM_GRID as PARAM_GRID
from modelagem.common import cliffs_delta as _cliffs_delta
from modelagem.common import cliffs_magnitude as _cliffs_magnitude
from modelagem.common import compute_metrics as _metrics
from modelagem.common import load_split as _load
from modelagem.common import register_experiment as _register
from modelagem.common import save_confusion_figure as _save_confusion_figure

NESTED_CV_PATH  = Path("experiments/nested_cv_scores.csv")
WILCOXON_PATH   = Path("experiments/wilcoxon_results.csv")
SUMMARY_PATH    = Path("experiments/nested_cv_summary.csv")


def _print_results(tag: str, val: dict, test: dict) -> None:
    print(f"\n  {tag}")
    print(f"    val  — acc={val['accuracy']:.4f}  F1_macro={val['f1_macro']:.4f}"
          f"  F1_app={val['f1_application']:.4f}"
          f"  F1_hlp={val['f1_helper']:.4f}"
          f"  F1_ext={val['f1_extender']:.4f}")
    print(f"    test — acc={test['accuracy']:.4f}  F1_macro={test['f1_macro']:.4f}"
          f"  F1_app={test['f1_application']:.4f}"
          f"  F1_hlp={test['f1_helper']:.4f}"
          f"  F1_ext={test['f1_extender']:.4f}")


def _print_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion matrix (test_indist) — linhas=verdadeiro, colunas=predito")
    print(f"  {'':>12}  {'app':>6}  {'helper':>6}  {'extender':>8}")
    for i, cls in enumerate(LABEL_CLASSES):
        print(f"  {cls:>12}  {cm[i,0]:>6}  {cm[i,1]:>6}  {cm[i,2]:>8}")
    print("\n  Classification report (test_indist):")
    print(classification_report(y_true, y_pred, target_names=LABEL_CLASSES, digits=4))


def run_baseline(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    """Experimento 1: RF com parametros padrao, sem tuning, sem SMOTE."""
    print("\n[1/3] RF baseline (n_estimators=100, default params)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)
    val_m = _metrics(y_val, rf.predict(X_val))
    test_m = _metrics(y_test, y_pred_test)
    _print_results("baseline", val_m, test_m)
    _print_confusion(y_test, y_pred_test)
    exp_id = str(uuid.uuid4())[:8]
    _save_confusion_figure(y_test, y_pred_test, exp_id, "RF Baseline — test_indist")
    _register({
        "experiment_id": exp_id,
        "phase": 4,
        "model": "RandomForest",
        "features": "doc2vec_100",
        "seed": GLOBAL_SEED,
        "params": json.dumps({"n_estimators": 100, "max_depth": None,
                              "min_samples_leaf": 1, "class_weight": None}),
        "param_grid": None,
        "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
        "val_f1_application": val_m["f1_application"], "val_f1_helper": val_m["f1_helper"],
        "val_f1_extender": val_m["f1_extender"],
        "test_accuracy": test_m["accuracy"], "test_f1_macro": test_m["f1_macro"],
        "test_f1_application": test_m["f1_application"], "test_f1_helper": test_m["f1_helper"],
        "test_f1_extender": test_m["f1_extender"],
        "notes": "baseline_default_params",
        "timestamp": datetime.now().isoformat(),
    })


def run_gridsearch(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    """Experimento 2: GridSearchCV sem SMOTE."""
    print("\n[2/3] GridSearchCV sem SMOTE (CV=5, scoring=f1_macro)...")
    grid = GridSearchCV(
        RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1),
        PARAM_GRID, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1, refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"\n  Best params: {grid.best_params_}")
    print(f"  Best CV f1_macro: {grid.best_score_:.4f}")
    y_pred_test = grid.predict(X_test)
    val_m = _metrics(y_val, grid.predict(X_val))
    test_m = _metrics(y_test, y_pred_test)
    _print_results("tuned (sem SMOTE)", val_m, test_m)
    _print_confusion(y_test, y_pred_test)
    exp_id = str(uuid.uuid4())[:8]
    _save_confusion_figure(y_test, y_pred_test, exp_id, "RF GridSearchCV (sem SMOTE) — test_indist")
    _register({
        "experiment_id": exp_id,
        "phase": 4,
        "model": "RandomForest",
        "features": "doc2vec_100",
        "seed": GLOBAL_SEED,
        "params": json.dumps(grid.best_params_),
        "param_grid": json.dumps(PARAM_GRID),
        "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
        "val_f1_application": val_m["f1_application"], "val_f1_helper": val_m["f1_helper"],
        "val_f1_extender": val_m["f1_extender"],
        "test_accuracy": test_m["accuracy"], "test_f1_macro": test_m["f1_macro"],
        "test_f1_application": test_m["f1_application"], "test_f1_helper": test_m["f1_helper"],
        "test_f1_extender": test_m["f1_extender"],
        "notes": f"gridsearchcv_cv_f1_macro={grid.best_score_:.4f}",
        "timestamp": datetime.now().isoformat(),
    })


def run_smote_gridsearch(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    """Experimento 3: SMOTE dentro de cada fold do CV via imblearn Pipeline.

    SMOTE e aplicado apenas no sub-conjunto de treino de cada fold — os dados
    sinteticos nunca vazam para o fold de validacao interna do CV nem para val/test.
    """
    print("\n[3/3] SMOTE + GridSearchCV (SMOTE dentro de cada fold do CV)...")

    # Prefixo 'rf__' porque os params pertencem ao estimador 'rf' no pipeline
    param_grid_smote = {f"rf__{k}": v for k, v in PARAM_GRID.items()}

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=GLOBAL_SEED)),
        # n_jobs=1 no RF: paralelismo gerenciado pelo GridSearchCV (n_jobs=-1)
        ("rf", RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=1)),
    ])

    grid = GridSearchCV(
        pipeline, param_grid_smote, cv=5, scoring="f1_macro",
        n_jobs=-1, verbose=1, refit=True,
    )
    grid.fit(X_train, y_train)

    best = {k.replace("rf__", ""): v for k, v in grid.best_params_.items()}
    print(f"\n  Best params (RF): {best}")
    print(f"  Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred_test = grid.predict(X_test)
    val_m = _metrics(y_val, grid.predict(X_val))
    test_m = _metrics(y_test, y_pred_test)
    _print_results("SMOTE + tuned", val_m, test_m)
    _print_confusion(y_test, y_pred_test)
    exp_id = str(uuid.uuid4())[:8]
    _save_confusion_figure(y_test, y_pred_test, exp_id, "RF SMOTE + GridSearchCV — test_indist")

    gate = test_m["accuracy"] >= 0.85
    status = "APROVADO" if gate else "REPROVADO"
    print(f"\n  Gate de aceitacao (acuracia >= 85% em test): {status}  ({test_m['accuracy']:.4f})")

    _register({
        "experiment_id": exp_id,
        "phase": 4,
        "model": "RandomForest+SMOTE",
        "features": "doc2vec_100",
        "seed": GLOBAL_SEED,
        "params": json.dumps(best),
        "param_grid": json.dumps(PARAM_GRID),
        "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
        "val_f1_application": val_m["f1_application"], "val_f1_helper": val_m["f1_helper"],
        "val_f1_extender": val_m["f1_extender"],
        "test_accuracy": test_m["accuracy"], "test_f1_macro": test_m["f1_macro"],
        "test_f1_application": test_m["f1_application"], "test_f1_helper": test_m["f1_helper"],
        "test_f1_extender": test_m["f1_extender"],
        "notes": f"smote+gridsearchcv_cv_f1_macro={grid.best_score_:.4f}",
        "timestamp": datetime.now().isoformat(),
    })


def run_nested_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_outer: int = 10,
    n_inner: int = 5,
    param_grid: dict | None = None,
    cv_scores_path: Path = NESTED_CV_PATH,
    wilcoxon_path: Path = WILCOXON_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> None:
    """Nested CV + Wilcoxon signed-rank test (dois lados, alpha=0.05).

    Loop externo: StratifiedKFold(n_outer) — gera as observacoes emparelhadas.
    Loop interno: StratifiedKFold(n_inner) dentro do GridSearchCV — seleciona
      hiperparametros sem ver os dados do fold externo.

    Usa train+val+test_indist (12.690 exemplos supervisionados).
    test_ood e mantido fora (ver avaliacao/ood_analysis.py).

    Referencia: Demsar (2006). Statistical Comparisons of Classifiers over
    Multiple Data Sets. JMLR 7, 1-30.
    """
    pg = param_grid if param_grid is not None else PARAM_GRID
    param_grid_smote = {f"rf__{k}": v for k, v in pg.items()}

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=GLOBAL_SEED)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=GLOBAL_SEED)

    fold_records: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all)):
        print(f"\n  Fold {fold + 1}/{n_outer}")
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # Baseline
        rf = RandomForestClassifier(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        m_b = _metrics(y_te, rf.predict(X_te))

        # GridSearchCV (sem SMOTE)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1),
            pg, cv=inner_cv, scoring="f1_macro", n_jobs=-1, refit=True, verbose=0,
        )
        grid.fit(X_tr, y_tr)
        m_g = _metrics(y_te, grid.predict(X_te))

        # SMOTE + GridSearchCV
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=GLOBAL_SEED)),
            ("rf", RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=1)),
        ])
        grid_s = GridSearchCV(
            pipeline, param_grid_smote, cv=inner_cv, scoring="f1_macro",
            n_jobs=-1, refit=True, verbose=0,
        )
        grid_s.fit(X_tr, y_tr)
        m_s = _metrics(y_te, grid_s.predict(X_te))

        best_g = json.dumps(grid.best_params_)
        best_s = json.dumps({k.replace("rf__", ""): v for k, v in grid_s.best_params_.items()})

        print(f"    F1-macro: baseline={m_b['f1_macro']:.4f}  "
              f"gridsearch={m_g['f1_macro']:.4f}  smote={m_s['f1_macro']:.4f}")
        print(f"    F1-ext:   baseline={m_b['f1_extender']:.4f}  "
              f"gridsearch={m_g['f1_extender']:.4f}  smote={m_s['f1_extender']:.4f}")
        print(f"    best_CV:  gridsearch={grid.best_score_:.4f}  smote={grid_s.best_score_:.4f}")

        # Salva todas as metricas do fold — base para tabela TCC (media +/- std) e Wilcoxon
        record: dict = {"fold": fold}
        for prefix, m, best_params, best_cv in [
            ("baseline",   m_b, None,   None),
            ("gridsearch", m_g, best_g, float(grid.best_score_)),
            ("smote",      m_s, best_s, float(grid_s.best_score_)),
        ]:
            record[f"f1_macro_{prefix}"]       = m["f1_macro"]
            record[f"accuracy_{prefix}"]        = m["accuracy"]
            record[f"f1_application_{prefix}"]  = m["f1_application"]
            record[f"f1_helper_{prefix}"]       = m["f1_helper"]
            record[f"f1_extender_{prefix}"]     = m["f1_extender"]
            if best_params is not None:
                record[f"best_params_{prefix}"]  = best_params
                record[f"best_cv_score_{prefix}"] = best_cv
        fold_records.append(record)

    cv_scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(fold_records)
    scores_df.to_csv(cv_scores_path, index=False)
    print(f"\n  Scores por fold salvos em {cv_scores_path}")

    # ── Resumo media +/- std (todas as metricas, todos os modelos) ────────────
    metric_cols = ["f1_macro", "accuracy", "f1_application", "f1_helper", "f1_extender"]
    prefixes = [
        ("baseline",   "RF baseline"),
        ("gridsearch", "RF + GridSearchCV"),
        ("smote",      "RF + SMOTE + GridSearchCV"),
    ]
    print(f"\n  Media +/- desvio padrao ({n_outer} folds):")
    summary_rows = []
    for prefix, name in prefixes:
        row = {"modelo": name}
        for mc in metric_cols:
            col = f"{mc}_{prefix}"
            if col in scores_df.columns:
                arr = scores_df[col].to_numpy()
                row[f"{mc}_mean"] = arr.mean()
                row[f"{mc}_std"]  = arr.std()
                row[f"{mc}_min"]  = arr.min()
                row[f"{mc}_max"]  = arr.max()
        # best_cv_score (inner CV) — indica gap de generalizacao
        bcs_col = f"best_cv_score_{prefix}"
        if bcs_col in scores_df.columns:
            arr = scores_df[bcs_col].to_numpy()
            row["best_cv_score_mean"] = arr.mean()
            row["best_cv_score_std"]  = arr.std()
        summary_rows.append(row)
        print(f"    {name:30s}: F1-macro={row['f1_macro_mean']:.4f} +/- {row['f1_macro_std']:.4f}"
              f"  F1-ext={row['f1_extender_mean']:.4f} +/- {row['f1_extender_std']:.4f}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Resumo salvo em {summary_path}")

    # ── Wilcoxon + Cliff's delta ───────────────────────────────────────────────
    pairs = [
        ("baseline",   "gridsearch", "Exp1 (baseline) vs Exp2 (GridSearchCV)"),
        ("baseline",   "smote",      "Exp1 (baseline) vs Exp3 (SMOTE+GridSearchCV)"),
        ("gridsearch", "smote",      "Exp2 (GridSearchCV) vs Exp3 (SMOTE+GridSearchCV)"),
    ]
    print("\n  Wilcoxon signed-rank test + Cliff's delta (bicaudal, alpha = 0.05):")
    rows = []
    for a, b, label in pairs:
        arr_a = scores_df[f"f1_macro_{a}"].tolist()
        arr_b = scores_df[f"f1_macro_{b}"].tolist()
        try:
            stat, p = wilcoxon_test(arr_a, arr_b, alternative="two-sided")
        except ValueError:
            # Diferenca zero em todos os folds: modelos identicos
            stat, p = 0.0, 1.0
            print(f"    [aviso] {label}: diferenca zero em todos os folds — p=1.0")
        sig   = "sim" if p < 0.05 else "nao"
        delta = _cliffs_delta(arr_a, arr_b)
        mag   = _cliffs_magnitude(delta)
        mean_diff = float(np.mean(np.array(arr_a) - np.array(arr_b)))
        print(f"    {label}")
        print(f"      W={stat:.4f}  p={p:.4f}  sig={sig}"
              f"  delta={delta:.4f} ({mag})  mean_diff={mean_diff:+.4f}")
        rows.append({
            "comparacao":          label,
            "statistic":           stat,
            "p_value":             p,
            "significativo_005":   sig,
            "cliffs_delta":        delta,
            "effect_size_magnitude": mag,
            "mean_diff":           mean_diff,
            "mean_a":              float(np.mean(arr_a)),
            "mean_b":              float(np.mean(arr_b)),
        })

    pd.DataFrame(rows).to_csv(wilcoxon_path, index=False)
    print(f"\n  Resultados Wilcoxon salvos em {wilcoxon_path}")


def main() -> None:
    X_train, y_train = _load("train")
    X_val, y_val = _load("val")
    X_test, y_test = _load("test_indist")

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    counts = np.bincount(y_train)
    for i, cls in enumerate(LABEL_CLASSES):
        print(f"  {cls}: {counts[i]} ({counts[i]/len(y_train)*100:.1f}%)")

    existing = pd.read_csv(REGISTRY_PATH)
    n_existing = len(existing)
    print(f"\nExperimentos ja existentes no registry: {n_existing}")

    # Gera figuras para experimentos ja registrados (sem re-registrar)
    for _, row in existing.iterrows():
        out = FIGURES_DIR / f"cm_{row['experiment_id']}.tiff"
        if not out.exists():
            params = {k: v for k, v in json.loads(row["params"]).items()}
            rf = RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1, **params)
            rf.fit(X_train, y_train)
            _save_confusion_figure(
                y_test, rf.predict(X_test),
                row["experiment_id"], f"RF {row['notes']} — test_indist",
            )

    if n_existing < 1:
        run_baseline(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        print("  [1/3] baseline ja registrado — pulando")

    if n_existing < 2:
        run_gridsearch(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        print("  [2/3] gridsearch sem SMOTE ja registrado — pulando")

    if n_existing < 3:
        run_smote_gridsearch(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        print("  [3/3] SMOTE+gridsearch ja registrado — pulando")

    print("\nResultados salvos em experiments/registry.csv")

    # Nested CV para comparacao estatistica (executa uma unica vez, ~3-4h)
    if not NESTED_CV_PATH.exists():
        print("\n[4/4] Nested 10x5 CV + Wilcoxon (~3-4h)...")
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        run_nested_cv(X_all, y_all)
    else:
        print(f"\n[4/4] Nested CV ja executado ({NESTED_CV_PATH}) — relendo Wilcoxon")
        scores_df = pd.read_csv(NESTED_CV_PATH)
        pairs = [
            ("f1_macro_baseline",   "f1_macro_gridsearch", "Exp1 vs Exp2"),
            ("f1_macro_baseline",   "f1_macro_smote",      "Exp1 vs Exp3"),
            ("f1_macro_gridsearch", "f1_macro_smote",      "Exp2 vs Exp3"),
        ]
        print("  Wilcoxon (a partir dos scores ja salvos):")
        for a, b, label in pairs:
            stat, p = wilcoxon_test(scores_df[a], scores_df[b], alternative="two-sided")
            print(f"    {label}: p={p:.4f}  {'*' if p < 0.05 else 'n.s.'}")


if __name__ == "__main__":
    main()
