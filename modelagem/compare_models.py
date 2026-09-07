"""Comparacao entre representacoes textuais (Doc2Vec/SBERT/CodeBERT) e
classificadores (RandomForest/LogisticRegression/XGBoost).

Etapa de selecao (--phase a): GridSearchCV simples (SMOTE + estimador via
ImbPipeline, cv=5, scoring=f1_macro) nas 9 combinacoes representacao x
classificador. Escolhe o melhor classificador por representacao
(val_f1_macro, empate resolvido pela ordem fixa rf > logreg > xgboost) —
usado como entrada da etapa de comparacao.

Etapa de comparacao (--phase b): nested 10x5 CV comparando as 3
representacoes entre si — cada uma com seu melhor classificador escolhido na
etapa anterior — + matriz de confusao agregada (soma dos 10 folds externos,
contagem e normalizada, comparacao lado a lado entre representacoes). Usa
train+val+test_indist concatenados; test_ood fica fora (reservado para a
avaliacao out-of-distribution, ver avaliacao/ood_analysis.py).

Resultados salvos em experiments/registry.csv (etapa de selecao, phase=5,
mesmo schema do baseline Doc2Vec+RF) e
experiments/nested_cv_embeddings_{scores,summary}.csv +
wilcoxon_embeddings_results.csv + nested_cv_confusion_matrices.json (etapa
de comparacao, nomes de arquivo distintos). modelagem/baseline.py e seus
artefatos (nested_cv_scores.csv, nested_cv_summary.csv, wilcoxon_results.csv)
nao sao tocados por este script.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import (
    CONFUSION_COMPARISON_FIGURE_PATH,
    CONFUSION_MATRICES_JSON_PATH,
    DATA_DIR,
    FIGURES_DIR,
    NESTED_EMBEDDINGS_CV_PATH,
    NESTED_EMBEDDINGS_SUMMARY_PATH,
    REGISTRY_PATH,
    RF_PARAM_GRID,
    WILCOXON_EMBEDDINGS_PATH,
    compute_metrics,
    load_split,
    register_experiment,
    run_nested_cv_configs,
    save_confusion_figure,
)

PHASE = 5
FEATURES = ["doc2vec", "sbert", "codebert"]
FEATURE_DIMS = {"doc2vec": 100, "sbert": 384, "codebert": 768}
MODEL_TYPES = ["rf", "logreg", "xgboost"]
MODEL_ORDER = {"rf": 0, "logreg": 1, "xgboost": 2}  # tie-break: rf > logreg > xgboost
MODEL_LABELS = {
    "rf": "RandomForest+SMOTE",
    "logreg": "LogisticRegression+SMOTE",
    "xgboost": "XGBoost+SMOTE",
}

LOGREG_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "max_iter": [1000],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
}


def _param_grid_for(model_type: str) -> dict:
    if model_type == "rf":
        return RF_PARAM_GRID
    if model_type == "logreg":
        return LOGREG_PARAM_GRID
    if model_type == "xgboost":
        return XGB_PARAM_GRID
    raise ValueError(f"model_type desconhecido: {model_type}")


def build_pipeline(model_type: str) -> ImbPipeline:
    """SMOTE + estimador — mesmo padrao de modelagem/baseline.py::run_smote_gridsearch."""
    if model_type == "rf":
        # n_jobs=1: paralelismo gerenciado pelo GridSearchCV (n_jobs=-1)
        est = RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=1)
    elif model_type == "logreg":
        est = LogisticRegression(random_state=GLOBAL_SEED)
    elif model_type == "xgboost":
        # tree_method="hist" sem device="cuda": mantem XGBoost em CPU — GPU
        # fica livre para a geracao de embeddings (SBERT/CodeBERT)
        est = XGBClassifier(
            random_state=GLOBAL_SEED, tree_method="hist", n_jobs=1,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"model_type desconhecido: {model_type}")

    return ImbPipeline([
        ("smote", SMOTE(random_state=GLOBAL_SEED)),
        ("clf", est),
    ])


def feature_label(feature: str) -> str:
    return f"{feature}_{FEATURE_DIMS[feature]}"


def already_registered(registry_path: Path, feature: str, model_type: str) -> bool:
    df = pd.read_csv(registry_path)
    if df.empty:
        return False
    mask = (
        (df["phase"] == PHASE)
        & (df["features"] == feature_label(feature))
        & (df["model"] == MODEL_LABELS[model_type])
    )
    return bool(mask.any())


def run_one_combination(
    feature: str,
    model_type: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    cv: int = 5,
    registry_path: Path = REGISTRY_PATH,
    figures_dir: Path = FIGURES_DIR,
    param_grid: dict | None = None,
) -> dict:
    """Roda GridSearchCV para uma combinacao embedding x classificador e
    registra o resultado em registry_path. Retorna o registro (dict)."""
    pg = param_grid if param_grid is not None else _param_grid_for(model_type)
    param_grid_prefixed = {f"clf__{k}": v for k, v in pg.items()}

    pipeline = build_pipeline(model_type)
    grid = GridSearchCV(
        pipeline, param_grid_prefixed, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1, refit=True,
    )
    grid.fit(X_train, y_train)

    best = {k.replace("clf__", ""): v for k, v in grid.best_params_.items()}
    y_pred_test = grid.predict(X_test)
    val_m = compute_metrics(y_val, grid.predict(X_val))
    test_m = compute_metrics(y_test, y_pred_test)

    exp_id = str(uuid.uuid4())[:8]
    save_confusion_figure(
        y_test, y_pred_test, exp_id,
        f"{MODEL_LABELS[model_type]} ({feature}) — test_indist",
        figures_dir=figures_dir,
    )

    row = {
        "experiment_id": exp_id,
        "phase": PHASE,
        "model": MODEL_LABELS[model_type],
        "features": feature_label(feature),
        "seed": GLOBAL_SEED,
        "params": json.dumps(best),
        "param_grid": json.dumps(pg),
        "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
        "val_f1_application": val_m["f1_application"], "val_f1_helper": val_m["f1_helper"],
        "val_f1_extender": val_m["f1_extender"],
        "test_accuracy": test_m["accuracy"], "test_f1_macro": test_m["f1_macro"],
        "test_f1_application": test_m["f1_application"], "test_f1_helper": test_m["f1_helper"],
        "test_f1_extender": test_m["f1_extender"],
        "notes": f"fase_a_cv_f1_macro={grid.best_score_:.4f}",
        "timestamp": datetime.now().isoformat(),
    }
    register_experiment(row, registry_path=registry_path)
    print(f"  [{feature}/{model_type}] val_f1_macro={val_m['f1_macro']:.4f}  "
          f"test_f1_macro={test_m['f1_macro']:.4f}  CV={grid.best_score_:.4f}")
    return row


def run_phase_a(
    features: list[str] = FEATURES,
    model_types: list[str] = MODEL_TYPES,
    data_dir: Path = DATA_DIR,
    registry_path: Path = REGISTRY_PATH,
    figures_dir: Path = FIGURES_DIR,
    cv: int = 5,
    param_grids: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Roda as combinacoes embedding x classificador (default: 3x3=9),
    pulando combinacoes ja registradas em registry.csv (phase=5)."""
    results = []
    for feature in features:
        print(f"\n=== Embedding: {feature} ===")
        X_train, y_train = load_split("train", feature, data_dir=data_dir)
        X_val, y_val = load_split("val", feature, data_dir=data_dir)
        X_test, y_test = load_split("test_indist", feature, data_dir=data_dir)
        for model_type in model_types:
            if already_registered(registry_path, feature, model_type):
                print(f"  [{feature}/{model_type}] ja registrado — pulando")
                continue
            pg = param_grids.get(model_type) if param_grids else None
            row = run_one_combination(
                feature, model_type,
                X_train, y_train, X_val, y_val, X_test, y_test,
                cv=cv, registry_path=registry_path, figures_dir=figures_dir,
                param_grid=pg,
            )
            results.append(row)
    return pd.DataFrame(results)


def select_best_per_embedding(
    registry_path: Path = REGISTRY_PATH, features: list[str] = FEATURES
) -> pd.DataFrame:
    """Para cada embedding, seleciona o classificador com maior val_f1_macro
    entre os experimentos da etapa de selecao (phase=5). Empate resolvido pela ordem
    fixa rf > logreg > xgboost."""
    df = pd.read_csv(registry_path)
    df = df[df["phase"] == PHASE].copy()
    if df.empty:
        raise ValueError(
            f"Nenhum experimento da etapa de selecao (phase={PHASE}) encontrado em {registry_path}"
        )

    label_to_model_type = {v: k for k, v in MODEL_LABELS.items()}
    df["_feature"] = df["features"].apply(lambda f: f.rsplit("_", 1)[0])
    df["_model_type"] = df["model"].map(label_to_model_type)
    df["_order"] = df["_model_type"].map(MODEL_ORDER)

    best_rows = []
    for feature in features:
        sub = df[df["_feature"] == feature]
        if sub.empty:
            continue
        sub = sub.sort_values(by=["val_f1_macro", "_order"], ascending=[False, True])
        best_rows.append(sub.iloc[0])

    if not best_rows:
        raise ValueError(f"Nenhum embedding em {features} encontrado nos experimentos da etapa de selecao")

    best_df = pd.DataFrame(best_rows).drop(columns=["_feature", "_model_type", "_order"])
    return best_df.reset_index(drop=True)


def run_phase_b(
    features: list[str] = FEATURES,
    data_dir: Path = DATA_DIR,
    registry_path: Path = REGISTRY_PATH,
    n_outer: int = 10,
    n_inner: int = 5,
    cv_scores_path: Path = NESTED_EMBEDDINGS_CV_PATH,
    wilcoxon_path: Path = WILCOXON_EMBEDDINGS_PATH,
    summary_path: Path = NESTED_EMBEDDINGS_SUMMARY_PATH,
    confusion_matrices_path: Path = CONFUSION_MATRICES_JSON_PATH,
    confusion_figure_path: Path = CONFUSION_COMPARISON_FIGURE_PATH,
) -> None:
    """Nested 10x5 CV comparando as representacoes entre si — cada uma com
    o classificador vencedor da etapa de selecao (select_best_per_embedding).
    Requer que a etapa de selecao ja tenha rodado para os `features` pedidos.

    Usa train+val+test_indist concatenados por representacao; test_ood fica
    fora (reservado para avaliacao/ood_analysis.py). Os 3
    arrays X_all (um por embedding) sao linha-a-linha alinhados por
    construcao (mesmos repos, mesma ordem) — o pareamento por fold exigido
    pelo Wilcoxon e garantido dentro de run_nested_cv_configs.
    """
    best = select_best_per_embedding(registry_path=registry_path, features=features)
    label_to_model_type = {v: k for k, v in MODEL_LABELS.items()}

    configs: dict[str, dict] = {}
    y_all: np.ndarray | None = None
    print("Melhor classificador por embedding (escolhido na etapa de selecao):")
    for _, row in best.iterrows():
        feature = row["features"].rsplit("_", 1)[0]
        model_type = label_to_model_type[row["model"]]
        print(f"  {feature:10s} -> {MODEL_LABELS[model_type]:28s} "
              f"(val_f1_macro={row['val_f1_macro']:.4f})")

        X_train, y_train = load_split("train", feature, data_dir=data_dir)
        X_val, y_val = load_split("val", feature, data_dir=data_dir)
        X_test, y_test = load_split("test_indist", feature, data_dir=data_dir)

        if y_all is None:
            # y_*.npy independe do embedding — concatenado uma unica vez
            y_all = np.concatenate([y_train, y_val, y_test])

        configs[feature] = {
            "X_all": np.concatenate([X_train, X_val, X_test]),
            "pipeline_factory": partial(build_pipeline, model_type),
            "param_grid": _param_grid_for(model_type),
        }

    run_nested_cv_configs(
        configs, y_all,
        n_outer=n_outer, n_inner=n_inner,
        cv_scores_path=cv_scores_path, wilcoxon_path=wilcoxon_path,
        summary_path=summary_path,
        confusion_matrices_path=confusion_matrices_path,
        confusion_figure_path=confusion_figure_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["a", "b"], required=True,
        help="a: GridSearchCV nas 9 combinacoes embedding x classificador. "
             "b: nested CV entre os embeddings (requer a etapa de selecao ja executada).",
    )
    args = parser.parse_args()

    if args.phase == "a":
        print("Etapa de selecao: GridSearchCV (SMOTE + estimador) nas combinacoes embedding x classificador")
        run_phase_a()
        print("\nResultados salvos em experiments/registry.csv (phase=5)")
        best = select_best_per_embedding()
        print("\nMelhor classificador por embedding (val_f1_macro):")
        for _, row in best.iterrows():
            print(f"  {row['features']:15s} -> {row['model']:28s} val_f1_macro={row['val_f1_macro']:.4f}")
    elif args.phase == "b":
        print("Etapa de comparacao: nested 10x5 CV entre os embeddings (cada um com o melhor classificador da etapa de selecao)")
        run_phase_b()
        print(f"\nResultados salvos em {NESTED_EMBEDDINGS_CV_PATH}, {NESTED_EMBEDDINGS_SUMMARY_PATH}, "
              f"{WILCOXON_EMBEDDINGS_PATH}, {CONFUSION_MATRICES_JSON_PATH}")


if __name__ == "__main__":
    main()
