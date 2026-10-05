"""Representacoes textuais e classificadores comparados no trabalho.

Define as tres representacoes (Doc2Vec, SBERT e CodeBERT), os tres
classificadores (Random Forest, Regressao Logistica e XGBoost), as grades de
hiperparametros e o pipeline SMOTE + classificador usados pela validacao
cruzada aninhada com selecao completa (modelagem/nested_cv_full_selection.py)
e pelo modelo final da avaliacao nas classes excluidas do treino
(avaliacao/ood_analysis.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import RF_PARAM_GRID

FEATURES = ["doc2vec", "sbert", "codebert"]
FEATURE_DIMS = {"doc2vec": 100, "sbert": 384, "codebert": 768}
MODEL_TYPES = ["rf", "logreg", "xgboost"]
MODEL_ORDER = {"rf": 0, "logreg": 1, "xgboost": 2}  # desempate: rf > logreg > xgboost
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
