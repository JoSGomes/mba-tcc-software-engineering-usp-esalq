"""Avaliação nas classes excluídas do treino ("other" e incerto).

Treina o modelo final (SBERT + XGBoost com SMOTE) sobre todo o conjunto
supervisionado (train+val+test_indist, 12.690 repositórios) e o aplica aos
2.310 repositórios rotulados como "other" ou incerto (split test_ood), que
nunca participaram de treino, seleção ou avaliação.

O classificador e os hiperparâmetros do modelo final são os escolhidos com
mais frequência no laço interno da validação cruzada aninhada
(experiments/nested_cv_full_selection_scores.csv).

A confiança de cada predição é a maior probabilidade retornada por
predict_proba (no XGBoost multiclasse, a softmax das pontuações das árvores),
e a taxa de rejeição é a proporção de repositórios com confiança inferior ao
corte operacional tau.

Saídas:
  - experiments/ood_analysis_result.json   resumo (taxa de rejeição, confiança média etc.)
  - experiments/ood_confidences.csv        confiança e predição por repositório

Uso: python avaliacao/ood_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import DATA_DIR, FULL_SELECTION_CV_PATH, LABEL_CLASSES, load_split
from modelagem.compare_models import build_pipeline

FEATURE = "sbert"
TAU = 0.5
OOD_RESULT_PATH = Path("experiments/ood_analysis_result.json")
OOD_CONF_PATH = Path("experiments/ood_confidences.csv")
SPLITS_PATH = Path("data/processed/splits.json")
LABELS_PATH = Path("data/raw/labels.parquet")


def load_full_supervised(
    feature: str, data_dir: Path = DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Concatena train+val+test_indist: o modelo final usa todo o conjunto
    supervisionado, já que a avaliação é feita no split test_ood."""
    X_train, y_train = load_split("train", feature, data_dir=data_dir)
    X_val, y_val = load_split("val", feature, data_dir=data_dir)
    X_test, y_test = load_split("test_indist", feature, data_dir=data_dir)
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    return X_all, y_all


def build_final_model(feature: str = FEATURE, scores_path: Path = FULL_SELECTION_CV_PATH):
    """Classificador e hiperparâmetros escolhidos com mais frequência no laço
    interno da validação cruzada aninhada para a representação `feature`."""
    scores = pd.read_csv(scores_path)
    model_type, n_model = Counter(scores[f"chosen_model_{feature}"]).most_common(1)[0]
    chosen = scores[scores[f"chosen_model_{feature}"] == model_type]
    params_json, n_params = Counter(chosen[f"best_params_{feature}"]).most_common(1)[0]
    params = json.loads(params_json)
    pipeline = build_pipeline(model_type)
    pipeline.set_params(**{f"clf__{k}": v for k, v in params.items()})
    print(f"Modelo final: {feature} + {model_type} {params} "
          f"(classificador em {n_model}/{len(scores)} rodadas; hiperparâmetros em {n_params}/{n_model})")
    return pipeline, model_type, params


def main(
    feature: str = FEATURE,
    tau: float = TAU,
    data_dir: Path = DATA_DIR,
    result_path: Path = OOD_RESULT_PATH,
    conf_path: Path = OOD_CONF_PATH,
) -> dict:
    pipeline, model_type, params = build_final_model(feature)

    X_all, y_all = load_full_supervised(feature, data_dir)
    print(f"Treinando sobre {X_all.shape[0]} repositórios supervisionados (train+val+test_indist)...")
    pipeline.fit(X_all, y_all)

    X_ood = np.load(data_dir / f"X_{feature}_test_ood.npy")
    ids = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["test_ood"]
    if len(ids) != X_ood.shape[0]:
        raise ValueError(f"test_ood: {len(ids)} ids, {X_ood.shape[0]} vetores")
    print(f"Aplicando aos {X_ood.shape[0]} repositórios das classes excluídas do treino...")

    proba = pipeline.predict_proba(X_ood)
    max_conf = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    rejected = max_conf < tau

    labels = pd.read_parquet(LABELS_PATH)[["repo_id", "label", "confidence", "label_source"]]
    conf = pd.DataFrame({
        "repo_id": ids,
        "ood_confidence": max_conf,
        "ood_pred": [LABEL_CLASSES[i] for i in preds],
        "ood_rejected": rejected,
    }).merge(labels, on="repo_id", how="left")
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf.to_csv(conf_path, index=False)
    print(f"Confiança por repositório salva em {conf_path}")

    result = {
        "feature": feature,
        "model_type": model_type,
        "params": params,
        "n_train_total": int(X_all.shape[0]),
        "n_ood": int(X_ood.shape[0]),
        "tau": tau,
        "rejection_rate": float(rejected.mean()),
        "n_rejected": int(rejected.sum()),
        "n_accepted": int((~rejected).sum()),
        "confidence_mean": float(max_conf.mean()),
        "confidence_median": float(np.median(max_conf)),
        "confidence_std": float(max_conf.std()),
        "predicted_class_distribution_all": {
            LABEL_CLASSES[c]: int((preds == c).sum()) for c in range(len(LABEL_CLASSES))
        },
        "predicted_class_distribution_accepted": {
            LABEL_CLASSES[c]: int((preds[~rejected] == c).sum()) for c in range(len(LABEL_CLASSES))
        },
        "seed": GLOBAL_SEED,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado salvo em {result_path}")
    print(f"\nTaxa de rejeição (tau={tau}): {result['rejection_rate']:.4f} "
          f"({result['n_rejected']}/{result['n_ood']}); confiança média {result['confidence_mean']:.4f}")
    return result


if __name__ == "__main__":
    main()
