"""Análise Out-of-Distribution (OOD).

Treina o modelo final (melhor combinação encontrada na comparação entre
representações e classificadores: SBERT +
XGBoost+SMOTE, hiperparâmetros lidos de experiments/registry.csv) sobre TODO
o conjunto supervisionado (train+val+test_indist, 12.690 repositórios) e
aplica esse modelo aos 2.310 repositórios reservados como test_ood (rótulos
Haiku "other"/"incerto", nunca usados em treino, seleção ou avaliação até
aqui — ver preparacao/split.py e docs/feature_schema.md).

Como esses repositórios não pertencem às 3 classes supervisionadas
(application/helper/extender), o esperado é que o modelo tenha baixa
confiança neles. A taxa de rejeição (fração de repositórios com
max(softmax) abaixo do limiar tau) mede quão bem o modelo reconhece que
esses casos estão fora do seu domínio de treino, em vez de forçar uma
classificação com alta confiança.

Uso: python avaliação/ood_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import DATA_DIR, FIGURES_DIR, LABEL_CLASSES, REGISTRY_PATH, load_split
from modelagem.compare_models import MODEL_LABELS, build_pipeline, select_best_per_embedding

FEATURE = "sbert"
TAU = 0.5
OOD_RESULT_PATH = Path("experiments/ood_analysis_result.json")
OOD_FIGURE_PATH = Path("experiments/figures/ood_confidence_histogram.tiff")


def load_full_supervised(
    feature: str, data_dir: Path = DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Concatena train+val+test_indist — modelo final treinado sobre todo o
    conjunto rotulado disponível, já que test_ood (não test_indist) é o
    conjunto reservado para esta avaliação."""
    X_train, y_train = load_split("train", feature, data_dir=data_dir)
    X_val, y_val = load_split("val", feature, data_dir=data_dir)
    X_test, y_test = load_split("test_indist", feature, data_dir=data_dir)
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    return X_all, y_all


def build_final_model(
    feature: str = FEATURE, registry_path: Path = REGISTRY_PATH
):
    """Reconstroi o pipeline vencedor (SMOTE + classificador) da etapa de
    seleção com
    os hiperparâmetros exatos já registrados em registry.csv — nada é
    re-buscado por Grid Search aqui."""
    best_row = select_best_per_embedding(registry_path=registry_path, features=[feature]).iloc[0]
    label_to_model_type = {v: k for k, v in MODEL_LABELS.items()}
    model_type = label_to_model_type[best_row["model"]]
    params = json.loads(best_row["params"])

    pipeline = build_pipeline(model_type)
    pipeline.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipeline, model_type, params, float(best_row["val_f1_macro"])


def main(
    feature: str = FEATURE,
    tau: float = TAU,
    data_dir: Path = DATA_DIR,
    registry_path: Path = REGISTRY_PATH,
    result_path: Path = OOD_RESULT_PATH,
    figure_path: Path = OOD_FIGURE_PATH,
) -> dict:
    pipeline, model_type, params, val_f1_macro = build_final_model(feature, registry_path)
    print(f"Modelo final: {feature} + {model_type} {params} (val_f1_macro na seleção = {val_f1_macro:.4f})")

    X_all, y_all = load_full_supervised(feature, data_dir)
    print(f"Treinando sobre {X_all.shape[0]} repositórios supervisionados (train+val+test_indist)...")
    pipeline.fit(X_all, y_all)

    X_ood = np.load(data_dir / f"X_{feature}_test_ood.npy")
    print(f"Aplicando aos {X_ood.shape[0]} repositórios OOD (other/incerto)...")

    proba = pipeline.predict_proba(X_ood)
    max_conf = proba.max(axis=1)
    preds = proba.argmax(axis=1)

    rejected = max_conf < tau
    rejection_rate = float(rejected.mean())

    pred_class_counts = {
        LABEL_CLASSES[c]: int((preds == c).sum()) for c in range(len(LABEL_CLASSES))
    }
    pred_class_counts_accepted = {
        LABEL_CLASSES[c]: int((preds[~rejected] == c).sum()) for c in range(len(LABEL_CLASSES))
    }

    result = {
        "feature": feature,
        "model_type": model_type,
        "params": params,
        "n_train_total": int(X_all.shape[0]),
        "n_ood": int(X_ood.shape[0]),
        "tau": tau,
        "rejection_rate": rejection_rate,
        "n_rejected": int(rejected.sum()),
        "n_accepted": int((~rejected).sum()),
        "confidence_mean": float(max_conf.mean()),
        "confidence_median": float(np.median(max_conf)),
        "confidence_std": float(max_conf.std()),
        "predicted_class_distribution_all": pred_class_counts,
        "predicted_class_distribution_accepted": pred_class_counts_accepted,
        "seed": GLOBAL_SEED,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em {result_path}")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(max_conf, bins=30, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=f"τ = {tau}")
    ax.set_xlabel("Confiança máxima da predição (max softmax)", fontsize=11)
    ax.set_ylabel("Número de repositórios", fontsize=11)
    ax.legend()
    plt.tight_layout()
    fig.savefig(figure_path, dpi=300, format="tiff")
    plt.close(fig)
    print(f"Figura salva em {figure_path}")

    print(f"\nTaxa de rejeição (tau={tau}): {rejection_rate:.4f} "
          f"({result['n_rejected']}/{result['n_ood']})")
    print(f"Confiança média: {result['confidence_mean']:.4f} "
          f"(mediana: {result['confidence_median']:.4f})")
    print(f"Distribuição de predições (todos): {pred_class_counts}")
    print(f"Distribuição de predições (aceitos, conf >= tau): {pred_class_counts_accepted}")

    return result


if __name__ == "__main__":
    main()
