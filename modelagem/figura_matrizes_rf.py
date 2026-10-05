"""Figuras 2 a 4 do TCC: matrizes de confusão dos três experimentos do RF
sobre o Doc2Vec no conjunto de teste, em formato compacto (8 cm de largura),
com o mesmo estilo da Figura 5 (modelagem/figura_matrizes_confusao.py).

Retreina cada RF com os hiperparâmetros registrados em
experiments/registry.csv (mesma construção de modelagem/baseline.py, mesma
semente) e confere se o F1 por classe reproduz o registrado antes de
desenhar.

Saída: experiments/figures/cm_rf_{d2ac1ac9,018ed2b3,50fa0ee9}_compacta.{tiff,png}

Uso: python modelagem/figura_matrizes_rf.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import LABEL_CLASSES, REGISTRY_PATH, load_split

FIG_DIR = Path("experiments/figures")
EXPERIMENTS = ["d2ac1ac9", "018ed2b3", "50fa0ee9"]
INK = "#2b2b2b"
MUTED = "#6b6b6b"

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def build_model(row: pd.Series):
    params = json.loads(row["params"])
    if row["model"] == "RandomForest+SMOTE":
        return ImbPipeline([
            ("smote", SMOTE(random_state=GLOBAL_SEED)),
            ("rf", RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=1, **params)),
        ])
    return RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1, **params)


def draw(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.15, 2.55))
    scale = cm / cm.max()
    ax.imshow(scale, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center", fontsize=8.5,
                    color="white" if scale[i, j] > 0.55 else INK)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_CLASSES, fontsize=7.5)
    ax.set_yticklabels(LABEL_CLASSES, fontsize=7.5, rotation=90, va="center")
    ax.set_xlabel("Classe predita", fontsize=8, color=MUTED)
    ax.set_ylabel("Classe verdadeira", fontsize=8, color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    tight = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(out.with_suffix(".tiff"), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, **tight)
    fig.savefig(out.with_suffix(".png"), dpi=200, **tight)
    plt.close(fig)


def main() -> None:
    reg = pd.read_csv(REGISTRY_PATH).set_index("experiment_id")
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test_indist")
    for exp in EXPERIMENTS:
        row = reg.loc[exp]
        model = build_model(row)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        expected = [row["test_f1_application"], row["test_f1_helper"], row["test_f1_extender"]]
        if not np.allclose(f1, expected, atol=1e-4):
            raise ValueError(f"{exp}: F1 {f1.round(4)} != registrado {np.round(expected, 4)}")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        out = FIG_DIR / f"cm_rf_{exp}_compacta"
        draw(cm, out)
        print(f"{exp} ({row['notes']}): F1 confere; matriz {cm.tolist()} -> {out}.tiff")


if __name__ == "__main__":
    main()
