"""Figura 5 do TCC: matrizes de confusão agregadas das três representações na
validação cruzada aninhada (contagem absoluta e normalizada por linha), em
tamanho de página e com vírgula decimal. Painéis identificados por letras A–F
no canto superior esquerdo, sem pontuação (manual de normas USP/Esalq,
item 15.1).

Lê os mesmos dados da figura original (`common.py::save_confusion_comparison_figure`),
que não é alterada.

Entrada: experiments/nested_cv_full_selection_confusion_matrices.json
Saída:   experiments/figures/matrizes_confusao_representacoes.{tiff,png}

Uso: python modelagem/figura_matrizes_confusao.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
SRC = EXP_DIR / "nested_cv_full_selection_confusion_matrices.json"
OUT = EXP_DIR / "figures" / "matrizes_confusao_representacoes"
NAMES = {"doc2vec": "Doc2Vec", "sbert": "SBERT", "codebert": "CodeBERT"}
INK = "#2b2b2b"

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def fmt_int(v: float) -> str:
    return f"{int(round(v)):,}".replace(",", ".")


def fmt_pct(v: float) -> str:
    return f"{100 * v:.1f}%".replace(".", ",")


def draw(ax, m, scale, fmt):
    ax.imshow(scale, cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt(m[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if scale[i, j] > 0.55 else INK)
    ax.set_xticks(range(m.shape[1]))
    ax.set_yticks(range(m.shape[0]))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    labels = d["labels"]
    keys = list(NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.6))
    for col, key in enumerate(keys):
        raw = np.array(d["raw"][key], dtype=float)
        norm = np.array(d["normalized"][key], dtype=float)
        draw(axes[0, col], raw, raw / raw.max(), fmt_int)
        draw(axes[1, col], norm, norm, fmt_pct)
        axes[0, col].set_title(NAMES[key], fontsize=9, color=INK, pad=6)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_yticklabels(labels if col == 0 else [], fontsize=7.5, rotation=90, va="center")
            ax.set_xlabel("Classe predita", fontsize=8, color=INK)
        axes[0, col].set_ylabel("")
    for ax, letter in zip(axes.flat, "ABCDEF"):
        ax.set_title(letter, loc="left", fontsize=10, color=INK, pad=6)
    axes[0, 0].set_ylabel("Classe verdadeira\n(contagem)", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("Classe verdadeira\n(% da linha)", fontsize=8, color=INK)
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".tiff"), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Gravado: {OUT}.tiff / .png")


if __name__ == "__main__":
    main()
