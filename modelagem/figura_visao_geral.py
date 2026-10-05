"""Diagrama de visao geral do trabalho (Apendice A do TCC): etapas executadas
e sua correspondencia com as fases do CRISP-DM.

Mesmo estilo de modelagem/figura_fluxo_nested_cv.py: sem titulo interno (a
legenda fica no documento), Arial, 300 dpi, TIFF com LZW.
Saida: experiments/figures/visao_geral_trabalho.{tiff,png}

Uso:
    python modelagem/figura_visao_geral.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("experiments/figures")
OUT_NAME = "visao_geral_trabalho"

plt.rcParams["font.family"] = "Arial"

FILL_PHASE = "#e6e6e6"   # faixa da fase CRISP-DM
FILL_DATA = "#dbe7f3"    # dados
FILL_STEP = "#f2f2f2"    # processamento
FILL_OUT = "#dff0d8"     # avaliacao / resultado
FILL_OOD = "#f6d9c4"     # repositorios fora das tres classes
EDGE = "#333333"
FS = 8.3


def box(ax, x, y, w, h, text, fill=FILL_STEP, fs=FS, weight="normal", ls="-", pad=0.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
        linewidth=0.9, edgecolor=EDGE, facecolor=fill, linestyle=ls,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, connection="arc3"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color=EDGE, connectionstyle=connection,
    ))


def phase(ax, y, h, label):
    box(ax, 1, y, 17, h, label, fill=FILL_PHASE, weight="bold", fs=FS)


def draw(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 142)
    ax.axis("off")

    # 1. Compreensao do negocio
    phase(ax, 127, 13, "Compreensão\ndo negócio")
    box(ax, 22, 128, 76, 11,
        "Problema: identificar repositórios Open Source reutilizáveis\n"
        "Taxonomia funcional: application, helper e extender")
    arrow(ax, 60, 128, 60, 124.5)

    # 2. Compreensao dos dados
    phase(ax, 93, 31, "Compreensão\ndos dados")
    box(ax, 22, 112, 34, 12, "Coleta no SEART-GHS\n15.000 repositórios\n(mínimo de 10 estrelas)", fill=FILL_DATA)
    box(ax, 64, 112, 34, 12, "Conteúdo do README\npela API do GitHub\n(14.964 repositórios)", fill=FILL_DATA)
    arrow(ax, 56, 118, 64, 118)
    arrow(ax, 81, 112, 81, 108.5)
    box(ax, 64, 94, 34, 14.5, "Rotulagem automática\ncom Claude Haiku\n(application, helper, extender,\nother e incerto)")
    box(ax, 22, 94, 34, 14.5, "Validação manual de\n100 repositórios\n(kappa de Cohen = 0,8039)", fill=FILL_OUT)
    arrow(ax, 64, 101.25, 56, 101.25)
    arrow(ax, 81, 94, 81, 89.5)

    # 3. Preparacao dos dados
    phase(ax, 53, 37, "Preparação\ndos dados")
    box(ax, 22, 80, 76, 9.5, "Limpeza, deduplicação e separação por classe")
    arrow(ax, 45, 80, 45, 76.5)
    arrow(ax, 85, 80, 85, 76.5)
    box(ax, 22, 66, 46, 10.5, "12.690 repositórios supervisionados\n(application, helper e extender)\ndivisão estratificada 70/15/15", fill=FILL_DATA)
    box(ax, 72, 66, 26, 10.5, "2.310 repositórios\nother e incerto", fill=FILL_OOD)
    arrow(ax, 45, 66, 45, 62.5)
    box(ax, 22, 53.5, 46, 9, "Representações textuais: Doc2Vec,\nSBERT e CodeBERT (nome + README)")
    arrow(ax, 45, 53.5, 45, 50)

    # 4. Modelagem
    phase(ax, 32, 19, "Modelagem")
    box(ax, 22, 32, 46, 18,
        "RF sobre o Doc2Vec: parâmetros\npadrão, Grid Search e SMOTE\ncom Grid Search\n\n"
        "Três representações combinadas\na RF, Regressão Logística e\nXGBoost, com SMOTE", fs=FS - 0.3)
    arrow(ax, 45, 32, 45, 28.5)

    # 5. Avaliacao
    phase(ax, 9, 20, "Avaliação")
    box(ax, 22, 12, 46, 16.5,
        "Validação cruzada aninhada\n(10 × 5), matrizes de confusão\n"
        "e teste t com reamostragem\ncorrigido", fill=FILL_OUT)
    box(ax, 72, 12, 26, 16.5, "Aplicação do modelo\nfinal (SBERT com\nXGBoost) aos\nrepositórios other\ne incerto", fill=FILL_OUT, fs=FS - 0.3)
    arrow(ax, 85, 66, 85, 28.5)          # other/incerto seguem direto para a avaliacao
    arrow(ax, 68, 20.25, 72, 20.25)      # modelo final

    # 6. Implantacao
    phase(ax, 0.5, 6.5, "Implantação")
    box(ax, 22, 0.5, 76, 6.5, "Não realizada: o trabalho tem caráter experimental", fill="white", ls="--")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 8.6))  # ~16 cm de largura (area util A4)
    draw(ax)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("tiff", "png"):
        out = OUT_DIR / f"{OUT_NAME}.{ext}"
        fig.savefig(out, dpi=300, format=ext,
                    pil_kwargs={"compression": "tiff_lzw"} if ext == "tiff" else None)
        print(f"Figura salva: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
