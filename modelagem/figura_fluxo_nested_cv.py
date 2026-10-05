"""Diagrama de blocos do procedimento de avaliacao (nested CV com selecao
completa dentro do fold) para a Metodologia do TCC.

Mostra, para o leitor, onde cada decisao acontece: divisao externa em 10
partes, ajuste do Doc2Vec so no treino de cada rodada, laco interno de 5
partes com Grid Search (SMOTE dentro do pipeline) entre RF, Regressao
Logistica e XGBoost, reajuste do vencedor e avaliacao unica na parte de
teste, e agregacao das 10 rodadas.

Sem titulo interno (a legenda fica no documento, norma USP/Esalq), Arial,
300 dpi, com seta de retorno "proxima rodada" e as tres representacoes unidas
num traco antes do laco interno (versao B, escolhida entre tres rascunhos;
as paletas continuam configuraveis em PALETTES/VERSIONS).
Saida: experiments/figures/fluxo_nested_cv.{tiff,png}

Uso:
    python modelagem/figura_fluxo_nested_cv.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path("experiments/figures")
OUT_NAME = "fluxo_nested_cv"

plt.rcParams["font.family"] = "Arial"

PALETTES = {
    "cores": {"data": "#dbe7f3", "step": "#f2f2f2", "test": "#f6d9c4", "out": "#dff0d8"},
    "cinza": {"data": "#e8e8e8", "step": "#f7f7f7", "test": "#c8c8c8", "out": "#dcdcdc"},
}
VERSIONS = {
    "A": {"palette": "cores", "loop": False, "merge": False},
    "B": {"palette": "cores", "loop": True, "merge": True},
    "C": {"palette": "cinza", "loop": True, "merge": True},
}
FINAL_VERSION = "B"
FILL_DATA, FILL_STEP, FILL_TEST, FILL_OUT = (PALETTES["cores"][k] for k in ("data", "step", "test", "out"))
EDGE = "#333333"
FS = 8.5                 # tamanho de fonte padrao


def box(ax, x, y, w, h, text, fill=FILL_STEP, fs=FS, weight="normal", ls="-", lw=0.9, pad=0.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
        linewidth=lw, edgecolor=EDGE, facecolor=fill, linestyle=ls,
    ))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, ls="-", connection="arc3"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color=EDGE, linestyle=ls, connectionstyle=connection,
    ))


def split_bar(ax, x, y, w, h, n, test_idx, label_test, label_train):
    """Barra dividida em n partes, com a parte `test_idx` destacada."""
    seg = w / n
    for i in range(n):
        ax.add_patch(Rectangle(
            (x + i * seg, y), seg, h, linewidth=0.7, edgecolor=EDGE,
            facecolor=FILL_TEST if i == test_idx else FILL_DATA,
            hatch="///" if i == test_idx else None,
        ))
    ax.text(x + test_idx * seg + seg / 2, y - 1.6, label_test, ha="center", va="top", fontsize=FS - 1)
    ax.text(x + w / 2, y + h + 1.2, label_train, ha="center", va="bottom", fontsize=FS - 1)


def draw(ax, loop: bool = False, merge: bool = False) -> None:
    ax.set_xlim(0, 107 if loop else 100)
    ax.set_ylim(0, 132)
    ax.axis("off")

    # 1. Conjunto supervisionado
    box(ax, 18, 122, 64, 8,
        "Conjunto supervisionado: 12.690 repositórios\n(application, helper, extender)", fill=FILL_DATA)
    arrow(ax, 50, 122, 50, 117.5)

    # 2. Divisao externa
    ax.text(50, 116.5, "Divisão externa estratificada em 10 partes", ha="center", va="top",
            fontsize=FS, weight="bold")
    split_bar(ax, 18, 104, 64, 5, 10, 3,
              "parte de teste da rodada k\n(1.269 repositórios)",
              "9 partes de treino da rodada k (11.421 repositórios)")
    arrow(ax, 50, 99, 50, 95.5)

    # 3. Rodada k (laco externo)
    box(ax, 2, 14, 96, 81, "", fill="white", ls="--", lw=1.0, pad=1.2)
    ax.text(4, 93.3, "Rodada k = 1, …, 10 (laço externo)", ha="left", va="top", fontsize=FS, weight="bold")

    # 3a. Representacoes (treino do fold)
    ax.text(38, 88.5, "Representações textuais, calculadas para cada rodada", ha="center", va="top", fontsize=FS)
    box(ax, 5, 76, 20, 9, "Doc2Vec\najustado somente no\ntreino da rodada", fill=FILL_STEP, fs=FS - 0.5)
    box(ax, 28, 76, 20, 9, "SBERT\npré-treinado,\ncongelado", fill=FILL_STEP, fs=FS - 0.5)
    box(ax, 51, 76, 20, 9, "CodeBERT\npré-treinado,\ncongelado", fill=FILL_STEP, fs=FS - 0.5)
    if merge:
        for x in (15, 38, 61):
            ax.plot([x, x], [76, 73.8], color=EDGE, linewidth=0.9)
        ax.plot([15, 61], [73.8, 73.8], color=EDGE, linewidth=0.9)
        arrow(ax, 38, 73.8, 38, 71.5)
    else:
        arrow(ax, 38, 76, 38, 71.5)

    # 3b. Laco interno
    box(ax, 5, 36, 66, 35.5, "", fill="#fbfbfb", ls=":", lw=0.9, pad=1.0)
    ax.text(7, 70, "Laço interno (somente com o treino da rodada),\nrepetido para cada representação",
            ha="left", va="top", fontsize=FS, weight="bold")
    split_bar(ax, 13, 57, 50, 4, 5, 1, "validação interna", "validação cruzada em 5 partes")
    ax.text(38, 51.5, "Grid Search com SMOTE aplicado apenas às partes de treino", ha="center",
            va="center", fontsize=FS - 0.5, style="italic")
    box(ax, 8, 41, 18, 7, "Random Forest", fs=FS - 0.5)
    box(ax, 29, 41, 18, 7, "Regressão\nLogística", fs=FS - 0.5)
    box(ax, 50, 41, 18, 7, "XGBoost", fs=FS - 0.5)
    arrow(ax, 38, 36, 38, 32.5)

    # 3c. Escolha e reajuste
    box(ax, 12, 25, 52, 7.5,
        "Escolha do classificador e dos hiperparâmetros\ncom maior F1-macro médio no laço interno", fs=FS - 0.5)
    arrow(ax, 38, 25, 38, 22.5)
    box(ax, 12, 16, 52, 6.5, "Modelo vencedor reajustado com os 11.421\nrepositórios de treino da rodada",
        fs=FS - 0.5)

    # 3d. Parte de teste: reservada ate o fim
    box(ax, 76, 45, 19, 24, "Parte de teste\nda rodada\n(1.269)\n\nnão usada em\nnenhuma etapa\nà esquerda",
        fill=FILL_TEST, fs=FS - 0.5)
    box(ax, 76, 16, 19, 13, "Avaliação única:\nF1-macro, F1 por\nclasse e\nprobabilidades", fill=FILL_OUT, fs=FS - 0.5)
    arrow(ax, 85.5, 45, 85.5, 29.5)
    arrow(ax, 64, 19.25, 76, 19.25)

    if loop:
        # sai pela lateral direita da rodada, sobe por fora e volta ao topo
        ax.plot([98, 102, 102], [22, 22, 88], color=EDGE, linewidth=0.9)
        arrow(ax, 102, 88, 98, 88)
        ax.text(104.5, 55, "próxima rodada (k + 1)", rotation=90, ha="center", va="center", fontsize=FS - 0.5)

    # 4. Agregacao
    arrow(ax, 50, 14, 50, 10.5)
    box(ax, 12, 1, 76, 9.5,
        "Agregação das 10 rodadas: média ± desvio padrão das métricas,\n"
        "matriz de confusão agregada e comparação entre as representações", fill=FILL_OUT)


def main() -> None:
    global FILL_DATA, FILL_STEP, FILL_TEST, FILL_OUT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in [(FINAL_VERSION, VERSIONS[FINAL_VERSION])]:
        pal = PALETTES[cfg["palette"]]
        FILL_DATA, FILL_STEP, FILL_TEST, FILL_OUT = pal["data"], pal["step"], pal["test"], pal["out"]
        fig, ax = plt.subplots(figsize=(6.3, 8.3))  # ~16 cm de largura (area util A4)
        draw(ax, loop=cfg["loop"], merge=cfg["merge"])
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        for ext in ("tiff", "png"):
            out = OUT_DIR / f"{OUT_NAME}.{ext}"
            fig.savefig(out, dpi=300, format=ext,
                    pil_kwargs={"compression": "tiff_lzw"} if ext == "tiff" else None)
            print(f"Figura salva: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
