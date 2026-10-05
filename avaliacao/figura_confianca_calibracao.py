"""Figura 6 do TCC: distribuição da maior probabilidade de predição nas três
classes conhecidas (probabilidades das rodadas externas do nested CV, SBERT)
e nas classes excluídas do treino (a), e curva de confiabilidade nas classes
conhecidas (b).

Entradas:
  - experiments/nested_cv_full_selection_oof_probas.csv
  - experiments/ood_confidences.csv
  - experiments/revisao_manual_result.json   (faixas do ECE)

Saída: experiments/figures/confianca_calibracao.{tiff,png} (barras, 7 faixas; a variante
em degraus de 28 fatias sai com --degraus em confianca_calibracao_degraus.*)

Uso: python avaliacao/figura_confianca_calibracao.py [--degraus]
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
FIG_DIR = EXP_DIR / "figures"
OUT = FIG_DIR / "confianca_calibracao"
FEATURE = "sbert"
TAU = 0.5

KNOWN = "#2a78d6"     # classes conhecidas
EXCLUDED = "#eb6834"  # classes excluídas do treino
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e2"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

comma = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
integer = FuncFormatter(lambda v, _: f"{v:.0f}")


def main(bars: bool = True) -> None:
    out = OUT if bars else OUT.with_name(OUT.name + "_degraus")
    oof = pd.read_csv(EXP_DIR / "nested_cv_full_selection_oof_probas.csv")
    oof = oof[oof.feature == FEATURE]
    conf_known = oof[["p_application", "p_helper", "p_extender"]].max(axis=1).to_numpy()
    conf_excl = pd.read_csv(EXP_DIR / "ood_confidences.csv").ood_confidence.to_numpy()
    calib = json.loads((EXP_DIR / "revisao_manual_result.json").read_text(encoding="utf-8"))["calibracao_oof"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) distribuições, em % de cada grupo (os grupos têm tamanhos diferentes)
    groups = (
        (conf_known, KNOWN, "-", None, f"Classes conhecidas (n = {len(conf_known):,})".replace(",", ".")),
        (conf_excl, EXCLUDED, "--", "////", f"Classes excluídas do treino (n = {len(conf_excl):,})".replace(",", ".")),
    )
    if bars:
        edges = np.linspace(0.3, 1.0, 8)
        width = (edges[1] - edges[0]) / 2 - 0.004
        centers = (edges[:-1] + edges[1:]) / 2
        for k, (conf, color, _, hatch, label) in enumerate(groups):
            h, _ = np.histogram(conf, bins=edges)
            pct = 100 * h / len(conf)
            x = centers + (k - 0.5) * (width + 0.004)
            ax_a.bar(x, pct, width=width, color=color, hatch=hatch, edgecolor="white", linewidth=0, label=label)
            for xi, v in zip(x, pct):
                ax_a.text(xi, v + 0.4, f"{v:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=6.5, color=INK)
        ax_a.set_xticks(edges)
        ax_a.set_ylim(0, 30)
    else:
        edges = np.linspace(0.3, 1.0, 29)
        for conf, color, ls, _, label in groups:
            h, _ = np.histogram(conf, bins=edges)
            ax_a.stairs(100 * h / len(conf), edges, color=color, linewidth=2, linestyle=ls, label=label)
    ax_a.axvline(TAU, color=INK, linewidth=1, linestyle=":")
    ax_a.text(TAU - 0.01, ax_a.get_ylim()[1] * 0.97, "τ = 0,5", ha="right", va="top", color=INK)
    ax_a.set_xlabel("Maior probabilidade de predição")
    ax_a.set_ylabel("Repositórios do grupo (%)")
    ax_a.set_xlim(0.3, 1.0)
    ax_a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.2), fontsize=8, handlelength=2.5)
    ax_a.set_title("(a)", loc="left", fontsize=9, color=INK, pad=24)

    # (b) curva de confiabilidade nas classes conhecidas
    bins = [b for b in calib["bins"] if b["n"] > 0]
    x = [b["confianca_media"] for b in bins]
    y = [b["acuracia"] for b in bins]
    ax_b.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, linewidth=1, linestyle=":", label="Calibração perfeita")
    ax_b.plot(x, y, color=KNOWN, linewidth=2, marker="o", markersize=5,
              markeredgecolor="white", markeredgewidth=1, label="Classes conhecidas")
    ax_b.text(0.97, 0.33, f"ECE = {calib['ece']:.3f}".replace(".", ","), ha="right", va="bottom", color=INK)
    ax_b.set_xlabel("Confiança média da faixa")
    ax_b.set_ylabel("Proporção de acertos")
    ax_b.set_xlim(0.3, 1.0)
    ax_b.set_ylim(0.3, 1.0)
    ax_b.set_aspect("equal")
    ax_b.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.2), fontsize=8, handlelength=2.5)
    ax_b.set_title("(b)", loc="left", fontsize=9, color=INK, pad=24)

    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(comma)
    ax_a.yaxis.set_major_formatter(integer)
    ax_b.yaxis.set_major_formatter(comma)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".tiff"), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Gravado: {out}.tiff / .png")


if __name__ == "__main__":
    import sys
    main(bars="--degraus" not in sys.argv)
