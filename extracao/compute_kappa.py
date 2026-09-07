"""Compute Cohen's kappa between Haiku labels and manual review.

Workflow:
1. Run gold_standard.py to get Haiku labels
2. Manually review uncertain cases (confidence < threshold) — fill manual_label column
3. Run this script to compute kappa on the reviewed subset

Usage:
    python extracao/compute_kappa.py --gold data/raw/gold_standard.csv
    python extracao/compute_kappa.py --gold data/raw/gold_standard.csv --review data/raw/manual_review.csv
    python extracao/compute_kappa.py --gold data/raw/gold_standard.csv --confidence-threshold 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED  # noqa: F401

CLASSES = ["application", "helper", "extender", "other"]
FIGURES_DIR = Path("experiments/figures")


def load_gold(gold_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_path)
    required = {"repo_id", "label", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gold_standard.csv missing columns: {missing}")
    return df


def load_manual_review(review_path: Path) -> pd.DataFrame:
    """Load manual review CSV with columns: repo_id, manual_label."""
    df = pd.read_csv(review_path)
    required = {"repo_id", "manual_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual_review.csv missing columns: {missing}")
    return df


def export_uncertain_for_review(gold_df: pd.DataFrame, threshold: float, output: Path) -> None:
    """Export uncertain / low-confidence cases for manual review."""
    uncertain = gold_df[
        (gold_df["label"] == "incerto") | (gold_df["confidence"] < threshold)
    ].copy()
    uncertain["manual_label"] = ""  # empty column for human to fill
    uncertain[["repo_id", "name", "label", "confidence", "reasoning", "manual_label"]].to_csv(output, index=False)
    print(f"Exported {len(uncertain)} cases for manual review to {output}")
    print(f"Fill the 'manual_label' column with: application / helper / extender / other")
    print(f"Then re-run with --review {output}")


def compute_kappa_report(haiku_labels: list[str], manual_labels: list[str]) -> dict:
    kappa = cohen_kappa_score(manual_labels, haiku_labels, labels=CLASSES)
    report = classification_report(
        manual_labels, haiku_labels,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(manual_labels, haiku_labels, labels=CLASSES)
    return {"kappa": kappa, "classification_report": report, "confusion_matrix": cm.tolist()}


def save_confusion_figure(cm: list[list[int]], n_reviewed: int, out_dir: Path = FIGURES_DIR) -> Path:
    """Save the Haiku-vs-manual confusion matrix as a heatmap (mesmo padrão de modelagem/baseline.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Haiku", fontsize=11)
    ax.set_ylabel("Revisão manual", fontsize=11)
    ax.set_title(f"Kappa — Haiku vs revisão manual (n={n_reviewed})", fontsize=12)
    plt.tight_layout()
    out = out_dir / f"cm_kappa_{n_reviewed}repos.tiff"
    fig.savefig(out, dpi=300, format="tiff")
    plt.close(fig)
    return out


def print_report(results: dict, n_reviewed: int, n_total: int) -> None:
    kappa = results["kappa"]
    report = results["classification_report"]
    cm = results["confusion_matrix"]

    kappa_interp = (
        "Excelente (> 0.8)" if kappa > 0.8
        else "Bom (0.6–0.8)" if kappa > 0.6
        else "Moderado (0.4–0.6)" if kappa > 0.4
        else "Fraco (< 0.4)"
    )

    print("\n" + "=" * 60)
    print("COHEN'S KAPPA — Haiku vs Revisão Manual")
    print("=" * 60)
    print(f"Casos revisados:  {n_reviewed} / {n_total} ({n_reviewed/n_total*100:.1f}%)")
    print(f"Kappa:            {kappa:.4f} — {kappa_interp}")
    print()
    print("Acurácia por classe (haiku vs manual):")
    for cls in CLASSES:
        if cls in report:
            f1 = report[cls].get("f1-score", 0)
            prec = report[cls].get("precision", 0)
            rec = report[cls].get("recall", 0)
            print(f"  {cls:15s}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}")

    print()
    print("Matriz de confusão (linhas=manual, colunas=haiku):")
    print(f"              {'  '.join(f'{c[:4]:>6}' for c in CLASSES)}")
    for i, cls in enumerate(CLASSES):
        row = cm[i]
        print(f"  {cls[:15]:15s}  {'  '.join(f'{v:>6}' for v in row)}")

    print()
    if kappa < 0.6:
        print("ATENCAO: Kappa < 0.6: considerar revisar o prompt do Haiku ou expandir a amostra de revisao.")
    else:
        print("OK: Kappa aceitavel para uso como rotulos semi-automaticos.")

    print()
    print("Próximo passo: registrar kappa em experiments/registry.csv e prosseguir para a preparação dos dados.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa between Haiku and manual labels")
    parser.add_argument("--gold", type=Path, default=Path("data/raw/gold_standard.csv"))
    parser.add_argument("--review", type=Path, default=None, help="Manual review CSV with manual_label column")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Flag cases below this confidence for review (default: 0.7)")
    parser.add_argument("--export-uncertain", type=Path, default=Path("data/raw/uncertain_for_review.csv"), help="Export uncertain cases to this path")
    args = parser.parse_args()

    gold_df = load_gold(args.gold)
    n_total = len(gold_df)
    print(f"Loaded {n_total} labeled repos from {args.gold}")
    print(f"Label distribution:\n{gold_df['label'].value_counts().to_string()}")

    if args.review is None:
        # Export uncertain cases for manual review
        export_uncertain_for_review(gold_df, args.confidence_threshold, args.export_uncertain)
        print(f"\nNo manual review file provided.")
        print(f"After filling {args.export_uncertain}, run:")
        print(f"  python extracao/compute_kappa.py --gold {args.gold} --review {args.export_uncertain}")
        return

    review_df = load_manual_review(args.review)
    merged = gold_df.merge(review_df[["repo_id", "manual_label"]], on="repo_id", how="inner")
    merged = merged[
        merged["manual_label"].isin(CLASSES) & merged["label"].isin(CLASSES)
    ]

    if len(merged) < 20:
        print(f"WARNING: Only {len(merged)} matching reviewed cases — kappa may be unreliable (recommend >= 50).")

    results = compute_kappa_report(
        haiku_labels=merged["label"].tolist(),
        manual_labels=merged["manual_label"].tolist(),
    )

    print_report(results, n_reviewed=len(merged), n_total=n_total)

    # Save kappa result
    kappa_output = Path("data/raw/kappa_result.json")
    import json
    kappa_output.write_text(json.dumps({
        "kappa": results["kappa"],
        "n_reviewed": len(merged),
        "n_total": n_total,
        "classes": CLASSES,
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"],
    }, indent=2))
    print(f"\nKappa result saved to {kappa_output}")

    figure_path = save_confusion_figure(results["confusion_matrix"], n_reviewed=len(merged))
    print(f"Confusion matrix figure saved to {figure_path}")


if __name__ == "__main__":
    main()
