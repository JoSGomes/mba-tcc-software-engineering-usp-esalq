"""Smoke test para modelagem/compare_models.py (etapa de comparacao) —
executa em ~1 min sem tocar nos dados/registry/arquivos de producao.

Verifica:
  - imports e dependencias (run_nested_cv_configs em common.py)
  - run_phase_a (etapa de selecao, ja coberta por smoke_compare_models.py)
    popula um registry temporario a partir do qual select_best_per_embedding
    escolhe
  - run_phase_b roda o nested CV generalizado entre as representacoes,
    usando o classificador vencedor de cada uma
  - CSVs de scores/resumo/Wilcoxon tem o schema esperado (por config, nao
    mais por baseline/gridsearch/smote fixos)
  - matriz de confusao agregada (JSON raw+normalizada) e a figura 2xN saem
    consistentes (raw soma bate com n_outer * tamanho do fold de teste)
  - nenhum arquivo de producao (registry.csv + nested_cv_scores.csv/
    nested_cv_summary.csv/wilcoxon_results.csv do baseline Doc2Vec+RF) e
    alterado (hash antes/depois)

Uso:
    python modelagem/smoke_compare_models_nested_cv.py
"""
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.seeds import GLOBAL_SEED

SMOKE_DIR            = Path("experiments/_smoke_compare_models_phase_b")
SMOKE_DATA_DIR       = SMOKE_DIR / "data"
SMOKE_FIGURES_DIR    = SMOKE_DIR / "figures"
SMOKE_REGISTRY_PATH  = SMOKE_DIR / "registry.csv"
SMOKE_CV_PATH        = SMOKE_DIR / "nested_cv_embeddings_scores.csv"
SMOKE_SUMMARY_PATH   = SMOKE_DIR / "nested_cv_embeddings_summary.csv"
SMOKE_WILCOXON_PATH  = SMOKE_DIR / "wilcoxon_embeddings_results.csv"
SMOKE_CONFUSION_JSON = SMOKE_DIR / "nested_cv_confusion_matrices.json"
SMOKE_CONFUSION_FIG  = SMOKE_FIGURES_DIR / "cm_nested_embeddings_comparison.tiff"

# Arquivos de producao que este script NUNCA pode alterar.
PROD_FILES = [
    Path("experiments/registry.csv"),
    Path("experiments/nested_cv_scores.csv"),
    Path("experiments/nested_cv_summary.csv"),
    Path("experiments/wilcoxon_results.csv"),
    Path("experiments/nested_cv_embeddings_scores.csv"),
    Path("experiments/nested_cv_embeddings_summary.csv"),
    Path("experiments/wilcoxon_embeddings_results.csv"),
    Path("experiments/nested_cv_confusion_matrices.json"),
]

REGISTRY_COLUMNS = [
    "experiment_id", "phase", "model", "features", "seed", "params", "param_grid",
    "val_accuracy", "val_f1_macro", "val_f1_application", "val_f1_helper", "val_f1_extender",
    "test_accuracy", "test_f1_macro", "test_f1_application", "test_f1_helper", "test_f1_extender",
    "notes", "timestamp",
]

N_SAMPLES = 90  # 30 por classe — minimo confortavel para SMOTE (k_neighbors=5 default)
N_OUTER = 4     # producao usa 10
N_INNER = 2     # producao usa 5

PARAM_GRIDS_SMOKE = {
    "rf":      {"n_estimators": [10], "max_depth": [3], "min_samples_leaf": [1], "class_weight": [None]},
    "logreg":  {"C": [1.0], "max_iter": [200]},
    "xgboost": {"n_estimators": [10], "max_depth": [3], "learning_rate": [0.3]},
}


def _check(step: str, cond: bool, detail: str = "") -> bool:
    status = "PASSOU" if cond else "FALHOU"
    msg = f"  [{status}] {step}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return cond


def _hash_files(paths: list[Path]) -> dict[Path, str | None]:
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in paths}


def _make_synthetic_split(rng, n: int, dim: int, split_name: str, feature: str, out_dir: Path) -> None:
    X = rng.standard_normal((n, dim)).astype("float32")
    y = np.repeat([0, 1, 2], n // 3)
    X[y == 0, :max(1, dim // 10)] += 1.5
    X[y == 1, max(1, dim // 10):max(2, dim // 5)] += 1.5
    np.save(out_dir / f"X_{feature}_{split_name}.npy", X)
    y_path = out_dir / f"y_{split_name}.npy"
    if not y_path.exists():
        np.save(y_path, y)


def _cleanup() -> None:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)


def main() -> None:
    print("=" * 60)
    print("Smoke test: compare_models.py (etapa de comparacao)")
    print(f"  n_outer={N_OUTER}  n_inner={N_INNER}  n_samples={N_SAMPLES}")
    print("=" * 60)

    all_ok = True
    prod_hashes_before = _hash_files(PROD_FILES)

    _cleanup()
    SMOKE_DATA_DIR.mkdir(parents=True)
    SMOKE_FIGURES_DIR.mkdir(parents=True)
    pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(SMOKE_REGISTRY_PATH, index=False)

    # ── 1. Imports ─────────────────────────────────────────────────────────────
    try:
        from modelagem.compare_models import FEATURE_DIMS, run_phase_a, run_phase_b
        ok = _check("import compare_models", True)
    except Exception as e:
        ok = _check("import compare_models", False, str(e))
        all_ok &= ok

    if not ok:
        print("\nImport falhou — abortando.")
        _cleanup()
        sys.exit(1)

    # ── 2. Dados sinteticos por embedding (dims reais, poucas amostras) ────────
    rng = np.random.default_rng(GLOBAL_SEED)
    for feature, dim in FEATURE_DIMS.items():
        for split in ["train", "val", "test_indist"]:
            _make_synthetic_split(rng, N_SAMPLES, dim, split, feature, SMOKE_DATA_DIR)
    _check("dados sinteticos gerados", True, f"{len(FEATURE_DIMS)} embeddings x 3 splits")

    # ── 3. etapa de selecao (pre-requisito) roda e popula o registry temporario
    try:
        run_phase_a(
            data_dir=SMOKE_DATA_DIR, registry_path=SMOKE_REGISTRY_PATH,
            figures_dir=SMOKE_FIGURES_DIR, cv=2, param_grids=PARAM_GRIDS_SMOKE,
        )
        ok = _check("etapa de selecao (pre-requisito) executou sem excecao", True)
    except Exception:
        ok = _check("etapa de selecao (pre-requisito) executou sem excecao", False)
        traceback.print_exc()
        ok = False
    all_ok &= ok

    if not ok:
        _cleanup()
        _finish(all_ok)
        return

    # ── 4. etapa de comparacao roda sem excecao
    try:
        run_phase_b(
            data_dir=SMOKE_DATA_DIR,
            registry_path=SMOKE_REGISTRY_PATH,
            n_outer=N_OUTER, n_inner=N_INNER,
            cv_scores_path=SMOKE_CV_PATH,
            wilcoxon_path=SMOKE_WILCOXON_PATH,
            summary_path=SMOKE_SUMMARY_PATH,
            confusion_matrices_path=SMOKE_CONFUSION_JSON,
            confusion_figure_path=SMOKE_CONFUSION_FIG,
        )
        ok = _check("run_phase_b executou sem excecao", True)
    except Exception:
        ok = _check("run_phase_b executou sem excecao", False)
        traceback.print_exc()
        ok = False
    all_ok &= ok

    if not ok:
        _cleanup()
        _finish(all_ok)
        return

    # ── 5. CSV de scores por fold: N_OUTER linhas, colunas por embedding ────────
    try:
        scores_df = pd.read_csv(SMOKE_CV_PATH)
        ok = _check("CSV scores existe", True, f"{len(scores_df)} linhas, {len(scores_df.columns)} colunas")
    except Exception as e:
        ok = _check("CSV scores existe", False, str(e))
        all_ok &= ok
        _cleanup()
        _finish(all_ok)
        return
    all_ok &= ok

    ok = _check(f"CSV scores tem {N_OUTER} linhas", len(scores_df) == N_OUTER, f"encontrado: {len(scores_df)}")
    all_ok &= ok

    expected_cols = {"fold"}
    for feature in FEATURE_DIMS:
        for metric in ["f1_macro", "accuracy", "f1_application", "f1_helper", "f1_extender",
                       "best_params", "best_cv_score"]:
            expected_cols.add(f"{metric}_{feature}")
    missing = expected_cols - set(scores_df.columns)
    ok = _check("CSV scores tem colunas por embedding", len(missing) == 0,
                f"faltando: {missing}" if missing else "")
    all_ok &= ok

    f1_cols = [c for c in scores_df.columns if c.startswith("f1_") or c.startswith("accuracy_")]
    nan_count = scores_df[f1_cols].isna().sum().sum()
    ok = _check("sem NaN nos scores numericos", nan_count == 0, f"{nan_count} NaN encontrado(s)")
    all_ok &= ok

    # ── 6. Resumo e Wilcoxon (3 pares para 3 embeddings) ────────────────────────
    summary_df = pd.read_csv(SMOKE_SUMMARY_PATH)
    ok = _check("resumo tem 1 linha por embedding", len(summary_df) == len(FEATURE_DIMS),
                f"encontrado: {len(summary_df)}")
    all_ok &= ok

    wilcoxon_df = pd.read_csv(SMOKE_WILCOXON_PATH)
    ok = _check("Wilcoxon tem 3 comparacoes (3 embeddings, pares unicos)", len(wilcoxon_df) == 3,
                f"encontrado: {len(wilcoxon_df)}")
    all_ok &= ok

    p_ok = ((wilcoxon_df["p_value"] >= 0) & (wilcoxon_df["p_value"] <= 1)).all()
    ok = _check("p-values no intervalo [0,1]", bool(p_ok))
    all_ok &= ok

    # ── 7. Matriz de confusao agregada: JSON + figura ───────────────────────────
    try:
        with open(SMOKE_CONFUSION_JSON, encoding="utf-8") as f:
            cm_data = json.load(f)
        ok = _check("JSON de matrizes de confusao existe", True)
    except Exception as e:
        ok = _check("JSON de matrizes de confusao existe", False, str(e))
        cm_data = None
    all_ok &= ok

    if cm_data is not None:
        ok = _check("JSON tem raw e normalized para cada embedding",
                    set(cm_data["raw"]) == set(FEATURE_DIMS) == set(cm_data["normalized"]))
        all_ok &= ok

        # soma da matriz raw de cada embedding == numero total de exemplos
        # testados nos folds externos (== 3*N_SAMPLES: X_all da etapa de comparacao e a
        # concatenacao train+val+test_indist, cada exemplo cai em exatamente
        # um fold de teste ao longo dos N_OUTER folds)
        expected_total = 3 * N_SAMPLES
        for feature in FEATURE_DIMS:
            total = int(np.array(cm_data["raw"][feature]).sum())
            ok = _check(f"soma da matriz raw ({feature}) == 3*N_SAMPLES", total == expected_total,
                        f"encontrado: {total}, esperado: {expected_total}")
            all_ok &= ok

        # cada linha da matriz normalizada soma ~1.0 (recall por classe)
        for feature in FEATURE_DIMS:
            row_sums = np.array(cm_data["normalized"][feature]).sum(axis=1)
            ok = _check(f"linhas da matriz normalizada ({feature}) somam ~1.0",
                        bool(np.allclose(row_sums, 1.0, atol=1e-6)))
            all_ok &= ok

    ok = _check("figura comparativa de confusao existe", SMOKE_CONFUSION_FIG.exists())
    all_ok &= ok

    # ── 8. Nenhum arquivo de producao foi alterado (hash antes == depois) ──────
    prod_hashes_after = _hash_files(PROD_FILES)
    changed = [str(p) for p in PROD_FILES if prod_hashes_before[p] != prod_hashes_after[p]]
    ok = _check("arquivos de producao intactos (hash antes == depois)",
                len(changed) == 0, f"alterado(s): {changed}" if changed else "")
    all_ok &= ok

    _cleanup()
    _finish(all_ok)


def _finish(all_ok: bool) -> None:
    print("=" * 60)
    if all_ok:
        print("RESULTADO: TODOS OS CHECKS PASSARAM — pode executar a producao")
    else:
        print("RESULTADO: UM OU MAIS CHECKS FALHARAM — corrigir antes de executar")
    print("=" * 60)
    print("  Diretorio de smoke removido.")


if __name__ == "__main__":
    main()
