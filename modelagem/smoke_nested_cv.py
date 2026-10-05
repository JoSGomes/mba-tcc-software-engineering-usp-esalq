"""Smoke test para run_nested_cv — executa em ~1 min sem tocar nos dados reais.

Verifica:
  - imports e dependencias
  - StratifiedKFold externo e interno
  - pipeline SMOTE + GridSearchCV
  - escrita do CSV de scores por fold
  - escrita do resumo (media +/- dp) dos 3 experimentos
  - schema das colunas do CSV

Uso:
    python modelagem/smoke_nested_cv.py
"""
import hashlib
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.seeds import GLOBAL_SEED

SMOKE_CV_PATH       = Path("experiments/smoke_nested_cv_scores.csv")
SMOKE_SUMMARY_PATH  = Path("experiments/smoke_nested_cv_summary.csv")

# Arquivos de producao que o smoke test NUNCA pode alterar — checados por
# hash de conteudo antes/depois da execucao.
PROD_FILES = [
    Path("experiments/nested_cv_scores.csv"),
    Path("experiments/nested_cv_summary.csv"),
]


def _hash_files(paths: list[Path]) -> dict[Path, str | None]:
    hashes: dict[Path, str | None] = {}
    for p in paths:
        if p.exists():
            hashes[p] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            hashes[p] = None
    return hashes

EXPECTED_FOLD_COLS = {
    "fold",
    "f1_macro_baseline",   "accuracy_baseline",
    "f1_application_baseline", "f1_helper_baseline", "f1_extender_baseline",
    "f1_macro_gridsearch", "accuracy_gridsearch",
    "f1_application_gridsearch", "f1_helper_gridsearch", "f1_extender_gridsearch",
    "best_params_gridsearch", "best_cv_score_gridsearch",
    "f1_macro_smote",      "accuracy_smote",
    "f1_application_smote",    "f1_helper_smote",    "f1_extender_smote",
    "best_params_smote", "best_cv_score_smote",
}

EXPECTED_SUMMARY_COLS = {"modelo", "f1_macro_mean", "f1_macro_std", "f1_extender_mean"}

N_OUTER   = 4   # producao usa 10
N_INNER   = 2   # producao usa 5
N_SAMPLES = 120 # 40 por classe (extender minoria: 40 amostras — SMOTE precisa >= k+1=6)

# Grid reduzido: 4 combinacoes (producao usa 54)
PARAM_GRID_SMOKE = {
    "n_estimators":    [10, 20],
    "max_depth":       [3, 5],
    "min_samples_leaf": [1],
    "class_weight":    [None],
}

# ── Dados sinteticos com separabilidade parcial ────────────────────────────────
rng = np.random.default_rng(GLOBAL_SEED)
X = rng.standard_normal((N_SAMPLES, 100))
y = np.repeat([0, 1, 2], N_SAMPLES // 3)
X[y == 0, :15]  += 1.5   # application separavel
X[y == 1, 15:30] += 1.5  # helper separavel
# extender (y==2) sem sinal extra — simula a dificuldade real


def _check(step: str, cond: bool, detail: str = "") -> bool:
    status = "PASSOU" if cond else "FALHOU"
    msg = f"  [{status}] {step}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return cond


def main() -> None:
    print("=" * 60)
    print("Smoke test: nested CV pipeline")
    print(f"  n_outer={N_OUTER}  n_inner={N_INNER}  n_samples={N_SAMPLES}")
    print(f"  param_grid: {PARAM_GRID_SMOKE}")
    print("=" * 60)

    all_ok = True

    # Hash dos arquivos de producao ANTES de rodar — comparado no check 10
    prod_hashes_before = _hash_files(PROD_FILES)

    # ── 1. Imports ─────────────────────────────────────────────────────────────
    try:
        from modelagem.baseline import run_nested_cv
        ok = _check("import run_nested_cv", True)
    except Exception as e:
        ok = _check("import run_nested_cv", False, str(e))
    all_ok &= ok

    if not ok:
        print("\nImport falhou — abortando.")
        sys.exit(1)

    from modelagem.baseline import run_nested_cv  # noqa: F811

    # ── 2. Execucao do nested CV ───────────────────────────────────────────────
    try:
        run_nested_cv(
            X, y,
            n_outer=N_OUTER,
            n_inner=N_INNER,
            param_grid=PARAM_GRID_SMOKE,
            cv_scores_path=SMOKE_CV_PATH,
            summary_path=SMOKE_SUMMARY_PATH,
        )
        ok = _check("run_nested_cv executou sem excecao", True)
    except Exception:
        ok = _check("run_nested_cv executou sem excecao", False)
        traceback.print_exc()
    all_ok &= ok

    # ── 3. CSV de scores existe e tem linhas ───────────────────────────────────
    import pandas as pd
    try:
        df = pd.read_csv(SMOKE_CV_PATH)
        ok = _check("CSV scores existe", True,
                    f"{len(df)} linhas, {len(df.columns)} colunas")
    except Exception as e:
        ok = _check("CSV scores existe", False, str(e))
        all_ok &= ok
        _check_final(all_ok)
        return
    all_ok &= ok

    # ── 4. Numero correto de folds ─────────────────────────────────────────────
    ok = _check(f"CSV scores tem {N_OUTER} linhas", len(df) == N_OUTER,
                f"encontrado: {len(df)}")
    all_ok &= ok

    # ── 5. Colunas do CSV de scores ────────────────────────────────────────────
    missing_cols = EXPECTED_FOLD_COLS - set(df.columns)
    ok = _check("CSV scores tem todas as colunas esperadas",
                len(missing_cols) == 0,
                f"faltando: {missing_cols}" if missing_cols else "")
    all_ok &= ok

    # ── 6. Sem NaN nos scores F1 ───────────────────────────────────────────────
    f1_cols = [c for c in df.columns if c.startswith("f1_") or c.startswith("accuracy_")]
    nan_count = df[f1_cols].isna().sum().sum()
    ok = _check("Sem NaN nos scores numericos", nan_count == 0,
                f"{nan_count} NaN encontrado(s)")
    all_ok &= ok

    # ── 7. Scores no intervalo [0, 1] ─────────────────────────────────────────
    out_range = ((df[f1_cols] < 0) | (df[f1_cols] > 1)).sum().sum()
    ok = _check("Scores numericos no intervalo [0,1]", out_range == 0,
                f"{out_range} valor(es) fora do intervalo")
    all_ok &= ok

    # ── 8. Resumo (media +/- dp) existe e tem os 3 experimentos ──────────────
    try:
        ds = pd.read_csv(SMOKE_SUMMARY_PATH)
        ok = _check("CSV resumo existe", True, f"{len(ds)} linhas")
    except Exception as e:
        ok = _check("CSV resumo existe", False, str(e))
        all_ok &= ok
        _check_final(all_ok)
        return
    all_ok &= ok

    ok = _check("CSV resumo tem os 3 experimentos", len(ds) == 3,
                f"encontrado: {len(ds)}")
    all_ok &= ok

    missing_s = EXPECTED_SUMMARY_COLS - set(ds.columns)
    ok = _check("CSV resumo tem as colunas esperadas",
                len(missing_s) == 0,
                f"faltando: {missing_s}" if missing_s else "")
    all_ok &= ok

    # ── 10. Verifica que smoke nao alterou os arquivos de producao (por hash) ──
    prod_hashes_after = _hash_files(PROD_FILES)
    changed = [str(p) for p in PROD_FILES if prod_hashes_before[p] != prod_hashes_after[p]]
    ok = _check("Arquivos de producao intactos (hash antes == depois)",
                len(changed) == 0,
                f"alterado(s): {changed}" if changed else "")
    all_ok &= ok

    _check_final(all_ok)

    # Limpa arquivos de smoke
    for f in [SMOKE_CV_PATH, SMOKE_SUMMARY_PATH]:
        f.unlink(missing_ok=True)
    print("  Arquivos de smoke removidos.")


def _check_final(all_ok: bool) -> None:
    print("=" * 60)
    if all_ok:
        print("RESULTADO: TODOS OS CHECKS PASSARAM — pode executar a producao")
    else:
        print("RESULTADO: UM OU MAIS CHECKS FALHARAM — corrigir antes de executar")
    print("=" * 60)


if __name__ == "__main__":
    main()
