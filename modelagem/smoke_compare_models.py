"""Smoke test para modelagem/compare_models.py (etapa de selecao) — executa em ~1 min
sem tocar nos dados/registry de producao.

Verifica:
  - imports e dependencias (xgboost, imblearn)
  - run_phase_a roda as 9 combinacoes (3 embeddings x 3 classificadores) em
    dados sinteticos, registry/figures/data temporarios
  - idempotencia: rodar de novo nao duplica linhas ja registradas
  - select_best_per_embedding retorna 1 linha por embedding
  - experiments/registry.csv de producao permanece intacto (hash antes/depois)

Uso:
    python modelagem/smoke_compare_models.py
"""
import hashlib
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.seeds import GLOBAL_SEED

SMOKE_DIR         = Path("experiments/_smoke_compare_models")
SMOKE_DATA_DIR    = SMOKE_DIR / "data"
SMOKE_FIGURES_DIR = SMOKE_DIR / "figures"
SMOKE_REGISTRY_PATH = SMOKE_DIR / "registry.csv"

PROD_REGISTRY_PATH = Path("experiments/registry.csv")

REGISTRY_COLUMNS = [
    "experiment_id", "phase", "model", "features", "seed", "params", "param_grid",
    "val_accuracy", "val_f1_macro", "val_f1_application", "val_f1_helper", "val_f1_extender",
    "test_accuracy", "test_f1_macro", "test_f1_application", "test_f1_helper", "test_f1_extender",
    "notes", "timestamp",
]

N_SAMPLES = 90  # 30 por classe — minimo confortavel para SMOTE (k_neighbors=5 default)

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


def _hash_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _make_synthetic_split(rng, n: int, dim: int, split_name: str, feature: str, out_dir: Path) -> None:
    X = rng.standard_normal((n, dim)).astype("float32")
    y = np.repeat([0, 1, 2], n // 3)
    X[y == 0, :max(1, dim // 10)] += 1.5  # application separavel
    X[y == 1, max(1, dim // 10):max(2, dim // 5)] += 1.5  # helper separavel
    # extender (y==2) sem sinal extra — simula a dificuldade real (mesmo padrao de smoke_nested_cv.py)
    np.save(out_dir / f"X_{feature}_{split_name}.npy", X)
    y_path = out_dir / f"y_{split_name}.npy"
    if not y_path.exists():
        np.save(y_path, y)


def _cleanup() -> None:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)


def main() -> None:
    print("=" * 60)
    print("Smoke test: compare_models.py (etapa de selecao)")
    print(f"  n_samples={N_SAMPLES}  param_grids: {PARAM_GRIDS_SMOKE}")
    print("=" * 60)

    all_ok = True
    prod_hash_before = _hash_file(PROD_REGISTRY_PATH)

    _cleanup()
    SMOKE_DATA_DIR.mkdir(parents=True)
    SMOKE_FIGURES_DIR.mkdir(parents=True)
    pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(SMOKE_REGISTRY_PATH, index=False)

    # ── 1. Imports ─────────────────────────────────────────────────────────────
    try:
        from modelagem.compare_models import FEATURE_DIMS, run_phase_a, select_best_per_embedding
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

    # ── 3. run_phase_a executa sem excecao ──────────────────────────────────────
    try:
        results = run_phase_a(
            data_dir=SMOKE_DATA_DIR,
            registry_path=SMOKE_REGISTRY_PATH,
            figures_dir=SMOKE_FIGURES_DIR,
            cv=2,
            param_grids=PARAM_GRIDS_SMOKE,
        )
        ok = _check("run_phase_a executou sem excecao", True, f"{len(results)} combinacoes novas")
    except Exception:
        ok = _check("run_phase_a executou sem excecao", False)
        traceback.print_exc()
        ok = False
    all_ok &= ok

    if not ok:
        _cleanup()
        _finish(all_ok)
        return

    # ── 4. registry tem 9 linhas (3 embeddings x 3 classificadores) ────────────
    df = pd.read_csv(SMOKE_REGISTRY_PATH)
    ok = _check("registry tem 9 linhas", len(df) == 9, f"encontrado: {len(df)}")
    all_ok &= ok

    # ── 5. sem NaN nas metricas numericas ───────────────────────────────────────
    metric_cols = [c for c in df.columns if "f1" in c or "accuracy" in c]
    nan_count = df[metric_cols].isna().sum().sum()
    ok = _check("sem NaN nas metricas", nan_count == 0, f"{nan_count} NaN encontrado(s)")
    all_ok &= ok

    # ── 6. idempotencia: rodar de novo nao duplica linhas ja registradas ───────
    run_phase_a(
        data_dir=SMOKE_DATA_DIR, registry_path=SMOKE_REGISTRY_PATH,
        figures_dir=SMOKE_FIGURES_DIR, cv=2, param_grids=PARAM_GRIDS_SMOKE,
    )
    df2 = pd.read_csv(SMOKE_REGISTRY_PATH)
    ok = _check("2a execucao e idempotente (nao duplica)", len(df2) == 9, f"encontrado: {len(df2)}")
    all_ok &= ok

    # ── 7. select_best_per_embedding retorna 1 linha por embedding ─────────────
    try:
        best = select_best_per_embedding(registry_path=SMOKE_REGISTRY_PATH, features=list(FEATURE_DIMS))
        ok = _check("select_best_per_embedding retorna 3 linhas", len(best) == 3, f"encontrado: {len(best)}")
    except Exception as e:
        ok = _check("select_best_per_embedding retorna 3 linhas", False, str(e))
    all_ok &= ok

    # ── 8. Verifica que smoke nao alterou o registry.csv de producao (hash) ────
    prod_hash_after = _hash_file(PROD_REGISTRY_PATH)
    ok = _check("registry.csv de producao intacto (hash antes == depois)",
                prod_hash_before == prod_hash_after)
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
