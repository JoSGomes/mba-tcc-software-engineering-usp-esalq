"""Smoke test para modelagem/nested_cv_full_selection.py — executa em ~1-2 min
com dados sinteticos, sem tocar nos dados/arquivos de producao.

Verifica:
  - o Doc2Vec de cada fold e ajustado SO com os textos de treino do fold
    (marcadores unicos nos textos de teste nao entram no vocabulario)
  - os folds externos sao os mesmos de common.py::run_nested_cv_configs
  - uma execucao interrompida (--max-folds) e retomada gera exatamente os
    mesmos resultados de uma execucao direta (checkpoint por fold)
  - arquivos finais com o schema esperado: scores por fold (incl. classificador
    escolhido e scores internos dos 3), probabilidades out-of-fold (cada
    repositorio aparece uma vez por embedding, linhas somam 1), contagem de
    escolhas, matrizes de confusao
  - nenhum arquivo de producao e alterado (hash antes/depois)

Uso:
    python modelagem/smoke_nested_cv_full_selection.py
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

SMOKE_DIR = Path("experiments/_smoke_nested_cv_full_selection")

PROD_FILES = [
    Path("experiments/registry.csv"),
    Path("experiments/nested_cv_scores.csv"),
    Path("experiments/nested_cv_summary.csv"),
    Path("experiments/nested_cv_full_selection_scores.csv"),
    Path("experiments/nested_cv_full_selection_summary.csv"),
    Path("experiments/nested_cv_full_selection_oof_probas.csv"),
]

N_PER_CLASS = 30
N_SAMPLES = 3 * N_PER_CLASS
N_OUTER = 3   # producao usa 10
N_INNER = 2   # producao usa 5
FROZEN_DIMS = {"sbert": 16, "codebert": 24}  # dims reduzidas: so o fluxo importa aqui

PARAM_GRIDS_SMOKE = {
    "rf":      {"n_estimators": [10], "max_depth": [3], "min_samples_leaf": [1], "class_weight": [None]},
    "logreg":  {"C": [1.0], "max_iter": [200]},
    "xgboost": {"n_estimators": [10], "max_depth": [3], "learning_rate": [0.3]},
}

CLASS_VOCAB = {
    0: "app desktop gui window user click install run launch editor player".split(),
    1: "library import function api package module call return parse utility".split(),
    2: "plugin extension host addon vscode chrome theme hook integrate wordpress".split(),
}
COMMON_VOCAB = "the project open source code readme github build docs license".split()


def _check(step: str, cond: bool, detail: str = "") -> bool:
    status = "PASSOU" if cond else "FALHOU"
    msg = f"  [{status}] {step}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return cond


def _hash_files(paths: list[Path]) -> dict[Path, str | None]:
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in paths}


def _hash_dir_outputs(out: dict[str, Path]) -> dict[str, str]:
    return {k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in out.items() if p.is_file() and p.suffix != ".tiff"}


def _synthetic_data(rng):
    y_all = np.repeat([0, 1, 2], N_PER_CLASS)
    texts = []
    for i, label in enumerate(y_all):
        words = list(rng.choice(CLASS_VOCAB[label], 25)) + list(rng.choice(COMMON_VOCAB, 15))
        rng.shuffle(words)
        marker = f"zzmarker{i}"  # 2x: sobrevive ao min_count=2 se o texto entrar no treino
        texts.append(" ".join(words + [marker, marker]))
    X_frozen = {}
    for feature, dim in FROZEN_DIMS.items():
        X = rng.standard_normal((N_SAMPLES, dim)).astype("float32")
        X[y_all == 0, :2] += 1.5
        X[y_all == 1, 2:4] += 1.5
        X_frozen[feature] = X
    return texts, X_frozen, y_all


def _outputs(run_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint_dir":          run_dir / "checkpoints",
        "cv_scores_path":          run_dir / "scores.csv",
        "summary_path":            run_dir / "summary.csv",
        "confusion_matrices_path": run_dir / "confusion.json",
        "confusion_figure_path":   run_dir / "confusion.tiff",
        "oof_probas_path":         run_dir / "oof_probas.csv",
        "choices_path":            run_dir / "choices.csv",
    }


def main() -> None:
    print("=" * 60)
    print("Smoke test: nested_cv_full_selection.py")
    print(f"  n_outer={N_OUTER}  n_inner={N_INNER}  n_samples={N_SAMPLES}")
    print("=" * 60)

    all_ok = True
    prod_hashes_before = _hash_files(PROD_FILES)
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)

    try:
        from modelagem.nested_cv_full_selection import (
            doc2vec_for_fold, outer_folds, run_full_selection_nested_cv,
        )
        _check("import nested_cv_full_selection", True)
    except Exception as e:
        _check("import nested_cv_full_selection", False, str(e))
        shutil.rmtree(SMOKE_DIR)
        sys.exit(1)

    rng = np.random.default_rng(GLOBAL_SEED)
    texts, X_frozen, y_all = _synthetic_data(rng)
    features = ["doc2vec", *FROZEN_DIMS]
    common_kwargs = dict(
        features=features, param_grids=PARAM_GRIDS_SMOKE,
        n_outer=N_OUTER, n_inner=N_INNER, seed=GLOBAL_SEED,
    )

    # ── 1. Doc2Vec do fold nao ve os textos de teste ────────────────────────────
    folds = outer_folds(y_all, N_OUTER, GLOBAL_SEED)
    tr_idx, te_idx = folds[0]
    model, X_d2v = doc2vec_for_fold(texts, tr_idx)
    vocab = model.wv.key_to_index
    leaked = [i for i in te_idx if f"zzmarker{i}" in vocab]
    present = [i for i in tr_idx if f"zzmarker{i}" in vocab]
    all_ok &= _check("marcadores dos textos de teste ausentes do vocabulario do Doc2Vec",
                     len(leaked) == 0, f"vazados: {len(leaked)}")
    all_ok &= _check("marcadores dos textos de treino presentes (controle positivo)",
                     len(present) == len(tr_idx), f"{len(present)}/{len(tr_idx)}")
    all_ok &= _check("Doc2Vec infere vetores para todos os textos", X_d2v.shape == (N_SAMPLES, 100),
                     f"shape: {X_d2v.shape}")

    # ── 2. Mesmos folds externos de run_nested_cv_configs ──────────────────────────────────────
    from sklearn.model_selection import StratifiedKFold
    ref = list(StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=GLOBAL_SEED)
               .split(np.zeros((N_SAMPLES, 1)), y_all))
    same = all(np.array_equal(a[1], b[1]) for a, b in zip(folds, ref))
    all_ok &= _check("folds externos identicos aos de run_nested_cv_configs", same)

    # ── 3. Execucao direta ──────────────────────────────────────────────────────
    direct = _outputs(SMOKE_DIR / "direct")
    try:
        done = run_full_selection_nested_cv(texts, X_frozen, y_all, **common_kwargs, **direct)
        ok = _check("execucao direta completa", done is True)
    except Exception:
        ok = _check("execucao direta completa", False)
        traceback.print_exc()
    all_ok &= ok
    if not ok:
        _finish(all_ok, prod_hashes_before)
        return

    # ── 4. Execucao interrompida (1 fold por vez) e retomada ────────────────────
    resumed = _outputs(SMOKE_DIR / "resumed")
    partial_results = []
    for _ in range(N_OUTER):
        partial_results.append(
            run_full_selection_nested_cv(texts, X_frozen, y_all, **common_kwargs, max_folds=1, **resumed)
        )
    all_ok &= _check("execucao com --max-folds=1 para antes do fim e completa na ultima chamada",
                     partial_results == [False] * (N_OUTER - 1) + [True], str(partial_results))
    h_direct, h_resumed = _hash_dir_outputs(direct), _hash_dir_outputs(resumed)
    diff = [k for k in h_direct if h_direct[k] != h_resumed.get(k)]
    all_ok &= _check("resultados retomados identicos aos da execucao direta", len(diff) == 0,
                     f"diferentes: {diff}" if diff else "")

    ckpt_mtimes = {p: p.stat().st_mtime for p in resumed["checkpoint_dir"].iterdir()}
    run_full_selection_nested_cv(texts, X_frozen, y_all, **common_kwargs, **resumed)
    unchanged = all(p.stat().st_mtime == t for p, t in ckpt_mtimes.items())
    all_ok &= _check("rodar de novo com tudo concluido nao recalcula nenhum fold", unchanged)

    # ── 5. Schema dos arquivos finais ───────────────────────────────────────────
    scores = pd.read_csv(direct["cv_scores_path"])
    all_ok &= _check(f"scores tem {N_OUTER} linhas", len(scores) == N_OUTER, f"{len(scores)}")
    expected = {"fold"}
    for f in features:
        expected |= {f"{m}_{f}" for m in ["f1_macro", "accuracy", "f1_application", "f1_helper",
                                          "f1_extender", "chosen_model", "best_params", "best_cv_score"]}
        expected |= {f"inner_score_{f}_{m}" for m in PARAM_GRIDS_SMOKE}
    missing = expected - set(scores.columns)
    all_ok &= _check("scores tem colunas de metricas, escolha e scores internos por embedding",
                     not missing, f"faltando: {missing}" if missing else "")
    valid_choice = all(scores[f"chosen_model_{f}"].isin(PARAM_GRIDS_SMOKE).all() for f in features)
    all_ok &= _check("classificador escolhido e sempre um dos 3", valid_choice)
    best_is_max = all(
        np.isclose(r[f"best_cv_score_{f}"], max(r[f"inner_score_{f}_{m}"] for m in PARAM_GRIDS_SMOKE))
        for _, r in scores.iterrows() for f in features
    )
    all_ok &= _check("escolhido tem o maior F1-macro interno", best_is_max)

    oof = pd.read_csv(direct["oof_probas_path"])
    all_ok &= _check("OOF: cada repositorio aparece uma vez por embedding",
                     len(oof) == N_SAMPLES * len(features)
                     and all(sorted(oof[oof.feature == f].row_idx) == list(range(N_SAMPLES)) for f in features),
                     f"{len(oof)} linhas")
    psum = oof[["p_application", "p_helper", "p_extender"]].sum(axis=1)
    all_ok &= _check("OOF: probabilidades somam 1", bool(np.allclose(psum, 1.0, atol=1e-5)))
    argmax_ok = (oof[["p_application", "p_helper", "p_extender"]].to_numpy().argmax(axis=1) == oof.y_pred).mean()
    all_ok &= _check("OOF: y_pred coincide com a maior probabilidade", argmax_ok > 0.99, f"{argmax_ok:.3f}")
    all_ok &= _check("OOF: y_true bate com y_all", bool((oof.y_true.to_numpy() == y_all[oof.row_idx]).all()))

    choices = pd.read_csv(direct["choices_path"])
    per_feature = choices.groupby("feature").n_folds_escolhido.sum()
    all_ok &= _check("contagem de escolhas soma n_outer por embedding",
                     bool((per_feature == N_OUTER).all()), per_feature.to_dict().__str__())

    cm = json.loads(direct["confusion_matrices_path"].read_text(encoding="utf-8"))
    totals = {f: int(np.array(cm["raw"][f]).sum()) for f in features}
    all_ok &= _check("matriz de confusao agregada soma N_SAMPLES por embedding",
                     all(t == N_SAMPLES for t in totals.values()), str(totals))
    all_ok &= _check("figura comparativa gerada", direct["confusion_figure_path"].exists())

    _finish(all_ok, prod_hashes_before)


def _finish(all_ok: bool, prod_hashes_before: dict) -> None:
    prod_hashes_after = _hash_files(PROD_FILES)
    changed = [str(p) for p in PROD_FILES if prod_hashes_before[p] != prod_hashes_after[p]]
    all_ok &= _check("arquivos de producao intactos (hash antes == depois)",
                     not changed, f"alterado(s): {changed}" if changed else "")
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    print("=" * 60)
    print("RESULTADO: TODOS OS CHECKS PASSARAM — pode executar a producao" if all_ok
          else "RESULTADO: UM OU MAIS CHECKS FALHARAM — corrigir antes de executar")
    print("=" * 60)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
