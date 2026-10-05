"""Nested 10x5 CV com selecao completa dentro de cada fold externo.

Compara as tres representacoes textuais (Doc2Vec, SBERT e CodeBERT) sem que
nenhuma decisao de modelagem use os dados de teste do fold externo. Em cada
fold externo:

  1. o Doc2Vec e reajustado SOMENTE com os textos de treino daquele fold e
     infere os vetores de todos os repositorios (SBERT/CodeBERT sao encoders
     congelados, sem ajuste sobre o corpus — reusam os X_*.npy existentes);
  2. para cada embedding, os 3 classificadores (RF/LogReg/XGBoost, SMOTE no
     pipeline) passam pelo GridSearchCV no CV interno, e o de maior F1-macro
     interno e escolhido (empate: rf > logreg > xgboost);
  3. o vencedor e avaliado uma unica vez no fold externo de teste, e as
     probabilidades (predict_proba) de cada repositorio de teste sao salvas.

O Doc2Vec nao e reajustado dentro dos folds internos: a estimativa de
desempenho depende de o fold externo de teste ficar intocado, o que ja e
garantido; o reajuste interno multiplicaria o custo por ~6 e so afetaria a
escolha do classificador.

Os folds externos sao definidos por StratifiedKFold com semente fixa sobre
y_all = train+val+test_indist (os mesmos de common.py::run_nested_cv_configs),
o que garante o pareamento por fold entre as representacoes. Cada fold e salvo em disco assim que termina
(checkpoint): a execucao pode ser interrompida e retomada do fold seguinte,
ou limitada com --max-folds para dividir o trabalho em mais de uma sessao.

Uso:
    python modelagem/nested_cv_full_selection.py                # roda/retoma tudo
    python modelagem/nested_cv_full_selection.py --max-folds 5  # para apos 5 folds novos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED
from modelagem.common import (
    DATA_DIR,
    FULL_SELECTION_CHECKPOINT_DIR,
    FULL_SELECTION_CHOICES_PATH,
    FULL_SELECTION_CONFUSION_PATH,
    FULL_SELECTION_CV_PATH,
    FULL_SELECTION_FIGURE_PATH,
    FULL_SELECTION_OOF_PROBAS_PATH,
    FULL_SELECTION_SUMMARY_PATH,
    LABEL_CLASSES,
    compute_metrics,
    load_split,
    summarize_nested_cv,
)
from modelagem.compare_models import (
    FEATURES,
    MODEL_ORDER,
    MODEL_TYPES,
    _param_grid_for,
    build_pipeline,
)
from preparacao.features_doc2vec import infer_doc2vec, train_doc2vec_model
from preparacao.text_utils import make_combined_texts

SUPERVISED_SPLITS = ["train", "val", "test_indist"]
FROZEN_FEATURES = ["sbert", "codebert"]  # encoders congelados: sem ajuste por fold


def load_supervised_texts(
    dataset_path: Path = Path("data/processed/dataset_clean.parquet"),
    splits_path: Path = Path("data/processed/splits.json"),
) -> list[str]:
    """Textos (nome + README truncado) na mesma ordem das linhas de X_all:
    train, val, test_indist concatenados — mesma ordem usada para gerar os
    X_{feature}_{split}.npy e y_{split}.npy (build_features.py)."""
    df = pd.read_parquet(dataset_path)
    splits = json.loads(splits_path.read_text())
    id_to_idx = {rid: i for i, rid in enumerate(df["repo_id"])}
    all_texts = make_combined_texts(df)
    texts: list[str] = []
    for split in SUPERVISED_SPLITS:
        texts.extend(all_texts[id_to_idx[rid]] for rid in splits[split] if rid in id_to_idx)
    return texts


def load_frozen_features(
    features: list[str], data_dir: Path = DATA_DIR
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """X_all (train+val+test_indist) dos encoders congelados + y_all."""
    X_frozen: dict[str, np.ndarray] = {}
    y_all: np.ndarray | None = None
    for feature in features:
        parts = [load_split(split, feature, data_dir=data_dir) for split in SUPERVISED_SPLITS]
        X_frozen[feature] = np.concatenate([X for X, _ in parts])
        if y_all is None:
            y_all = np.concatenate([y for _, y in parts])
    if y_all is None:
        y_all = np.concatenate([np.load(data_dir / f"y_{s}.npy") for s in SUPERVISED_SPLITS])
    return X_frozen, y_all


def doc2vec_for_fold(texts: list[str], tr_idx: np.ndarray) -> tuple[object, np.ndarray]:
    """Ajusta o Doc2Vec so com os textos de treino do fold e infere os vetores
    de todos os textos (treino e teste do fold)."""
    model = train_doc2vec_model([texts[i] for i in tr_idx])
    return model, infer_doc2vec(model, texts)


def outer_folds(y_all: np.ndarray, n_outer: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Mesmos folds externos de common.py::run_nested_cv_configs."""
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    return list(outer_cv.split(np.zeros((len(y_all), 1)), y_all))


def select_and_evaluate(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
    model_types: list[str], param_grids: dict[str, dict], inner_cv,
) -> dict:
    """GridSearchCV no CV interno para cada classificador; escolhe o de maior
    F1-macro interno (empate pela ordem rf > logreg > xgboost) e prediz o
    fold externo com o vencedor (ja reajustado em todo X_tr, refit=True)."""
    grids = {}
    for model_type in model_types:
        pg = {f"clf__{k}": v for k, v in param_grids[model_type].items()}
        grid = GridSearchCV(
            build_pipeline(model_type), pg, cv=inner_cv, scoring="f1_macro",
            n_jobs=-1, refit=True, verbose=0,
        )
        grid.fit(X_tr, y_tr)
        grids[model_type] = grid

    chosen = min(model_types, key=lambda m: (-grids[m].best_score_, MODEL_ORDER[m]))
    best = grids[chosen]
    return {
        "chosen": chosen,
        "inner_scores": {m: float(g.best_score_) for m, g in grids.items()},
        "best_params": {k.replace("clf__", ""): v for k, v in best.best_params_.items()},
        "proba": best.predict_proba(X_te),
        "pred": best.predict(X_te),
    }


def _checkpoint_paths(checkpoint_dir: Path, fold: int) -> tuple[Path, Path]:
    return checkpoint_dir / f"fold_{fold:02d}.json", checkpoint_dir / f"fold_{fold:02d}.npz"


def run_fold(
    fold: int, tr_idx: np.ndarray, te_idx: np.ndarray,
    texts: list[str], X_frozen: dict[str, np.ndarray], y_all: np.ndarray,
    features: list[str], model_types: list[str], param_grids: dict[str, dict],
    n_inner: int, seed: int, checkpoint_dir: Path,
) -> None:
    """Executa um fold externo completo e grava o checkpoint (JSON + NPZ)."""
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    record: dict = {"fold": fold}
    arrays: dict[str, np.ndarray] = {"te_idx": te_idx}

    for feature in features:
        t0 = time.time()
        if feature == "doc2vec":
            _, X_all = doc2vec_for_fold(texts, tr_idx)
            print(f"    [doc2vec] reajustado no treino do fold ({len(tr_idx)} textos) "
                  f"em {(time.time() - t0) / 60:.1f} min")
            t0 = time.time()
        else:
            X_all = X_frozen[feature]

        res = select_and_evaluate(
            X_all[tr_idx], y_tr, X_all[te_idx], model_types, param_grids, inner_cv,
        )
        m = compute_metrics(y_te, res["pred"])

        record[f"f1_macro_{feature}"]       = m["f1_macro"]
        record[f"accuracy_{feature}"]       = m["accuracy"]
        record[f"f1_application_{feature}"] = m["f1_application"]
        record[f"f1_helper_{feature}"]      = m["f1_helper"]
        record[f"f1_extender_{feature}"]    = m["f1_extender"]
        record[f"chosen_model_{feature}"]   = res["chosen"]
        record[f"best_params_{feature}"]    = json.dumps(res["best_params"])
        record[f"best_cv_score_{feature}"]  = res["inner_scores"][res["chosen"]]
        for model_type, score in res["inner_scores"].items():
            record[f"inner_score_{feature}_{model_type}"] = score
        arrays[f"proba_{feature}"] = res["proba"]
        arrays[f"pred_{feature}"] = res["pred"]

        print(f"    [{feature}] escolhido={res['chosen']:8s} F1-macro={m['f1_macro']:.4f}  "
              f"F1-ext={m['f1_extender']:.4f}  ({(time.time() - t0) / 60:.1f} min)")

    json_path, npz_path = _checkpoint_paths(checkpoint_dir, fold)
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")  # por ultimo: marca o fold como completo


def aggregate(
    checkpoint_dir: Path, folds: list[tuple[np.ndarray, np.ndarray]],
    y_all: np.ndarray, features: list[str], model_types: list[str],
    cv_scores_path: Path, summary_path: Path,
    confusion_matrices_path: Path, confusion_figure_path: Path,
    oof_probas_path: Path, choices_path: Path,
    labels: list[str] = LABEL_CLASSES,
) -> None:
    """Junta os checkpoints de todos os folds nos arquivos finais."""
    n_classes = len(labels)
    records = []
    confusion_sums = {f: np.zeros((n_classes, n_classes), dtype=np.int64) for f in features}
    oof_rows = []

    for fold, (_, te_idx) in enumerate(folds):
        json_path, npz_path = _checkpoint_paths(checkpoint_dir, fold)
        records.append(json.loads(json_path.read_text(encoding="utf-8")))
        arrays = np.load(npz_path)
        if not np.array_equal(arrays["te_idx"], te_idx):
            raise RuntimeError(f"checkpoint do fold {fold} nao corresponde aos folds atuais")
        y_te = y_all[te_idx]
        for feature in features:
            pred = arrays[f"pred_{feature}"]
            proba = arrays[f"proba_{feature}"]
            confusion_sums[feature] += confusion_matrix(y_te, pred, labels=list(range(n_classes)))
            chosen = records[-1][f"chosen_model_{feature}"]
            for i, row_idx in enumerate(te_idx):
                oof_rows.append({
                    "row_idx": int(row_idx), "fold": fold, "feature": feature,
                    "chosen_model": chosen, "y_true": int(y_te[i]), "y_pred": int(pred[i]),
                    **{f"p_{label}": float(proba[i, k]) for k, label in enumerate(labels)},
                })

    scores_df = pd.DataFrame(records)
    summarize_nested_cv(
        scores_df, confusion_sums, features,
        n_outer=len(folds),
        cv_scores_path=cv_scores_path,
        summary_path=summary_path,
        confusion_matrices_path=confusion_matrices_path,
        confusion_figure_path=confusion_figure_path,
        labels=labels,
    )

    oof_probas_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof_rows).to_csv(oof_probas_path, index=False)
    print(f"  Probabilidades out-of-fold salvas em {oof_probas_path}")

    # Frequencia com que cada classificador venceu a selecao interna, por embedding
    choices = pd.DataFrame([
        {"feature": f, "model": m,
         "n_folds_escolhido": int((scores_df[f"chosen_model_{f}"] == m).sum())}
        for f in features for m in model_types
    ])
    choices.to_csv(choices_path, index=False)
    print("\n  Classificador escolhido por fold (contagem):")
    for feature in features:
        sub = choices[choices["feature"] == feature]
        counts = ", ".join(f"{r.model}={r.n_folds_escolhido}" for r in sub.itertuples())
        print(f"    {feature:10s}: {counts}")


def run_full_selection_nested_cv(
    texts: list[str],
    X_frozen: dict[str, np.ndarray],
    y_all: np.ndarray,
    features: list[str] = FEATURES,
    model_types: list[str] = MODEL_TYPES,
    param_grids: dict[str, dict] | None = None,
    n_outer: int = 10,
    n_inner: int = 5,
    seed: int = GLOBAL_SEED,
    max_folds: int | None = None,
    checkpoint_dir: Path = FULL_SELECTION_CHECKPOINT_DIR,
    cv_scores_path: Path = FULL_SELECTION_CV_PATH,
    summary_path: Path = FULL_SELECTION_SUMMARY_PATH,
    confusion_matrices_path: Path = FULL_SELECTION_CONFUSION_PATH,
    confusion_figure_path: Path = FULL_SELECTION_FIGURE_PATH,
    oof_probas_path: Path = FULL_SELECTION_OOF_PROBAS_PATH,
    choices_path: Path = FULL_SELECTION_CHOICES_PATH,
) -> bool:
    """Roda (ou retoma) o nested CV com selecao completa. Retorna True se
    todos os folds estao completos e os arquivos finais foram gerados."""
    if len(texts) != len(y_all):
        raise ValueError(f"textos ({len(texts)}) e y_all ({len(y_all)}) desalinhados")
    for feature, X in X_frozen.items():
        if len(X) != len(y_all):
            raise ValueError(f"X_all de {feature} ({len(X)}) e y_all ({len(y_all)}) desalinhados")

    grids = {m: (param_grids or {}).get(m, _param_grid_for(m)) for m in model_types}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    folds = outer_folds(y_all, n_outer, seed)

    new_folds = 0
    for fold, (tr_idx, te_idx) in enumerate(folds):
        if _checkpoint_paths(checkpoint_dir, fold)[0].exists():
            print(f"\n  Fold {fold + 1}/{n_outer}: ja concluido (checkpoint) — pulando")
            continue
        if max_folds is not None and new_folds >= max_folds:
            print(f"\n  Limite de {max_folds} fold(s) nesta execucao atingido — "
                  f"rode novamente para continuar do fold {fold + 1}.")
            return False
        t0 = time.time()
        print(f"\n  Fold {fold + 1}/{n_outer} (treino={len(tr_idx)}, teste={len(te_idx)})")
        run_fold(
            fold, tr_idx, te_idx, texts, X_frozen, y_all,
            features, model_types, grids, n_inner, seed, checkpoint_dir,
        )
        new_folds += 1
        print(f"  Fold {fold + 1}/{n_outer} concluido em {(time.time() - t0) / 60:.1f} min", flush=True)

    aggregate(
        checkpoint_dir, folds, y_all, features, model_types,
        cv_scores_path=cv_scores_path, summary_path=summary_path,
        confusion_matrices_path=confusion_matrices_path,
        confusion_figure_path=confusion_figure_path,
        oof_probas_path=oof_probas_path, choices_path=choices_path,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Numero maximo de folds novos nesta execucao (retoma depois).")
    args = parser.parse_args()

    print("Nested 10x5 CV com selecao completa dentro do fold (Doc2Vec + classificador)")
    texts = load_supervised_texts()
    X_frozen, y_all = load_frozen_features(FROZEN_FEATURES)
    print(f"  {len(texts)} repositorios supervisionados; embeddings: {FEATURES}; "
          f"classificadores: {MODEL_TYPES}")

    t0 = time.time()
    done = run_full_selection_nested_cv(texts, X_frozen, y_all, max_folds=args.max_folds)
    print(f"\nTempo desta execucao: {(time.time() - t0) / 3600:.2f} h")
    if done:
        print(f"Resultados salvos em {FULL_SELECTION_CV_PATH}, {FULL_SELECTION_SUMMARY_PATH}, "
              f"{FULL_SELECTION_OOF_PROBAS_PATH}, {FULL_SELECTION_CHOICES_PATH}")


if __name__ == "__main__":
    main()
