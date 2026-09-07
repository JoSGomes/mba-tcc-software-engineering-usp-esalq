"""Testes do pipeline de preparação de dados.

Cobre:
1. Split reprodutível a partir do seed
2. No-leakage: preprocessadores fitted apenas em train
3. Integridade de labels por subset
4. Subset OOD separado de in-distribution
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

TRAIN_LABELS = {"application", "helper", "extender"}
OOD_LABELS = {"other", "incerto"}


# ---------------------------------------------------------------------------
# Fixtures de dataset sintético
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_df():
    """200 repos sintéticos com distribuição realista."""
    rng = np.random.default_rng(42)
    n = 200
    labels = (
        ["application"] * 80
        + ["helper"] * 70
        + ["extender"] * 20
        + ["other"] * 16
        + ["incerto"] * 14
    )
    rng.shuffle(labels)
    df = pd.DataFrame(
        {
            "repo_id": [f"org{i}/repo{i}" for i in range(n)],
            "name": [f"repo{i}" for i in range(n)],
            "language": rng.choice(["Python", "JavaScript", "Go", "Rust", "Java"], n).tolist(),
            "stars": rng.integers(1, 100000, n).tolist(),
            "forks": rng.integers(0, 10000, n).tolist(),
            "watchers": rng.integers(0, 5000, n).tolist(),
            "open_issues": rng.integers(0, 500, n).tolist(),
            "size_kb": rng.integers(10, 100000, n).tolist(),
            "has_readme": rng.choice([True, False], n).tolist(),
            "has_license": rng.choice([True, False], n).tolist(),
            "has_wiki": rng.choice([True, False], n).tolist(),
            "commits": rng.integers(1, 5000, n).tolist(),
            "contributors": rng.integers(1, 500, n).tolist(),
            "created_at": ["2020-01-01T00:00:00"] * n,
            "updated_at": ["2024-01-01T00:00:00"] * n,
            "readme_text": [f"This is a {labels[i]} library for doing things." for i in range(n)],
            "collected_at": ["2026-05-08T00:00:00+00:00"] * n,
            "label": labels,
            "confidence": [0.9] * n,
            "split_eligibility": [
                "train_val_test" if lbl in TRAIN_LABELS else "test_only_ood"
                for lbl in labels
            ],
            "age_days": [2000] * n,
            "text": [f"repo{i} readme text" for i in range(n)],
        }
    )
    return df


@pytest.fixture()
def splits(synthetic_df):
    from preparacao.split import split_dataset
    return split_dataset(synthetic_df, seed=42)


# ---------------------------------------------------------------------------
# T1: split reprodutível
# ---------------------------------------------------------------------------

class TestSplitReproducibility:
    def test_same_seed_same_splits(self, synthetic_df):
        from preparacao.split import split_dataset
        s1 = split_dataset(synthetic_df, seed=42)
        s2 = split_dataset(synthetic_df, seed=42)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test_indist"] == s2["test_indist"]
        assert s1["test_ood"] == s2["test_ood"]

    def test_different_seed_different_splits(self, synthetic_df):
        from preparacao.split import split_dataset
        s1 = split_dataset(synthetic_df, seed=42)
        s2 = split_dataset(synthetic_df, seed=99)
        assert s1["train"] != s2["train"]

    def test_ratios_approximate(self, synthetic_df, splits):
        in_dist = synthetic_df[synthetic_df["split_eligibility"] == "train_val_test"]
        n = len(in_dist)
        assert abs(len(splits["train"]) / n - 0.70) < 0.05
        assert abs(len(splits["val"]) / n - 0.15) < 0.05
        assert abs(len(splits["test_indist"]) / n - 0.15) < 0.05


# ---------------------------------------------------------------------------
# T2: separação OOD / in-distribution
# ---------------------------------------------------------------------------

class TestOODSeparation:
    def test_train_val_test_indist_no_ood_labels(self, synthetic_df, splits):
        id_to_label = dict(zip(synthetic_df["repo_id"], synthetic_df["label"]))
        for subset in ("train", "val", "test_indist"):
            labels = {id_to_label[i] for i in splits[subset]}
            assert labels <= TRAIN_LABELS, (
                f"{subset} contém labels OOD: {labels - TRAIN_LABELS}"
            )

    def test_ood_has_only_ood_labels(self, synthetic_df, splits):
        id_to_label = dict(zip(synthetic_df["repo_id"], synthetic_df["label"]))
        ood_labels = {id_to_label[i] for i in splits["test_ood"]}
        assert ood_labels <= OOD_LABELS, (
            f"test_ood contém labels in-distribution: {ood_labels - OOD_LABELS}"
        )

    def test_no_overlap_between_subsets(self, splits):
        all_subsets = ["train", "val", "test_indist", "test_ood"]
        for i, a in enumerate(all_subsets):
            for b in all_subsets[i + 1:]:
                overlap = set(splits[a]) & set(splits[b])
                assert not overlap, f"Overlap entre {a} e {b}: {overlap}"


# ---------------------------------------------------------------------------
# T3: no-leakage — doc2vec treinado apenas em train
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_doc2vec_trained_only_on_train(self, synthetic_df, splits, tmp_path):
        from preparacao.features_doc2vec import build_doc2vec
        results = build_doc2vec(synthetic_df, splits, output_dir=tmp_path)

        from gensim.models.doc2vec import Doc2Vec
        model = Doc2Vec.load(str(tmp_path / "doc2vec_model.bin"))
        # Doc2Vec treinado em train: corpus_count deve igualar len(train)
        assert model.corpus_count == len(splits["train"]), (
            f"Doc2Vec treinado em {model.corpus_count} docs, "
            f"mas train tem {len(splits['train'])}"
        )

    def test_doc2vec_shapes_consistent(self, synthetic_df, splits, tmp_path):
        from preparacao.features_doc2vec import build_doc2vec
        results = build_doc2vec(synthetic_df, splits, output_dir=tmp_path)
        n_features = results["train"].shape[1]
        assert n_features == 100, f"Doc2Vec vec_size esperado 100, obtido {n_features}"
        for split_name in ("train", "val", "test_indist", "test_ood"):
            X = results[split_name]
            assert X.shape[0] == len(splits[split_name]), (
                f"Shape inconsistente em {split_name}: "
                f"X.shape[0]={X.shape[0]}, splits={len(splits[split_name])}"
            )
            assert X.shape[1] == n_features


# ---------------------------------------------------------------------------
# T4: integridade de labels codificados
# ---------------------------------------------------------------------------

class TestLabelEncoding:
    def test_label_classes_fixed(self, synthetic_df, splits, tmp_path):
        from preparacao.build_features import LABEL_TO_INT, _save_labels
        _save_labels(synthetic_df, splits, tmp_path)

        assert set(LABEL_TO_INT.keys()) == {"application", "helper", "extender"}
        assert set(LABEL_TO_INT.values()) == {0, 1, 2}

    def test_y_train_only_valid_classes(self, synthetic_df, splits, tmp_path):
        from preparacao.build_features import LABEL_TO_INT, _save_labels
        _save_labels(synthetic_df, splits, tmp_path)

        for split_name in ("train", "val", "test_indist"):
            y = np.load(tmp_path / f"y_{split_name}.npy")
            valid = set(LABEL_TO_INT.values())
            assert set(y.tolist()).issubset(valid), (
                f"{split_name} contém classes inválidas: {set(y.tolist()) - valid}"
            )

    def test_y_test_ood_not_saved(self, synthetic_df, splits, tmp_path):
        from preparacao.build_features import _save_labels
        _save_labels(synthetic_df, splits, tmp_path)
        assert not (tmp_path / "y_test_ood.npy").exists(), (
            "test_ood não deve ter y (não supervisionado)"
        )


# ---------------------------------------------------------------------------
# T5: pipeline clean.py
# ---------------------------------------------------------------------------

class TestClean:
    def test_no_duplicates(self, synthetic_df, tmp_path):
        from preparacao.clean import assign_eligibility, compute_age_days, make_text
        # Duplicar alguns repos
        doubled = pd.concat([synthetic_df, synthetic_df.head(10)], ignore_index=True)
        result = doubled.drop_duplicates(subset=["repo_id"], keep="first")
        assert len(result) == len(synthetic_df)
        assert result["repo_id"].is_unique

    def test_eligibility_assignment(self):
        from preparacao.clean import assign_eligibility
        assert assign_eligibility("application") == "train_val_test"
        assert assign_eligibility("helper") == "train_val_test"
        assert assign_eligibility("extender") == "train_val_test"
        assert assign_eligibility("other") == "test_only_ood"
        assert assign_eligibility("incerto") == "test_only_ood"

    def test_make_text_combines_name_readme(self):
        from preparacao.clean import make_text
        result = make_text("mylib", "A Python library")
        assert "mylib" in result
        assert "A Python library" in result

    def test_make_text_handles_empty_readme(self):
        from preparacao.clean import make_text
        result = make_text("mylib", None)
        assert result == "mylib"
