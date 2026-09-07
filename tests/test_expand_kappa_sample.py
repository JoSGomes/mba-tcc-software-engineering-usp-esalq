"""Testes de extracao/expand_kappa_sample.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extracao.expand_kappa_sample import (
    build_batch2_csv,
    build_review_input,
    compute_stratified_counts,
    sample_new_repos,
)


@pytest.fixture()
def review_df():
    """Réplica proporcional de kappa_review.csv real: app 29, helper 25, extender 4, other 2."""
    rows = (
        [{"repo_id": f"rev/app{i}", "name": f"app{i}", "label": "application",
          "confidence": 0.9, "reasoning": "r", "manual_label": "application"} for i in range(29)]
        + [{"repo_id": f"rev/helper{i}", "name": f"helper{i}", "label": "helper",
            "confidence": 0.9, "reasoning": "r", "manual_label": "helper"} for i in range(25)]
        + [{"repo_id": f"rev/ext{i}", "name": f"ext{i}", "label": "extender",
            "confidence": 0.9, "reasoning": "r", "manual_label": "extender"} for i in range(4)]
        + [{"repo_id": f"rev/other{i}", "name": f"other{i}", "label": "other",
            "confidence": 0.9, "reasoning": "r", "manual_label": "other"} for i in range(2)]
    )
    return pd.DataFrame(rows)


@pytest.fixture()
def gold_df(review_df):
    """Pool grande o suficiente para amostrar, incluindo os já revisados
    (para testar exclusão) e uma fatia 'incerto' (para testar exclusão)."""
    rng = np.random.default_rng(42)
    extra_rows = []
    counts = {"application": 200, "helper": 200, "extender": 60, "other": 60, "incerto": 30}
    for label, n in counts.items():
        for i in range(n):
            extra_rows.append(
                {
                    "repo_id": f"pool/{label}{i}",
                    "name": f"{label}{i}",
                    "label": label,
                    "confidence": float(rng.uniform(0.6, 1.0)),
                    "reasoning": "r",
                    "label_source": "haiku_auto",
                }
            )
    extra_df = pd.DataFrame(extra_rows)
    # inclui os já revisados no gold_df também (reflete a realidade: gold_standard.csv
    # tem TODOS os repos, incluindo os que já foram para kappa_review.csv)
    reviewed_in_gold = review_df[["repo_id", "name", "label", "confidence", "reasoning"]].copy()
    reviewed_in_gold["label_source"] = "haiku_auto"
    return pd.concat([extra_df, reviewed_in_gold], ignore_index=True)


class TestComputeStratifiedCounts:
    def test_sums_to_n(self, review_df):
        counts = compute_stratified_counts(review_df, n=40)
        assert sum(counts.values()) == 40

    def test_proportional_ordering_preserved(self, review_df):
        counts = compute_stratified_counts(review_df, n=40)
        # application (29/60) deve ter mais amostras que helper (25/60),
        # que deve ter mais que extender (4/60) e other (2/60)
        assert counts["application"] >= counts["helper"] >= counts["extender"]
        assert counts["other"] <= counts["extender"]

    def test_all_positive_for_nonzero_classes(self, review_df):
        counts = compute_stratified_counts(review_df, n=40)
        assert all(v > 0 for v in counts.values())


class TestSampleNewRepos:
    def test_no_overlap_with_already_reviewed(self, gold_df, review_df):
        sampled = sample_new_repos(gold_df, review_df, n=40, seed=42)
        assert not set(sampled["repo_id"]) & set(review_df["repo_id"])

    def test_excludes_incerto(self, gold_df, review_df):
        sampled = sample_new_repos(gold_df, review_df, n=40, seed=42)
        assert "incerto" not in set(sampled["label"])

    def test_sample_size_matches_n(self, gold_df, review_df):
        sampled = sample_new_repos(gold_df, review_df, n=40, seed=42)
        assert len(sampled) == 40

    def test_reproducible_with_same_seed(self, gold_df, review_df):
        s1 = sample_new_repos(gold_df, review_df, n=40, seed=42)
        s2 = sample_new_repos(gold_df, review_df, n=40, seed=42)
        assert set(s1["repo_id"]) == set(s2["repo_id"])

    def test_insufficient_pool_raises(self, review_df):
        # gold_df minúsculo, sem repos suficientes para 40 novos
        tiny_gold = pd.DataFrame(
            [{"repo_id": "x/1", "name": "x1", "label": "application", "confidence": 0.9, "reasoning": "r"}]
        )
        with pytest.raises(ValueError, match="Pool insuficiente"):
            sample_new_repos(tiny_gold, review_df, n=40, seed=42)


class TestBuildBatch2Csv:
    def test_schema_matches_kappa_review_csv(self, gold_df, review_df):
        sampled = sample_new_repos(gold_df, review_df, n=40, seed=42)
        batch2 = build_batch2_csv(sampled)
        assert list(batch2.columns) == ["repo_id", "name", "label", "confidence", "reasoning", "manual_label"]

    def test_manual_label_empty(self, gold_df, review_df):
        sampled = sample_new_repos(gold_df, review_df, n=40, seed=42)
        batch2 = build_batch2_csv(sampled)
        assert (batch2["manual_label"] == "").all()


class TestBuildReviewInput:
    def test_combines_reviewed_and_new(self, gold_df, review_df):
        sampled = sample_new_repos(gold_df, review_df, n=40, seed=42)
        batch2 = build_batch2_csv(sampled)
        dataset_df = pd.DataFrame(
            {
                "repo_id": pd.concat([review_df["repo_id"], batch2["repo_id"]], ignore_index=True),
                "readme_text": ["some readme text " * 5] * (len(review_df) + len(batch2)),
            }
        )
        records = build_review_input(review_df, batch2, dataset_df)
        assert len(records) == len(review_df) + len(batch2)
        repo_ids = {r["repo_id"] for r in records}
        assert repo_ids == set(review_df["repo_id"]) | set(batch2["repo_id"])

    def test_readme_truncated(self, review_df):
        batch2 = pd.DataFrame(columns=review_df.columns)
        long_readme = "x" * 5000
        dataset_df = pd.DataFrame(
            {"repo_id": review_df["repo_id"], "readme_text": [long_readme] * len(review_df)}
        )
        records = build_review_input(review_df, batch2, dataset_df, max_chars=100)
        assert all(len(r["readme_text"]) <= 100 for r in records)

    def test_missing_readme_becomes_empty_string(self, review_df):
        batch2 = pd.DataFrame(columns=review_df.columns)
        dataset_df = pd.DataFrame({"repo_id": [], "readme_text": []})
        records = build_review_input(review_df, batch2, dataset_df)
        assert all(r["readme_text"] == "" for r in records)
