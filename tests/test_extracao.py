"""Tests for extraction pipeline: run_collection, gold_standard, compute_kappa."""

import csv
import inspect
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_repos_df():
    """Small DataFrame mimicking repos.parquet schema."""
    return pd.DataFrame([
        {
            "repo_id": "owner/repo1", "name": "repo1", "description": "A web framework",
            "language": "Python", "stars": 500, "forks": 80, "watchers": 500,
            "open_issues": 10, "size_kb": 2000, "has_readme": True,
            "has_license": True, "has_wiki": False, "commits": 300, "contributors": 12,
            "created_at": "2020-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
            "readme_text": "A Python web framework for building APIs", "collected_at": "2026-05-06",
        },
        {
            "repo_id": "owner/repo2", "name": "repo2", "description": "A discord bot",
            "language": "JavaScript", "stars": 120, "forks": 15, "watchers": 120,
            "open_issues": 3, "size_kb": 500, "has_readme": True,
            "has_license": False, "has_wiki": False, "commits": 80, "contributors": 2,
            "created_at": "2021-06-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
            "readme_text": "A bot for Discord servers", "collected_at": "2026-05-06",
        },
        {
            "repo_id": "owner/repo3", "name": "my-app", "description": "A todo app",
            "language": "TypeScript", "stars": 45, "forks": 5, "watchers": 45,
            "open_issues": 1, "size_kb": 300, "has_readme": True,
            "has_license": True, "has_wiki": False, "commits": 50, "contributors": 1,
            "created_at": "2022-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
            "readme_text": "A simple todo application", "collected_at": "2026-05-06",
        },
    ])


@pytest.fixture
def sample_gold_csv(tmp_path, sample_repos_df):
    """Write a minimal gold_standard.csv and return its path."""
    gold_path = tmp_path / "gold_standard.csv"
    rows = [
        {"repo_id": "owner/repo1", "name": "repo1", "label": "helper",
         "confidence": 0.95, "reasoning": "framework", "label_source": "haiku_auto"},
        {"repo_id": "owner/repo2", "name": "repo2", "label": "extender",
         "confidence": 0.88, "reasoning": "discord bot", "label_source": "haiku_auto"},
        {"repo_id": "owner/repo3", "name": "my-app", "label": "application",
         "confidence": 0.80, "reasoning": "todo app", "label_source": "haiku_auto"},
    ]
    pd.DataFrame(rows).to_csv(gold_path, index=False)
    return gold_path


# ---------------------------------------------------------------------------
# run_collection tests
# ---------------------------------------------------------------------------

class TestRunCollection:
    def test_record_to_dict_has_all_fields(self, sample_repos_df):
        from extracao.run_collection import record_to_dict
        from extracao.seart_ghs import RepoRecord

        repo = RepoRecord(
            repo_id="owner/repo", name="repo", description="desc",
            language="Python", stars=100, forks=10, watchers=100,
            open_issues=2, size_kb=500, has_readme=True, has_license=True,
            has_wiki=False, commits=50, contributors=3,
            created_at="2020-01-01", updated_at="2024-01-01",
        )
        result = record_to_dict(repo, collected_at="2026-05-06")
        expected_keys = {
            "repo_id", "name", "description", "language", "stars", "forks",
            "watchers", "open_issues", "size_kb", "has_readme", "has_license",
            "has_wiki", "commits", "contributors", "created_at", "updated_at",
            "readme_text", "collected_at",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_flush_to_parquet_produces_valid_file(self, tmp_path, sample_repos_df):
        from extracao.run_collection import _flush_to_parquet

        records = sample_repos_df.to_dict("records")
        output = tmp_path / "test.parquet"
        _flush_to_parquet(records, output)

        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == len(records)
        assert "repo_id" in result.columns
        assert result["has_readme"].dtype == bool

    def test_checkpoint_roundtrip(self, tmp_path):
        from extracao import run_collection as rc_module
        orig_checkpoint = rc_module.CHECKPOINT_FILE
        rc_module.CHECKPOINT_FILE = tmp_path / "checkpoint.json"

        state = {"collected": 100, "timestamp": "2026-05-06"}
        rc_module.save_checkpoint(state)
        loaded = rc_module.load_checkpoint()
        assert loaded == state

        rc_module.CHECKPOINT_FILE = orig_checkpoint

    def test_no_checkpoint_returns_none(self, tmp_path):
        from extracao import run_collection as rc_module
        orig = rc_module.CHECKPOINT_FILE
        rc_module.CHECKPOINT_FILE = tmp_path / "nonexistent.json"
        assert rc_module.load_checkpoint() is None
        rc_module.CHECKPOINT_FILE = orig


# ---------------------------------------------------------------------------
# gold_standard tests
# ---------------------------------------------------------------------------

class TestGoldStandard:
    def test_build_prompt_contains_name(self, sample_repos_df):
        from extracao.gold_standard import build_prompt

        row = sample_repos_df.iloc[0].to_dict()
        prompt = build_prompt(row)
        assert row["name"] in prompt
        assert "application" in prompt.lower()
        assert "helper" in prompt.lower()
        assert "extender" in prompt.lower()

    def test_build_prompt_handles_missing_readme(self):
        from extracao.gold_standard import build_prompt

        row = {"name": "test", "description": "desc", "language": "Python",
               "stars": 10, "readme_text": None}
        prompt = build_prompt(row)
        assert "sem README" in prompt

    def test_valid_labels_set(self):
        from extracao.gold_standard import VALID_LABELS
        assert VALID_LABELS == {"application", "helper", "extender", "other", "incerto"}

    def test_call_haiku_parses_valid_response(self):
        from unittest.mock import patch, MagicMock as MM
        from extracao.gold_standard import call_haiku

        mock_proc = MM(returncode=0, stdout='{"label": "helper", "confidence": 0.9, "reasoning": "library"}')
        with patch("extracao.gold_standard.subprocess.run", return_value=mock_proc):
            result = call_haiku(None, "test prompt")
        assert result["label"] == "helper"
        assert result["confidence"] == 0.9

    def test_call_haiku_handles_invalid_json(self):
        from unittest.mock import patch, MagicMock as MM
        from extracao.gold_standard import call_haiku

        mock_proc = MM(returncode=0, stdout="not json")
        with patch("extracao.gold_standard.subprocess.run", return_value=mock_proc):
            result = call_haiku(None, "test", retries=1)
        assert result["label"] == "incerto"

    def test_call_haiku_normalizes_invalid_label(self):
        from unittest.mock import patch, MagicMock as MM
        from extracao.gold_standard import call_haiku

        mock_proc = MM(returncode=0, stdout='{"label": "invalid_class", "confidence": 0.8, "reasoning": "?"}')
        with patch("extracao.gold_standard.subprocess.run", return_value=mock_proc):
            result = call_haiku(None, "test prompt")
        assert result["label"] == "incerto"


# ---------------------------------------------------------------------------
# compute_kappa tests
# ---------------------------------------------------------------------------

class TestComputeKappa:
    def test_load_gold_valid(self, sample_gold_csv):
        from extracao.compute_kappa import load_gold

        df = load_gold(sample_gold_csv)
        assert "repo_id" in df.columns
        assert "label" in df.columns
        assert "confidence" in df.columns

    def test_load_gold_missing_columns(self, tmp_path):
        from extracao.compute_kappa import load_gold

        bad_path = tmp_path / "bad.csv"
        pd.DataFrame({"repo_id": ["a"]}).to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_gold(bad_path)

    def test_compute_kappa_perfect_agreement(self):
        from extracao.compute_kappa import compute_kappa_report

        labels = ["application", "helper", "extender", "application", "helper"]
        result = compute_kappa_report(haiku_labels=labels, manual_labels=labels)
        assert result["kappa"] == pytest.approx(1.0, abs=1e-9)

    def test_compute_kappa_report_structure(self):
        from extracao.compute_kappa import compute_kappa_report

        haiku = ["application", "helper", "extender", "helper"]
        manual = ["application", "helper", "application", "helper"]
        result = compute_kappa_report(haiku_labels=haiku, manual_labels=manual)
        assert "kappa" in result
        assert "classification_report" in result
        assert "confusion_matrix" in result
        assert isinstance(result["kappa"], float)

    def test_export_uncertain_creates_file(self, tmp_path, sample_gold_csv):
        from extracao.compute_kappa import load_gold, export_uncertain_for_review

        gold_df = load_gold(sample_gold_csv)
        # Force all to be uncertain
        gold_df["confidence"] = 0.5
        output = tmp_path / "uncertain.csv"
        export_uncertain_for_review(gold_df, threshold=0.7, output=output)
        assert output.exists()
        exported = pd.read_csv(output)
        assert "manual_label" in exported.columns
