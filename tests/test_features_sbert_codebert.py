"""Testes dos feature builders SBERT e CodeBERT.

Testes rápidos (padrão do `pytest`) usam monkeypatch para não baixar nem
rodar o modelo real. Testes marcados `@pytest.mark.slow` baixam e rodam o
modelo de verdade — gated por `RUN_SLOW_MODEL_TESTS=1`, não rodam por padrão.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

RUN_SLOW = os.environ.get("RUN_SLOW_MODEL_TESTS") == "1"
skip_slow = pytest.mark.skipif(
    not RUN_SLOW, reason="defina RUN_SLOW_MODEL_TESTS=1 para baixar e rodar o modelo real"
)


@pytest.fixture()
def tiny_df():
    n = 12
    return pd.DataFrame(
        {
            "repo_id": [f"org{i}/repo{i}" for i in range(n)],
            "name": [f"repo{i}" for i in range(n)],
            "readme_text": [f"This is a test repository number {i} for embeddings." for i in range(n)],
        }
    )


@pytest.fixture()
def tiny_splits():
    ids = [f"org{i}/repo{i}" for i in range(12)]
    return {
        "train": ids[0:6],
        "val": ids[6:9],
        "test_indist": ids[9:11],
        "test_ood": ids[11:12],
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# SBERT — testes rápidos (monkeypatch)
# ---------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Stub determinístico — evita baixar/rodar o modelo real nos testes rápidos."""

    VECTOR_SIZE = 384

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    def encode(self, texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False):
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), self.VECTOR_SIZE)).astype(np.float32)


class TestSBERTFast:
    def test_shapes_consistent_including_test_ood(self, tiny_df, tiny_splits, tmp_path, monkeypatch):
        sentence_transformers = pytest.importorskip("sentence_transformers")
        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)

        from preparacao import features_sbert
        results = features_sbert.build_sbert(tiny_df, tiny_splits, output_dir=tmp_path, device="cpu")

        for split_name, ids in tiny_splits.items():
            if split_name == "seed":
                continue
            assert results[split_name].shape == (len(ids), features_sbert.VECTOR_SIZE), (
                f"shape inconsistente em {split_name}"
            )
            assert (tmp_path / f"X_sbert_{split_name}.npy").exists()
        # test_ood recebe X (sem risco de leakage — encoder congelado), diferente
        # de y_test_ood.npy (nunca gerado — não supervisionado, ver build_features.py)
        assert "test_ood" in results

    def test_batching_covers_all_indices(self, tiny_df, tiny_splits, tmp_path, monkeypatch):
        sentence_transformers = pytest.importorskip("sentence_transformers")
        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)

        from preparacao import features_sbert
        results = features_sbert.build_sbert(
            tiny_df, tiny_splits, output_dir=tmp_path, batch_size=2, device="cpu"
        )
        assert results["train"].shape[0] == len(tiny_splits["train"])


# ---------------------------------------------------------------------------
# SBERT — teste lento (modelo real), opcional
# ---------------------------------------------------------------------------

@pytest.mark.slow
@skip_slow
def test_sbert_real_model_deterministic(tmp_path):
    from preparacao.features_sbert import SBERT_MODEL_NAME
    from preparacao.text_utils import resolve_device
    from sentence_transformers import SentenceTransformer

    device = resolve_device(None)
    print(f"\n[slow test] device resolvido: {device}")
    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    text = "This is a helper library imported as a dependency."

    v1 = model.encode([text], convert_to_numpy=True)
    v2 = model.encode([text], convert_to_numpy=True)
    assert np.allclose(v1, v2, atol=1e-4), "encoder não determinístico entre chamadas"
    assert v1.shape == (1, 384)


# ---------------------------------------------------------------------------
# CodeBERT — testes rápidos (monkeypatch)
# ---------------------------------------------------------------------------

class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeCodeBERTOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeCodeBERTTokenizer:
    """Stub determinístico — evita baixar o tokenizer real nos testes rápidos."""

    SEQ_LEN = 5

    def __call__(self, texts, truncation=True, max_length=512, padding=True, return_tensors="pt"):
        import torch

        n = len(texts)
        input_ids = torch.zeros((n, self.SEQ_LEN), dtype=torch.long)
        attention_mask = torch.ones((n, self.SEQ_LEN), dtype=torch.long)
        return _FakeBatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    @classmethod
    def from_pretrained(cls, name):
        return cls()


class FakeCodeBERTModel:
    """Stub determinístico — evita baixar/rodar o modelo real nos testes rápidos."""

    VECTOR_SIZE = 768

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        import torch

        n, seq_len = input_ids.shape
        gen = torch.Generator().manual_seed(42)
        hidden = torch.randn(n, seq_len, self.VECTOR_SIZE, generator=gen)
        return _FakeCodeBERTOutput(hidden)

    @classmethod
    def from_pretrained(cls, name):
        return cls()


def _patch_codebert(monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeCodeBERTTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", FakeCodeBERTModel)
    return transformers


class TestCodeBERTFast:
    def test_shapes_consistent_including_test_ood(self, tiny_df, tiny_splits, tmp_path, monkeypatch):
        _patch_codebert(monkeypatch)

        from preparacao import features_codebert
        results = features_codebert.build_codebert(tiny_df, tiny_splits, output_dir=tmp_path, device="cpu")

        for split_name, ids in tiny_splits.items():
            if split_name == "seed":
                continue
            assert results[split_name].shape == (len(ids), features_codebert.VECTOR_SIZE), (
                f"shape inconsistente em {split_name}"
            )
            assert (tmp_path / f"X_codebert_{split_name}.npy").exists()
        assert "test_ood" in results

    def test_batching_covers_all_indices(self, tiny_df, tiny_splits, tmp_path, monkeypatch):
        _patch_codebert(monkeypatch)

        from preparacao import features_codebert
        results = features_codebert.build_codebert(
            tiny_df, tiny_splits, output_dir=tmp_path, batch_size=2, device="cpu"
        )
        assert results["train"].shape[0] == len(tiny_splits["train"])

    def test_oom_triggers_batch_size_fallback(self, monkeypatch):
        """Simula OOM no primeiro forward (batch_size>1); a função deve reduzir
        batch_size pela metade e reprocessar, sem perder nenhum índice."""
        _patch_codebert(monkeypatch)

        from preparacao import features_codebert

        tokenizer = FakeCodeBERTTokenizer()

        class OOMOnceModel(FakeCodeBERTModel):
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids=None, attention_mask=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("CUDA out of memory.")
                return super().__call__(input_ids=input_ids, attention_mask=attention_mask)

        oom_model = OOMOnceModel()
        result = features_codebert._encode_texts(
            ["texto a", "texto b", "texto c", "texto d"], tokenizer, oom_model, "cpu", batch_size=4
        )
        assert result.shape == (4, features_codebert.VECTOR_SIZE)
        assert oom_model.calls >= 2, "esperava retry apos reduzir batch_size"


# ---------------------------------------------------------------------------
# CodeBERT — teste lento (modelo real), opcional
# ---------------------------------------------------------------------------

@pytest.mark.slow
@skip_slow
def test_codebert_real_model_deterministic_and_eval_mode():
    import torch

    from preparacao.features_codebert import CODEBERT_MODEL_NAME, _encode_texts
    from preparacao.text_utils import resolve_device
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(None)
    print(f"\n[slow test] device resolvido: {device}")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(CODEBERT_MODEL_NAME).to(device)
    model.eval()
    assert not model.training, "model.eval() obrigatorio -- dropout ativo quebraria determinismo"

    texts = ["def foo(): return 1", "A helper library imported as a dependency."]
    v1 = _encode_texts(texts, tokenizer, model, device, batch_size=2)
    v2 = _encode_texts(texts, tokenizer, model, device, batch_size=2)
    assert np.allclose(v1, v2, atol=1e-4), "encoder nao deterministico entre chamadas"
    assert v1.shape == (2, 768)
