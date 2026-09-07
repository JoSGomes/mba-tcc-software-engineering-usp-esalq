# Feature Schema — TCC ML OSS

Schema canônico das features utilizadas na classificação de repositórios OSS.
Todas as features são extraídas do SEART GHS e/ou calculadas a partir dos metadados brutos.

---

## 1. Features Textuais — Doc2Vec

Representação densa do texto combinado de nome + README (primeiros 2000 chars).

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `doc2vec_0` … `doc2vec_99` | float32[100] | Vetor Doc2Vec de 100 dimensões (gensim, replicando PROMISE'24) |

**Parâmetros Doc2Vec**: `vector_size=100, window=5, min_count=2, epochs=40, dm=1, workers=1, seed=GLOBAL_SEED`

Modelo treinado apenas no split `train`. Splits val/test usam `infer_vector`.

---

## 1b. Features Textuais — SBERT

Representação densa via encoder de transformer pré-treinado (`sentence-transformers`).

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `sbert_0` … `sbert_383` | float32[384] | Vetor `all-MiniLM-L6-v2` (Reimers & Gurevych 2019; Wang et al. 2020) |

**Diferença de raciocínio de no-leakage vs. Doc2Vec**: o SBERT é um encoder pré-treinado e **congelado** — não há etapa de fit sobre o corpus deste projeto. A mesma transformação determinística é aplicada igualmente a todos os splits, incluindo `test_ood`, sem risco de vazamento (ao contrário do Doc2Vec, que é treinado só em `train`).

Gerado por `preparação/features_sbert.py::build_sbert`. GPU (CUDA) usada quando disponível para acelerar a inferência; `.npy` resultantes não são commitados (encoder determinista, regenerar é barato).

---

## 1c. Features Textuais — CodeBERT

Representação densa via encoder de transformer pré-treinado em código + linguagem natural.

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `codebert_0` … `codebert_767` | float32[768] | Mean pooling (ponderado por attention mask) da última camada oculta de `microsoft/codebert-base` (Feng et al. 2020) |

Mesmo raciocínio de no-leakage do SBERT (encoder congelado, sem fit). Pooling por média (não CLS) — RoBERTa/CodeBERT não tem objetivo de pré-treino tipo NSP, então o token CLS não é especificamente treinado como representação de sentença (Reimers & Gurevych, 2019).

**Limitação a documentar na discussão do TCC**: CodeBERT é pré-treinado em CodeSearchNet (código + docstring, majoritariamente inglês) — não em prosa livre de README em PT-BR/EN. Divergência de domínio esperada; o desempenho de CodeBERT vs. SBERT/Doc2Vec na comparação entre representações é o próprio teste empírico dessa hipótese, não um bloqueador técnico.

Gerado por `preparação/features_codebert.py::build_codebert`. `model.eval()` + `torch.no_grad()` obrigatórios (dropout ativo por padrão quebraria determinismo). GPU usada quando disponível, com fallback automático de `batch_size` em caso de OOM. `.npy` resultantes não são commitados.

---

## 2. Rótulo (Target)

| Campo | Tipo | Valores | Descrição |
|-------|------|---------|-----------|
| `label` | str | `application`, `helper`, `extender` | Taxonomia PROMISE'24 — single-label |
| `label_source` | str | `haiku_auto`, `manual_review` | Origem do rótulo |
| `label_confidence` | float | 0.0–1.0 | Confiança do Haiku (NaN para revisão manual) |

Codificação numérica: `0=application, 1=helper, 2=extender` (salva em `y_*.npy`).

---

## 3. Campos de Identificação (não usados como features)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `repo_id` | str | `owner/name` — identificador único |
| `collected_at` | datetime | Timestamp da coleta |
| `split` | str | `train`, `val`, `test_indist`, `test_ood` |

---

## 4. Texto Combinado (campo intermediário)

Gerado na etapa de preparação pelo `clean.py`; salvo na coluna `text` do `dataset_clean.parquet`.

```
text = f"{name} {readme_text}"
```

Usado como entrada para Doc2Vec.

---

## Notas

- Doc2Vec fitted **apenas no train split** (`preparação/features_doc2vec.py`).
- Modelo serializado em `data/processed/doc2vec_model.bin` (gensim nativo).
- SBERT (`preparação/features_sbert.py`) e CodeBERT (`preparação/features_codebert.py`) são congelados — aplicados igualmente a todos os splits, sem fit.
- Seeds derivam de `config.seeds.GLOBAL_SEED = 42`.
- `other` e `incerto` ficam em `test_ood` — não supervisionados, sem `y_test_ood.npy`.
