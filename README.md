# Classificação de Repositórios Open Source com Machine Learning

Código, dados e resultados do TCC de MBA em Engenharia de Software (USP/Esalq)
*"Classificação de Repositórios Open Source para Apoio à Reutilização de
Software: Uma Abordagem sobre o SEART-GHS"*.

**Objetivo**: classificar repositórios open source do GitHub em três
categorias funcionais — *Application*, *Helper*, *Extender* — a partir de
metadados e texto (nome + README), comparando três representações textuais
(Doc2Vec, SBERT, CodeBERT) e três classificadores (Random Forest, Regressão
Logística, XGBoost).

**Resultado principal**: SBERT + XGBoost, F1-macro = 0,633 ± 0,023 (validação
cruzada aninhada), estatisticamente superior a Doc2Vec (0,597) e CodeBERT
(0,579). Qualidade da rotulagem automática validada por kappa de Cohen =
0,8039 (concordância substancial) sobre 100 repositórios revisados
manualmente.

Este repositório é um recorte enxuto, focado em reprodutibilidade — para
executar o pipeline do zero e regenerar todos os resultados e figuras.

---

## Taxonomia

| Classe | Definição |
|--------|-----------|
| `application` | Produto executado diretamente pelo usuário final; funciona standalone |
| `helper` | Biblioteca/utilitário importado como dependência por outros desenvolvedores |
| `extender` | Plugin/extensão que depende de um host específico para funcionar |
| `other` | Não é software executável: listas curadas, tutoriais, datasets |

`other` e repositórios de baixa confiança (`incerto`) ficam fora do
treino/validação/teste supervisionados e são usados só na avaliação
out-of-distribution (OOD).

---

## Estrutura do repositório

```
.
├── extracao/          # Coleta de dados (SEART GHS), rotulagem (Claude Haiku), kappa
├── preparacao/        # Limpeza, split, geração de embeddings (Doc2Vec/SBERT/CodeBERT)
├── modelagem/         # Treino e comparação de classificadores
├── avaliacao/         # Avaliação out-of-distribution (OOD)
├── docs/              # Schema de features e taxonomia operacional
├── data/
│   ├── raw/           # Dados brutos coletados (repos.parquet, labels.parquet, kappa)
│   └── processed/     # Splits, embeddings Doc2Vec e y_*.npy (ver nota de tamanho abaixo)
├── notebooks/         # Análise exploratória (jupytext .py)
├── experiments/       # Registro de experimentos, métricas e figuras
├── config/            # Seed global (GLOBAL_SEED = 42)
└── tests/             # Testes automatizados
```

---

## Instalação

Requer Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Embeddings SBERT/CodeBERT (opcional, requer `torch`)

Para GPU (recomendado, acelera bastante a geração de embeddings):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"  # deve imprimir True
pip install -e ".[dev,embeddings]"
```

Sem GPU, `sentence-transformers`/`transformers` funcionam em CPU normalmente
(mais lento) — basta instalar `torch` sem `--index-url` e seguir com o mesmo
`pip install -e ".[dev,embeddings]"`.

### Rodar os testes

```bash
pytest
```

---

## Nota sobre o que está incluído neste repositório

Os rótulos e artefatos de rotulagem (`data/raw/*.csv`, `labels.parquet`,
`kappa_*`), os splits e os embeddings Doc2Vec (`data/processed/`), o
registro de experimentos e todas as figuras (`experiments/`) estão
versionados aqui.

**Não estão incluídos** (grandes e regeneráveis por script, listados abaixo
com o comando que os recria):

| Arquivo | Tamanho aprox. | Como regenerar |
|---|---|---|
| `data/raw/repos.parquet` | 17 MB | `python extracao/run_collection.py --max-repos 15000` |
| `data/processed/dataset_clean.parquet` | 33 MB | `python preparacao/clean.py` |
| `data/processed/doc2vec_model.bin` | 54 MB | `python preparacao/features_doc2vec.py` |
| `data/processed/X_sbert_*.npy` | ~23 MB | `python preparacao/build_features.py --feature-sets sbert` |
| `data/processed/X_codebert_*.npy` | ~44 MB | `python preparacao/build_features.py --feature-sets codebert` |

`repos.parquet` é o único que não é bit-a-bit reprodutível: ele coleta
estatísticas ao vivo da API SEART GHS/GitHub (estrelas, forks etc.), que
mudam com o tempo. Os demais arquivos são determinísticos (mesma seed,
mesma entrada).

⚠️ **Aviso de tamanho**: mesmo sem esses arquivos, o repositório é grande
(~200 MB) por causa das figuras em `experiments/figures/` (`.tiff`, alta
resolução, formato exigido pelas normas de TCC da instituição). Um `git
clone` pode demorar dependendo da conexão.

---

## Passo a passo — reproduzindo tudo do zero

Cada etapa lê a saída da anterior. Os artefatos já commitados (splits,
Doc2Vec, `registry.csv`, etc.) permitem pular direto para qualquer etapa
intermediária sem rodar as anteriores.

### 1. Coleta de dados

Requer uma variável de ambiente `GITHUB_TOKEN` (personal access token do
GitHub) para `fetch_readmes.py` — ou passar `--token` diretamente.

```bash
python extracao/run_collection.py --max-repos 15000
export GITHUB_TOKEN=ghp_...  # Windows: set GITHUB_TOKEN=ghp_...
python extracao/fetch_readmes.py --input data/raw/repos.parquet
```

### 2. Rotulagem automática

Requer a Claude Code CLI (`claude`) instalada e autenticada — o script chama
`claude -p --model claude-haiku-4-5-20251001` via subprocesso.

```bash
python extracao/gold_standard.py --all --batch-size 50
python extracao/compute_kappa.py --gold data/raw/gold_standard.csv --review data/raw/kappa_review_combined.csv
```

### 3. Limpeza e split

```bash
python preparacao/clean.py
python preparacao/split.py
```

### 4. Geração de embeddings

```bash
python preparacao/build_features.py --feature-sets doc2vec sbert codebert
```

### 5. Comparação entre representações e classificadores

```bash
# Etapa de seleção: Grid Search nas 9 combinações representação x classificador
python modelagem/compare_models.py --phase a

# Etapa de comparação: validação cruzada aninhada entre as 3 representações
python modelagem/compare_models.py --phase b
```

Ou, para rodar as duas etapas (mais a geração de embeddings, se necessário)
em uma única chamada:

```bash
python modelagem/run_pipeline.py
```

### 6. Baseline Doc2Vec + Random Forest (opcional, referência histórica)

```bash
python modelagem/baseline.py
```

### 7. Avaliação out-of-distribution

```bash
python avaliacao/ood_analysis.py
```

Saídas: `experiments/ood_analysis_result.json` e
`experiments/figures/ood_confidence_histogram.tiff`.

---

## Reprodutibilidade

- Seed global fixa em `config/seeds.py` (`GLOBAL_SEED = 42`), usada em todo
  estimador scikit-learn, no split e no treino do Doc2Vec.
- Preprocessadores (Doc2Vec) ajustados **somente** no split de treino; SBERT
  e CodeBERT são encoders pré-treinados e congelados, sem etapa de fit —
  aplicados igualmente a todos os splits.
- Splits estratificados 70/15/15 com índices fixos em
  `data/processed/splits.json` — reprodutível via
  `pytest tests/test_preparation.py::TestSplitReproducibility`.

---

## Resultados principais

| Representação (classificador) | F1-macro (média ± DP) | Acurácia | F1 extender |
|---|---|---|---|
| Doc2Vec (XGBoost) | 0,597 ± 0,011 | 0,737 ± 0,009 | 0,273 ± 0,026 |
| **SBERT (XGBoost)** | **0,633 ± 0,023** | 0,740 ± 0,011 | **0,395 ± 0,070** |
| CodeBERT (Regressão Logística) | 0,579 ± 0,019 | 0,689 ± 0,017 | 0,278 ± 0,040 |

Teste de Wilcoxon de postos sinalizados (α = 0,05): SBERT superior a Doc2Vec
e a CodeBERT em todas as comparações pareadas (p < 0,05, delta de Cliff
grande).

Kappa de Cohen entre rotulagem automática (Claude Haiku) e revisão manual:
K = 0,8039 (concordância substancial) sobre 100 repositórios; a classe
`extender` é sistematicamente subestimada pelo modelo de linguagem.

Detalhes completos, discussão e limitações: ver o TCC final (não incluído
neste repositório — código e dados de reprodução apenas).

---

## Licença

MIT — ver `LICENSE`.
