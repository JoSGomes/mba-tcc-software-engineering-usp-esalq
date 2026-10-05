# Classificação de Repositórios Open Source com Machine Learning

Código, dados e resultados do TCC de MBA em Engenharia de Software (USP/Esalq)
*"Classificação de repositórios open source para apoio à reutilização de
software via SEART-GHS"*.

**Objetivo**: classificar repositórios open source do GitHub em três
categorias funcionais (*application*, *helper* e *extender*) a partir do
nome e do README, comparando três representações textuais (Doc2Vec, SBERT e
CodeBERT) e três classificadores (Random Forest, Regressão Logística e
XGBoost).

**Resultado principal**: SBERT + XGBoost obteve o maior F1-macro médio na
validação cruzada aninhada (0,633 ± 0,023), superior ao Doc2Vec (0,582) e
ao CodeBERT (0,579) em todos os dez folds externos (teste t com
reamostragem corrigido, p < 0,01). A classe *extender* permaneceu a mais
difícil (F1 = 0,395).

Este repositório é um recorte enxuto, focado em reprodutibilidade: permite
executar o pipeline do zero e regenerar todos os resultados e figuras.

---

## Taxonomia

| Classe | Definição |
|--------|-----------|
| `application` | Produto executado diretamente pelo usuário final; funciona standalone |
| `helper` | Biblioteca/utilitário importado como dependência por outros desenvolvedores |
| `extender` | Plugin/extensão que depende de um host específico para funcionar |
| `other` | Não é software executável: listas curadas, tutoriais, datasets |

Os repositórios rotulados como `other` ou `incerto` (baixa confiança do
modelo de linguagem) ficam fora do treino e formam o conjunto de **classes
excluídas do treino** (split `test_ood`), usado só na avaliação final.

---

## Estrutura do repositório

```
.
├── extracao/          # Coleta (SEART-GHS), rotulagem (Claude Haiku), kappa e revisão manual
├── preparacao/        # Limpeza, split, embeddings (Doc2Vec/SBERT/CodeBERT)
├── modelagem/         # Classificadores, validação cruzada aninhada, testes estatísticos, figuras
├── avaliacao/         # Avaliação nas classes excluídas do treino e calibração
├── docs/              # Schema de features e taxonomia operacional
├── data/
│   ├── raw/           # Rótulos, amostras de revisão manual
│   └── processed/     # Splits, embeddings Doc2Vec e y_*.npy
├── notebooks/         # Análise exploratória (jupytext .py)
├── experiments/       # Resultados (CSV/JSON) e figuras
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

Sem GPU, `sentence-transformers`/`transformers` funcionam em CPU (mais
lento): basta instalar `torch` sem `--index-url` e seguir com o mesmo
`pip install -e ".[dev,embeddings]"`.

### Rodar os testes

```bash
pytest
python modelagem/smoke_nested_cv.py                  # ~1 min, dados sintéticos
python modelagem/smoke_nested_cv_full_selection.py   # ~1-2 min, dados sintéticos
```

---

## O que está incluído

Rótulos e artefatos de rotulagem (`data/raw/`), splits e embeddings Doc2Vec
(`data/processed/`), todos os resultados usados no TCC e as figuras
(`experiments/`) estão versionados.

**Não estão incluídos** (grandes e regeneráveis por script):

| Arquivo | Tamanho aprox. | Como regenerar |
|---|---|---|
| `data/raw/repos.parquet` | 17 MB | `python extracao/run_collection.py --max-repos 15000` |
| `data/processed/dataset_clean.parquet` | 33 MB | `python preparacao/clean.py` |
| `data/processed/doc2vec_model.bin` | 54 MB | `python preparacao/features_doc2vec.py` |
| `data/processed/X_sbert_*.npy` | ~23 MB | `python preparacao/build_features.py --feature-sets sbert` |
| `data/processed/X_codebert_*.npy` | ~44 MB | `python preparacao/build_features.py --feature-sets codebert` |

`repos.parquet` é o único que não é reprodutível bit a bit: ele coleta
estatísticas ao vivo da API SEART-GHS/GitHub (estrelas, forks etc.), que
mudam com o tempo. Os demais arquivos são determinísticos (mesma seed, mesma
entrada).

⚠️ **Aviso de tamanho**: as figuras em `experiments/figures/` estão em
`.tiff` de alta resolução (formato exigido pelas normas de TCC da
instituição); um `git clone` pode demorar dependendo da conexão.

---

## Passo a passo — reproduzindo tudo do zero

Cada etapa lê a saída da anterior. Os artefatos já versionados permitem
começar de qualquer etapa intermediária.

### 1. Coleta de dados

Requer a variável de ambiente `GITHUB_TOKEN` (personal access token do
GitHub) para `fetch_readmes.py`, ou o argumento `--token`.

```bash
python extracao/run_collection.py --max-repos 15000
export GITHUB_TOKEN=...  # Windows: set GITHUB_TOKEN=...
python extracao/fetch_readmes.py --input data/raw/repos.parquet
```

### 2. Rotulagem automática e validação

Requer a Claude Code CLI (`claude`) instalada e autenticada: o script chama
`claude -p --model claude-haiku-4-5-20251001` via subprocesso. Os primeiros
500 repositórios foram rotulados individualmente (1.000 caracteres do
README) e o restante em lotes de 50 (500 caracteres cada).

```bash
python extracao/gold_standard.py --all --batch-size 50
python extracao/compute_kappa.py --gold data/raw/gold_standard.csv --review data/raw/kappa_review_combined.csv
```

### 3. Limpeza, split e embeddings

```bash
python preparacao/clean.py
python preparacao/split.py
python preparacao/build_features.py --feature-sets doc2vec sbert codebert
```

### 4. Random Forest sobre o Doc2Vec (estratégias de ajuste)

Padrão, Grid Search e SMOTE + Grid Search, no split fixo e em validação
cruzada aninhada 10x5.

```bash
python modelagem/baseline.py
```

### 5. Comparação entre as representações

Validação cruzada aninhada 10x5 em que, **em cada fold externo**, o Doc2Vec é
reajustado só com os textos de treino do fold e a escolha entre RF,
Regressão Logística e XGBoost (com seus hiperparâmetros) é feita no laço
interno. Nenhum dado de teste do fold externo participa de qualquer decisão.
Leva ~15 h em CPU; cada fold é salvo ao terminar, e a execução pode ser
retomada (`--max-folds N` limita os folds novos por execução).

```bash
python modelagem/nested_cv_full_selection.py
python modelagem/comparacoes_estatisticas.py   # teste t com reamostragem corrigido
```

### 6. Classes excluídas do treino, revisão manual e calibração

```bash
python avaliacao/ood_analysis.py               # modelo final + confiança por repositório
python extracao/sample_revisao_manual.py       # confere o sorteio das amostras revisadas
python extracao/compute_revisao_manual.py      # kappa por procedimento, revisão manual, ECE
```

### 7. Figuras

```bash
python modelagem/figura_fluxo_nested_cv.py
python modelagem/figura_matrizes_rf.py
python modelagem/figura_matrizes_confusao.py
python avaliacao/figura_confianca_calibracao.py
python modelagem/figura_visao_geral.py
```

As etapas 5 a 7 (e a geração de embeddings) podem ser encadeadas com
`python modelagem/run_pipeline.py` (`--skip-nested-cv` reaproveita os
resultados do nested CV já salvos).

---

## Onde está cada resultado do TCC

| No TCC | Arquivo | Gerado por |
|---|---|---|
| Tabelas 4 e 5 (RF no split fixo) | `experiments/registry.csv` | `modelagem/baseline.py` |
| Figuras 2 a 4 (matrizes do RF no teste) | `experiments/figures/cm_rf_{d2ac1ac9,018ed2b3,50fa0ee9}_compacta.tiff` | `modelagem/figura_matrizes_rf.py` |
| Tabela 6 (RF na validação cruzada aninhada) | `experiments/nested_cv_{scores,summary}.csv` | `modelagem/baseline.py` |
| Tabelas 7 e 10 (comparações pareadas) | `experiments/comparacoes_estatisticas.csv` | `modelagem/comparacoes_estatisticas.py` |
| Tabela 8 (laço interno e escolhas por rodada) | `experiments/nested_cv_full_selection_{scores,choices}.csv` | `modelagem/nested_cv_full_selection.py` |
| Tabela 9 (desempenho das representações) | `experiments/nested_cv_full_selection_summary.csv` | `modelagem/nested_cv_full_selection.py` |
| Figura 1 (procedimento de validação) | `experiments/figures/fluxo_nested_cv.tiff` | `modelagem/figura_fluxo_nested_cv.py` |
| Figura 5 (matrizes de confusão) | `experiments/figures/matrizes_confusao_representacoes.tiff` | `modelagem/figura_matrizes_confusao.py` |
| Classes excluídas do treino (taxa de rejeição) | `experiments/ood_analysis_result.json`, `ood_confidences.csv` | `avaliacao/ood_analysis.py` |
| Kappa por procedimento, revisão manual, ECE e falsos rejeitados; Apêndice B | `experiments/revisao_manual_result.json` | `extracao/compute_revisao_manual.py` |
| Figura 6 (confiança e calibração) | `experiments/figures/confianca_calibracao.tiff` | `avaliacao/figura_confianca_calibracao.py` |
| Apêndice A (etapas × CRISP-DM) | `experiments/figures/visao_geral_trabalho.tiff` | `modelagem/figura_visao_geral.py` |

---

## Reprodutibilidade

- Seed global fixa em `config/seeds.py` (`GLOBAL_SEED = 42`), usada em todo
  estimador scikit-learn, no split, no SMOTE e no treino do Doc2Vec.
- Na comparação entre representações, o Doc2Vec é ajustado **dentro de cada
  fold externo**, só com os textos de treino do fold; SBERT e CodeBERT são
  encoders pré-treinados e congelados, sem etapa de ajuste.
- O SMOTE fica dentro do pipeline avaliado, aplicado só às partes de treino.
- Splits estratificados 70/15/15 com índices fixos em
  `data/processed/splits.json`.

---

## Resultados principais

Validação cruzada aninhada 10x5 (média ± desvio padrão nos 10 folds
externos), com o classificador escolhido no laço interno de cada rodada:

| Representação (classificador escolhido) | F1-macro | Acurácia | F1 extender |
|---|---|---|---|
| Doc2Vec (XGBoost) | 0,582 ± 0,023 | 0,740 ± 0,007 | 0,230 ± 0,066 |
| **SBERT (XGBoost)** | **0,633 ± 0,023** | 0,740 ± 0,011 | **0,395 ± 0,070** |
| CodeBERT (Regressão Logística) | 0,579 ± 0,019 | 0,689 ± 0,017 | 0,278 ± 0,040 |

Teste t com reamostragem corrigido (Nadeau e Bengio, 2003): SBERT superou o
Doc2Vec (p = 0,007) e o CodeBERT (p = 0,003) em 10 de 10 folds; Doc2Vec e
CodeBERT não diferiram (p = 0,844).

Concordância entre a rotulagem automática e a revisão manual (kappa de
Cohen): 0,972 nos repositórios rotulados individualmente e 0,559 nos
rotulados em lote; 0,19 numa amostra de 30 rotulados em lote com confiança
inferior a 0,8. A classe `extender` é sistematicamente subestimada pelo
modelo de linguagem.

Nas classes excluídas do treino, a regra de rejeição por confiança
(τ = 0,5) atingiu 7,1% dos repositórios, praticamente a mesma proporção dos
falsos rejeitados nas classes conhecidas (6,8%), embora as probabilidades
estejam bem calibradas (ECE = 0,018).

Detalhes, discussão e limitações: ver o TCC (não incluído neste
repositório).

---

## Licença

MIT — ver `LICENSE`.
