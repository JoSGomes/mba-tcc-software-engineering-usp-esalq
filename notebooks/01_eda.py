# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 01 — Análise Exploratória de Dados (EDA)
#
# **Projeto**: Mineração e Classificação de Repositórios Open Source com Machine Learning
#
# Este notebook analisa o dataset coletado via SEART GHS e o gold standard rotulado pelo Haiku.
#
# ## Pré-requisitos
# ```bash
# python extracao/run_collection.py --max-repos 15000
# python extracao/gold_standard.py --input data/raw/repos.parquet --sample 500
# ```

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path("..").resolve()))
from config.seeds import GLOBAL_SEED  # noqa: F401

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

RAW_DIR = Path("../data/raw")
GOLD_PATH = RAW_DIR / "gold_standard.csv"
REPOS_PATH = RAW_DIR / "repos.parquet"

# %% [markdown]
# ## 1. Carregamento e Visão Geral

# %%
df = pd.read_parquet(REPOS_PATH)
print(f"Dataset: {len(df):,} repositórios × {df.shape[1]} colunas")
df.head(3)

# %%
df.dtypes

# %%
df.describe()

# %% [markdown]
# ## 2. Qualidade dos Dados

# %%
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
quality = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
quality[quality["missing_count"] > 0]

# %%
# README presence
print(f"Com README:    {df['has_readme'].sum():,} ({df['has_readme'].mean()*100:.1f}%)")
print(f"Com licença:   {df['has_license'].sum():,} ({df['has_license'].mean()*100:.1f}%)")
print(f"Com wiki:      {df['has_wiki'].sum():,} ({df['has_wiki'].mean()*100:.1f}%)")
print(f"Descrição vazia: {(df['description'] == '').sum() + df['description'].isna().sum():,}")

# %% [markdown]
# ## 3. Distribuição de Features Estruturais

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
numeric_cols = ["stars", "forks", "watchers", "commits", "contributors", "size_kb"]
for ax, col in zip(axes.flat, numeric_cols):
    log_vals = np.log1p(df[col].dropna())
    ax.hist(log_vals, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title(f"log1p({col})")
    ax.set_xlabel("Valor (escala log)")
plt.suptitle("Distribuição das Features Estruturais (escala logarítmica)", y=1.01)
plt.tight_layout()
plt.show()

# %%
# Outliers — percentis extremos
for col in ["stars", "forks", "size_kb"]:
    p95 = df[col].quantile(0.95)
    p99 = df[col].quantile(0.99)
    print(f"{col}: P95={p95:.0f}, P99={p99:.0f}, max={df[col].max():.0f}")

# %% [markdown]
# ## 4. Distribuição por Linguagem

# %%
top_languages = df["language"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(12, 5))
top_languages.plot(kind="bar", ax=ax, color="steelblue", edgecolor="none")
ax.set_title("Top 20 Linguagens de Programação")
ax.set_xlabel("Linguagem")
ax.set_ylabel("Nº de Repositórios")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %%
print(f"Repositórios sem linguagem identificada: {df['language'].isna().sum() + (df['language']=='').sum():,}")
print(f"Total de linguagens únicas: {df['language'].nunique()}")

# %% [markdown]
# ## 5. Análise Temporal

# %%
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["year_created"] = df["created_at"].dt.year

year_counts = df.groupby("year_created").size()
fig, ax = plt.subplots(figsize=(12, 4))
year_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="none")
ax.set_title("Repositórios por Ano de Criação")
ax.set_xlabel("Ano")
ax.set_ylabel("Nº de Repositórios")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Gold Standard — Distribuição de Classes

# %%
if not GOLD_PATH.exists():
    print(f"Gold standard não encontrado em {GOLD_PATH}")
    print("Execute: python extracao/gold_standard.py --input data/raw/repos.parquet --sample 500")
else:
    gold = pd.read_csv(GOLD_PATH)
    print(f"Gold standard: {len(gold):,} repositórios rotulados")
    print(f"\nDistribuição de labels:")
    print(gold["label"].value_counts())
    print(f"\nConfiança média: {gold['confidence'].mean():.3f}")
    print(f"Casos incertos (confidence < 0.7): {(gold['confidence'] < 0.7).sum()}")

# %%
if GOLD_PATH.exists():
    gold = pd.read_csv(GOLD_PATH)
    confident = gold[gold["label"] != "incerto"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    label_counts = confident["label"].value_counts()
    ax1.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2", 3))
    ax1.set_title("Distribuição de Classes (gold standard)")

    # Confidence distribution by class
    for label in ["application", "helper", "extender"]:
        subset = confident[confident["label"] == label]["confidence"]
        ax2.hist(subset, bins=20, alpha=0.6, label=label)
    ax2.set_title("Distribuição de Confiança por Classe")
    ax2.set_xlabel("Confidence Score (Haiku)")
    ax2.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Correlações entre Features Estruturais

# %%
struct_features = ["stars", "forks", "watchers", "commits", "contributors", "size_kb", "open_issues"]
corr = df[struct_features].apply(np.log1p).corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlação entre Features Estruturais (log-transformadas)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Presença de README por Faixa de Estrelas

# %%
df["stars_bucket"] = pd.cut(df["stars"], bins=[0, 10, 50, 100, 500, 1000, float("inf")],
                             labels=["10-50", "50-100", "100-500", "500-1k", "1k-5k", "5k+"])
readme_by_stars = df.groupby("stars_bucket")["has_readme"].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(readme_by_stars["stars_bucket"].astype(str), readme_by_stars["has_readme"], color="steelblue")
ax.set_title("Proporção de Repositórios com README por Faixa de Estrelas")
ax.set_ylabel("Proporção com README")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Sumário para Capítulo 2 / Metodologia

# %%
summary = {
    "total_repos": len(df),
    "languages": df["language"].nunique(),
    "top_language": df["language"].value_counts().index[0],
    "median_stars": df["stars"].median(),
    "median_forks": df["forks"].median(),
    "has_readme_pct": df["has_readme"].mean() * 100,
    "year_range": f"{df['year_created'].min():.0f}–{df['year_created'].max():.0f}",
}

print("=== Sumário do Dataset ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

if GOLD_PATH.exists():
    gold = pd.read_csv(GOLD_PATH)
    confident_gold = gold[gold["label"] != "incerto"]
    print(f"\n=== Gold Standard ===")
    print(f"  Total rotulado: {len(gold):,}")
    print(f"  Confiantes (label ≠ incerto): {len(confident_gold):,}")
    print(confident_gold["label"].value_counts().to_string())
