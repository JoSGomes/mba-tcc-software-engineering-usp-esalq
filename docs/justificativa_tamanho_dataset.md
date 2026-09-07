# Justificativa do Tamanho do Dataset

> **Versão**: v1 — 2026-05-06
> Fundamentação para a escolha de 10.000–50.000 repositórios como alvo de coleta.

---

## 1. Contexto

Este trabalho treina e avalia modelos de classificação supervisionada de repositórios OSS em três classes. A escolha do tamanho do dataset afeta diretamente:

- **Capacidade estatística**: datasets maiores reduzem variância nas estimativas de performance
- **Representatividade**: cobertura das três classes e subtipos dentro de cada classe
- **Viabilidade**: limite de tempo de coleta, custo de rotulação e capacidade computacional local

---

## 2. Referência Principal: PROMISE'24

O estudo [PROMISE'24] [TODO: citar autores e título completo] utilizou [TODO: N] repositórios em sua avaliação. Este trabalho expande o dataset em relação ao original por dois motivos:

1. **Robustez estatística**: o aumento do conjunto de treino em datasets balanceados tipicamente melhora generalização até o ponto de saturação [TODO: citar curva de aprendizagem — ex. Banko & Brill 2001 ou estudo específico de OSS Mining]
2. **Representatividade das classes**: a classe Extender tende a ser sub-representada em amostragens aleatórias; um dataset maior aumenta a probabilidade de inclusão natural de exemplos de cada classe

---

## 3. Evidência da Literatura

### 3.1 Estudos com datasets de ordem similar

| Estudo | N repos | Resultado |
|--------|---------|-----------|
| [PROMISE'24] | [TODO] | [TODO métricas] |
| [TODO: estudo 2 do SLR] | [TODO] | [TODO] |
| [TODO: estudo 3 do SLR] | [TODO] | [TODO] |
| Munaiah et al. (2017) | ~1.8M (triagem) → ~4.5k curados | — (estudo de engenharia, não ML) |

[TODO: descrever tendência geral — datasets de 10k–100k são comuns em ML aplicado a repositórios OSS]

### 3.2 Sobre o limite superior (50.000 repos)

O limite superior de 50.000 repositórios é motivado por:

- **Custo de rotulação**: o gold standard requer rotulação semi-automática com LLM (Haiku) + revisão manual de ~50–80 casos incertos. Para amostras maiores que 500 a 1.000 repositórios, o custo de rotulação humana cresce linearmente.
- **Tempo de coleta**: a API SEART GHS opera com rate-limiting; 50k repositórios equivalem a aproximadamente [TODO: estimar N horas de coleta com 0.5s de delay por página de 100 repos]
- **Capacidade computacional**: treinamento e avaliação de modelos em 50k amostras é viável em CPU em poucas horas para as técnicas selecionadas

---

## 4. Critérios de Filtragem

Para garantir qualidade, apenas repositórios que atendam **todos** os critérios abaixo serão incluídos:

| Critério | Valor | Justificativa |
|----------|-------|---------------|
| Mínimo de estrelas | ≥ 10 | Indica algum nível de uso/relevância; filtra repositórios abandonados e experimentos privados tornados públicos |
| Possui README | Sim | Necessário para feature textual; repositórios sem README são inutilizáveis como entrada do modelo |
| Linguagem principal | Qualquer | Sem filtro de linguagem para maximizar diversidade; linguagem como feature estrutural |
| Data de criação | [TODO: definir range] | [TODO: justificar — ex. após 2015 para maior cobertura de boas práticas] |

### 4.1 Justificativa do threshold de estrelas

[TODO: referenciar estudos que usam filtros de estrelas similares, ex. Cosentino et al., Borges et al.]

Repositórios com < 10 estrelas são predominantemente projetos pessoais, tutoriais e experimentos — categorias que não refletem o ecossistema OSS relevante para reutilização. O threshold de 10 estrelas é amplamente utilizado na literatura [TODO: verificar e citar].

---

## 5. Estimativa de Distribuição de Classes

Com base em [PROMISE'24] e na natureza do ecossistema GitHub:

| Classe | Estimativa % | Bases |
|--------|-------------|-------|
| Helper | ~50–60% | Maioria dos repositórios GitHub são bibliotecas/utilitários |
| Application | ~25–35% | Aplicações completas são menos frequentes |
| Extender | ~10–20% | Plugins/extensões são minoritários mas bem definidos |

[TODO: validar com dados reais após coleta — comparar com distribuição de PROMISE'24]

O desbalanceamento moderado é tratado com F1-macro como métrica principal (peso igual por classe).

---

## 6. Conclusão

O alvo de **10.000–50.000 repositórios** é justificado por:

1. Comparabilidade e expansão do PROMISE'24 [referência]
2. Viabilidade de coleta via SEART GHS (sem acesso WoC)
3. Suficiência estatística para as técnicas de ML selecionadas
4. Custo de rotulação semi-automática controlável com Haiku

O tamanho final dependerá da disponibilidade da API SEART GHS no momento da coleta e dos critérios de filtragem aplicados. O piso mínimo aceitável é **10.000 repositórios** com todos os campos obrigatórios preenchidos.
