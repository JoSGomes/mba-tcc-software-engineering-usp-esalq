# Taxonomia Operacional — Application / Helper / Extender

> **Versão**: v1 — 2026-05-06
> Definição precisa das três classes para uso na rotulação do gold standard.
> Derivada da taxonomia PROMISE'24 com critérios operacionais adicionais para reduzir ambiguidade.

---

## Definições

### Application

**Definição**: Repositório que implementa uma solução completa e utilizável diretamente pelo usuário final para resolver um problema de domínio específico, sem depender de outro sistema de software para fornecer sua funcionalidade principal.

**Características-chave**:
- Possui ponto de entrada definido (executável, interface gráfica, CLI, serviço web)
- O valor entregue está na funcionalidade do domínio, não na infraestrutura técnica
- Pode ser instalado e usado independentemente por um usuário final não-desenvolvedor

**Exemplos positivos**:
- Editor de texto (VS Code, Notepad++)
- Jogo (qualquer jogo completo)
- CMS / Blog engine (WordPress)
- Cliente de email
- Ferramenta de linha de comando autossuficiente (ex: `youtube-dl`, `ffmpeg`)
- Dashboard / ferramenta de monitoramento com UI própria

**Exemplos negativos** (NÃO é Application):
- Um bot de Discord que depende da API Discord → **Extender**
- Uma biblioteca de parsing CSV → **Helper**
- Um plugin para VS Code → **Extender**

---

### Helper

**Definição**: Repositório que provê funcionalidades técnicas reutilizáveis para outros desenvolvedores, sem constituir uma aplicação independente. Resolve um problema técnico, não um problema de domínio do usuário final.

**Características-chave**:
- Destina-se a ser importado/instalado como dependência de outros projetos
- O usuário principal é um desenvolvedor, não um usuário final
- Não depende de nenhuma plataforma ou framework específico para funcionar (genérico)

**Exemplos positivos**:
- Biblioteca de parsing (ex: `requests`, `Beautiful Soup`, `Pandas`)
- Framework web (ex: `Flask`, `Express`, `Spring`)
- SDK de propósito geral
- Utilitário de linha de comando como ferramenta de desenvolvimento (ex: `jq`, `ripgrep`)
- Biblioteca de testes (ex: `pytest`, `Jest`)
- Coleção de algoritmos ou estruturas de dados

**Exemplos negativos** (NÃO é Helper):
- Uma biblioteca específica para um framework → **Extender** (ex: plugin para Django)
- Uma aplicação completa → **Application**

**Caso especial**: SDKs para APIs de terceiros (ex: AWS SDK, Stripe SDK) → classificar como **Helper** se genérico o suficiente para ser usado em múltiplos contextos; **Extender** se depende exclusivamente de uma plataforma.

---

### Extender

**Definição**: Repositório que amplia, personaliza ou adiciona funcionalidades a um sistema ou plataforma específica, dependendo fundamentalmente desse sistema para operar.

**Características-chave**:
- Depende de um host system específico para funcionar (plugin, extensão, tema, módulo)
- O escopo é limitado ao contexto do sistema hospedeiro
- Não funciona de forma autônoma fora do sistema ao qual se destina

**Exemplos positivos**:
- Plugin para VS Code, IntelliJ, Vim
- Extensão para Chrome / Firefox
- Tema para WordPress, Ghost, Jekyll
- Módulo/pacote para um framework específico (ex: `django-rest-framework`, `express-middleware`)
- Bot de Discord / Slack / Telegram
- Action para GitHub Actions
- Chart para Helm (Kubernetes)

**Exemplos negativos** (NÃO é Extender):
- Um SDK para AWS com casos de uso independentes → **Helper**
- Uma aplicação completa que usa Discord como notificação secundária → **Application**

---

## Critérios de Desempate

Quando o repositório é ambíguo, aplicar os critérios nesta ordem:

1. **Critério de dependência**: o repositório precisa de um sistema hospedeiro específico para funcionar? Se sim → **Extender**. Se funciona de forma independente → passa para critério 2.

2. **Critério de usuário**: o usuário primário é um desenvolvedor que integra o componente, ou um usuário final que usa o produto? Desenvolvedor → **Helper**; Usuário final → **Application**.

3. **Critério do README**: o README instrui a fazer `import`/`require`/`pip install` como dependência? → **Helper**. Instrui a executar ou instalar como produto final? → **Application**. Instrui a instalar dentro de outro sistema? → **Extender**.

4. **Critério de dúvida**: se após os 3 critérios acima ainda não for possível determinar com ≥ 70% de confiança → marcar como `incerto` para revisão manual.

---

## Casos Especiais Documentados

| Tipo de repositório | Classificação | Justificativa |
|--------------------|---------------|---------------|
| Boilerplate / Template | Helper | Destina-se a ser copiado/adaptado por outros devs |
| Tutorial / Curso (código de exemplo) | Helper | Não é produto final; é recurso de aprendizagem |
| Dotfiles / Configurações pessoais | Application | Uso pessoal direto, não é componente reutilizável |
| CLI tool genérico (jq, ripgrep) | Helper | Ferramenta de desenvolvimento, não resolve domínio |
| CLI tool de domínio (youtube-dl) | Application | Resolve problema do usuário final diretamente |
| SDK de plataforma fechada (AWS, GCP SDK) | Helper | Reutilizável em múltiplos contextos |
| SDK de produto único (Discord.py) | Extender | Depende exclusivamente de uma plataforma |
| Framework com CLI própria (Django) | Helper | Destina-se a ser base de outros projetos |
| Aplicação criada com Django | Application | Produto final que usa Django como dep |

---

## Uso na Rotulação Haiku

O modelo Haiku receberá esta taxonomia como contexto e o seguinte prompt para cada repositório:

```
Você é um classificador de repositórios GitHub.

TAXONOMIA:
- Application: aplicação completa para usuário final; funciona de forma independente
- Helper: biblioteca/utilitário para desenvolvedores; importado como dependência
- Extender: plugin/extensão que depende de um sistema hospedeiro específico

REPOSITÓRIO:
Nome: {name}
Descrição: {description}
README (primeiros 300 tokens): {readme_snippet}
Linguagem: {language}
Estrelas: {stars}

Responda APENAS com um JSON:
{"label": "application"|"helper"|"extender", "confidence": 0.0-1.0, "reasoning": "uma frase"}

Se a confiança for < 0.7, use "incerto" como label para revisão manual.
```
