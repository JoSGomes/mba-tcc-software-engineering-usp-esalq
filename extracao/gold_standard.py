"""Semi-automatic gold standard labeling using Claude Haiku via claude CLI.

Usage:
    # Gold standard (500 repos, 1 por chamada — modo original):
    python extracao/gold_standard.py --sample 500

    # Dataset inteiro em batches de 50 repos por chamada (recomendado):
    python extracao/gold_standard.py --all --batch-size 50

    # Testar qualidade do batch antes do run completo (não salva):
    python extracao/gold_standard.py --all --batch-size 50 --test

    # Retomar de onde parou:
    python extracao/gold_standard.py --all --batch-size 50 --resume

Output:
    data/raw/gold_standard.csv   (sample para kappa)
    data/raw/labels.parquet      (dataset inteiro para treino)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.seeds import GLOBAL_SEED  # noqa: F401

VALID_LABELS = {"application", "helper", "extender", "other", "incerto"}
HAIKU_MODEL = "claude-haiku-4-5-20251001"
README_CHARS_SINGLE = 1000
README_CHARS_BATCH = 500  # mais curto por repo para caber 50 repos no contexto

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TAXONOMY_HEADER = """\
Você é um classificador especializado em repositórios GitHub. Classifique em UMA das quatro categorias abaixo.

TAXONOMIA:
- application: produto que o usuário final executa diretamente (editor, jogo, CLI que É o produto, web app, ferramenta standalone)
- helper: biblioteca/utilitário importado como dependência por outros desenvolvedores (ex: pandas, flask, requests, axios, sqlalchemy)
- extender: plugin/extensão que depende de um host específico para funcionar (ex: plugin VS Code, tema WordPress, bot Discord, extension Gatsby)
- other: não é software executável — listas curadas, tutoriais, datasets, coleções de exemplos, templates educacionais

SINAIS DECISIVOS:

application:
  • README tem "Download", "brew install X", "apt install X" ou link para binário
  • Tem interface direta com o usuário (GUI, TUI, CLI que o usuário chama)
  • Verbos de uso: "manage", "track", "monitor", "deploy", "run"

helper:
  • README tem "pip install X", "npm install X", "cargo add X" + exemplos de import/require/use
  • Tem setup.py / pyproject.toml / package.json como biblioteca publicada
  • README mostra snippets de código com import dentro do projeto do usuário

extender:
  • Nome/descrição contém: plugin, theme, extension, addon, integration
  • README diz "requires X" ou "for X" onde X é um sistema hospedeiro específico

other:
  • Nome começa com "awesome-" ou "learning-" ou contém "tutorial", "course", "examples", "dataset"
  • README é uma lista de links ou recursos, sem código executável próprio
  • É um repositório de soluções de exercícios, notas de aula ou coleção de snippets

DESEMPATE (em ordem):
1. Tem host obrigatório específico? → extender
2. Instalado via package manager para ser importado no código? → helper
3. Executado diretamente como produto pelo usuário final? → application
4. Não é software, é recurso de referência/aprendizado? → other
5. Ainda ambíguo após as 4 regras? → incerto

EXEMPLOS:
  application: gitui ("terminal-ui for git"), httpie ("HTTP client"), vscode ("Code editor")
  helper:      requests ("HTTP library"), axios ("HTTP client for node"), pytest ("testing framework")
  extender:    vim-fugitive ("Git plugin for Vim"), gatsby-plugin-image, prettier-plugin-tailwindcss
  other:       awesome-python, javascript-algorithms, deep-learning-coursera, chatgpt_system_prompt
"""

_SINGLE_SUFFIX = """\
REPOSITÓRIO:
Nome: {name}
Descrição: {description}
Linguagem principal: {language}
Estrelas: {stars}
Forks: {forks}
README (primeiros 1000 caracteres): {readme_snippet}

Responda SOMENTE com JSON válido (sem texto adicional):
{{"label": "application"|"helper"|"extender"|"other"|"incerto", "confidence": 0.0-1.0, "reasoning": "uma frase com o sinal decisivo usado"}}"""


def build_prompt(row: dict) -> str:
    readme = str(row.get("readme_text", "") or "")[:README_CHARS_SINGLE]
    return TAXONOMY_HEADER + _SINGLE_SUFFIX.format(
        name=row.get("name", ""),
        description=row.get("description", "") or "(sem descrição)",
        language=row.get("language", "") or "desconhecida",
        stars=row.get("stars", 0),
        forks=row.get("forks", 0),
        readme_snippet=readme or "(sem README)",
    )


def build_batch_prompt(rows: list[dict]) -> str:
    n = len(rows)
    intro = (
        f"Classifique os {n} repositórios abaixo. "
        "Responda SOMENTE com um JSON array válido — um objeto por repositório, "
        "NA MESMA ORDEM em que aparecem. Não adicione texto fora do array.\n\n"
        "Formato obrigatório:\n"
        "[\n"
        '  {"repo_id": "org/repo", "label": "application|helper|extender|other|incerto", '
        '"confidence": 0.0-1.0, "reasoning": "uma frase"},\n'
        "  ...\n"
        "]\n"
    )
    sections = []
    for i, row in enumerate(rows, 1):
        readme = str(row.get("readme_text", "") or "")[:README_CHARS_BATCH]
        sections.append(
            f"--- REPOSITÓRIO {i}: {row.get('repo_id', '')} ---\n"
            f"Nome: {row.get('name', '')}\n"
            f"Linguagem: {row.get('language', '') or 'desconhecida'} | "
            f"Estrelas: {row.get('stars', 0)} | Forks: {row.get('forks', 0)}\n"
            f"README: {readme or '(sem README)'}"
        )
    return TAXONOMY_HEADER + intro + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_label_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not match:
        return {"label": "incerto", "confidence": 0.0, "reasoning": "parse_error"}
    try:
        result = json.loads(match.group())
        if result.get("label") not in VALID_LABELS:
            result["label"] = "incerto"
        result["confidence"] = float(result.get("confidence", 0.5))
        return result
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"label": "incerto", "confidence": 0.0, "reasoning": "parse_error"}


def _parse_batch_response(text: str, rows: list[dict]) -> list[dict] | None:
    """Parse JSON array response. Returns None if parsing fails."""
    text = text.strip()
    # Extract JSON array (tolerates markdown fences)
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return None
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(items, list) or len(items) != len(rows):
        return None

    results = []
    for item, row in zip(items, rows):
        if not isinstance(item, dict):
            return None
        label = item.get("label", "incerto")
        if label not in VALID_LABELS:
            label = "incerto"
        results.append({
            "repo_id": row["repo_id"],
            "name": row.get("name", ""),
            "label": label,
            "confidence": float(item.get("confidence", 0.5)),
            "reasoning": str(item.get("reasoning", "")),
            "label_source": "haiku_auto_batch",
        })
    return results


# ---------------------------------------------------------------------------
# Claude CLI calls
# ---------------------------------------------------------------------------

def _run_claude(prompt: str, timeout: int = 90) -> str:
    """Run claude CLI.

    Prompts > 8000 chars are written to a temp file and piped via stdin to avoid
    the Windows CreateProcess 32767-char command-line limit.
    """
    import tempfile

    if len(prompt) > 8_000:
        # Write to temp file and pipe via stdin
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(prompt)
            tmp = Path(f.name)
        try:
            with open(tmp, encoding="utf-8") as stdin_f:
                result = subprocess.run(
                    ["claude", "-p", "--model", HAIKU_MODEL],
                    stdin=stdin_f,
                    capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                )
        finally:
            tmp.unlink(missing_ok=True)
    else:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", HAIKU_MODEL],
            capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(stderr if stderr else f"exit code {result.returncode}")
    return result.stdout


def call_haiku(client, prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            return _parse_label_response(_run_claude(prompt))
        except subprocess.TimeoutExpired:
            if attempt == retries - 1:
                return {"label": "incerto", "confidence": 0.0, "reasoning": "timeout"}
        except Exception as e:
            if attempt == retries - 1:
                return {"label": "incerto", "confidence": 0.0, "reasoning": f"cli_error: {e}"}
        time.sleep(3 ** attempt)
    return {"label": "incerto", "confidence": 0.0, "reasoning": "max_retries"}


def call_haiku_batch(client, rows: list[dict], retries: int = 3, debug: bool = False) -> list[dict]:
    """Label a batch of repos in one CLI call. Falls back to 1-by-1 if parsing fails."""
    prompt = build_batch_prompt(rows)
    last_response = ""
    last_error = ""

    for attempt in range(retries):
        try:
            text = _run_claude(prompt, timeout=120)
            last_response = text
            parsed = _parse_batch_response(text, rows)
            if parsed is not None:
                return parsed
            last_error = f"parse_error (response len={len(text)})"
        except subprocess.TimeoutExpired:
            last_error = "timeout"
        except Exception as e:
            last_error = str(e)
        time.sleep(3 ** attempt)

    print(f"\n  [WARN] batch falhou após {retries} tentativas: {last_error}")
    if debug and last_response:
        print(f"\n  --- RESPOSTA CRUA (primeiros 1000 chars) ---")
        print(f"  {last_response[:1000]}")
        print(f"  --- FIM ---\n")

    # Fallback: individual calls
    print("  Revertendo para chamadas individuais para este batch...")
    results = []
    for row in rows:
        r = call_haiku(client, build_prompt(row if isinstance(row, dict) else row.to_dict()))
        results.append({
            "repo_id": row["repo_id"],
            "name": row.get("name", ""),
            "label": r["label"],
            "confidence": r["confidence"],
            "reasoning": r.get("reasoning", ""),
            "label_source": "haiku_auto_fallback",
        })
        time.sleep(2)
    return results


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def run_test(df: pd.DataFrame, batch_size: int, debug: bool = False) -> None:
    """Label one batch and display quality report. Does not save output."""
    sample = df.sample(min(batch_size, len(df)), random_state=GLOBAL_SEED)
    rows = sample.to_dict("records")

    print(f"\n=== MODO TEST: {len(rows)} repos, batch_size={batch_size} ===")
    prompt = build_batch_prompt(rows)
    print(f"Prompt: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
    print("Enviando uma chamada ao Claude Haiku...\n")

    results = call_haiku_batch(None, rows, debug=debug)

    print(f"\n{'='*60}")
    print(f"RESULTADO DO TEST — {len(results)} repos classificados")
    print(f"{'='*60}")

    for r in results:
        conf_str = f"{r['confidence']:.2f}"
        print(f"  [{r['label']:>11}] ({conf_str}) {r['name'][:40]:<40} — {r['reasoning'][:60]}")

    labels = [r["label"] for r in results]
    print(f"\nDistribuição:")
    for lbl, cnt in pd.Series(labels).value_counts().items():
        print(f"  {lbl}: {cnt} ({cnt/len(results)*100:.0f}%)")

    errors = [r for r in results if "error" in r.get("reasoning", "").lower()]
    print(f"\nErros de parse: {len(errors)}")
    print("\nNenhum arquivo foi salvo. Se os labels parecem corretos, rode sem --test.")


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def _check_claude_cli() -> None:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return
        print("ERROR: claude CLI not found and ANTHROPIC_API_KEY not set.")
        sys.exit(1)


def run_labeling(
    input_path: Path,
    sample: int | None,
    all_repos: bool,
    output_gold: Path,
    output_labels: Path,
    resume: bool,
    retry_errors: bool,
    batch_size: int = 1,
    test_mode: bool = False,
    debug: bool = False,
) -> None:
    _check_claude_cli()

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} repos from {input_path}")

    if all_repos:
        target_df = df.copy()
        print(f"Labeling all {len(target_df):,} repos (batch_size={batch_size})")
    else:
        n = min(sample or 500, len(df))
        target_df = df.sample(n=n, random_state=GLOBAL_SEED).copy()
        print(f"Sampled {len(target_df):,} repos for gold standard")

    if test_mode:
        run_test(target_df, batch_size, debug=debug)
        return

    # Resume / retry-errors
    existing_ids: set[str] = set()
    existing_rows: list[dict] = []
    if (resume or retry_errors) and output_gold.exists():
        existing = pd.read_csv(output_gold)
        if retry_errors:
            error_pattern = "cli_error|timeout|max_retries|parse_error|batch_parse"
            error_mask = existing["reasoning"].str.contains(error_pattern, na=False)
            n_errors = error_mask.sum()
            existing = existing[~error_mask]
            print(f"Retrying {n_errors} failed rows; keeping {len(existing)} good labels")
        existing_ids = set(existing["repo_id"])
        existing_rows = existing.to_dict("records")
        if resume and not retry_errors:
            print(f"Resuming: {len(existing_ids)} repos already labeled")

    target_df = target_df[~target_df["repo_id"].isin(existing_ids)]
    total = len(target_df)
    print(f"Repos to label: {total:,}")

    results = list(existing_rows)
    uncertain_count = 0
    labeled_count = 0

    if batch_size <= 1:
        # Modo original: 1 repo por chamada
        for i, (_, row) in enumerate(target_df.iterrows()):
            label_result = call_haiku(None, build_prompt(row.to_dict()))
            results.append({
                "repo_id": row["repo_id"],
                "name": row["name"],
                "label": label_result["label"],
                "confidence": label_result["confidence"],
                "reasoning": label_result.get("reasoning", ""),
                "label_source": "haiku_auto",
            })
            if label_result["label"] == "incerto":
                uncertain_count += 1
            labeled_count += 1
            if labeled_count % 50 == 0:
                _save_gold_csv(results, output_gold)
                pct = labeled_count / total * 100
                print(f"  Progress: {labeled_count:,}/{total:,} ({pct:.1f}%) — uncertain: {uncertain_count}", end="\r")
            time.sleep(2)
    else:
        # Modo batch
        rows_list = target_df.to_dict("records")
        batches = [rows_list[i:i + batch_size] for i in range(0, len(rows_list), batch_size)]
        total_batches = len(batches)

        for batch_i, batch in enumerate(batches, 1):
            batch_results = call_haiku_batch(None, batch)
            results.extend(batch_results)
            uncertain_count += sum(1 for r in batch_results if r["label"] == "incerto")
            labeled_count += len(batch_results)
            _save_gold_csv(results, output_gold)
            pct = labeled_count / total * 100
            print(
                f"  Batch {batch_i}/{total_batches} — "
                f"{labeled_count:,}/{total:,} repos ({pct:.1f}%) — "
                f"uncertain: {uncertain_count}",
                end="\r",
            )
            time.sleep(3)

    _save_gold_csv(results, output_gold)

    if all_repos:
        labels_df = pd.DataFrame(results)[["repo_id", "label", "confidence", "label_source"]]
        labels_df.to_parquet(output_labels, index=False)
        print(f"\nFull labels saved to {output_labels}")

    gold_df = pd.read_csv(output_gold)
    print(f"\n\nLabeling complete!")
    print(f"  Total labeled: {len(gold_df):,}")
    print(f"  Label distribution:\n{gold_df['label'].value_counts().to_string()}")
    print(f"  Uncertain: {(gold_df['label'] == 'incerto').sum():,}")
    print(f"  Low confidence (< 0.7): {(gold_df['confidence'] < 0.7).sum():,}")


def _save_gold_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label repositories using Claude Haiku")
    parser.add_argument("--input", type=Path, default=Path("data/raw/repos.parquet"))
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--all", dest="all_repos", action="store_true")
    parser.add_argument("--output-gold", type=Path, default=Path("data/raw/gold_standard.csv"))
    parser.add_argument("--output-labels", type=Path, default=Path("data/raw/labels.parquet"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Repos por chamada ao Claude (1=original, 50=recomendado para --all)",
    )
    parser.add_argument(
        "--test", dest="test_mode", action="store_true",
        help="Roda um único batch e exibe resultado — não salva nada",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mostra resposta crua do modelo quando o parse do batch falha",
    )
    args = parser.parse_args()

    run_labeling(
        input_path=args.input,
        sample=args.sample,
        all_repos=args.all_repos,
        output_gold=args.output_gold,
        output_labels=args.output_labels,
        resume=args.resume,
        retry_errors=args.retry_errors,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
