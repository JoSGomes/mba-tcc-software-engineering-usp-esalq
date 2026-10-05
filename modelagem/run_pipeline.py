"""Executa a comparação entre representações e as análises seguintes de ponta a ponta.

Encadeia, em uma única chamada:

  1. preparacao/build_features.py --feature-sets sbert codebert
     (gera data/processed/X_{sbert,codebert}_*.npy)
  2. modelagem/nested_cv_full_selection.py
     (validação cruzada aninhada 10x5 com o ajuste do Doc2Vec e a escolha do
      classificador dentro de cada rodada externa — etapa mais demorada,
      ~15 h em CPU; retoma do último fold concluído se interrompida)
  3. modelagem/comparacoes_estatisticas.py
     (teste t com reamostragem corrigido entre as configurações)
  4. avaliacao/ood_analysis.py
     (modelo final aplicado às classes excluídas do treino)
  5. extracao/compute_revisao_manual.py
     (métricas da revisão manual, calibração e falsos rejeitados)
  6. figuras: modelagem/figura_matrizes_confusao.py e
     avaliacao/figura_confianca_calibracao.py

Cada etapa roda como subprocesso (saída em tempo real no console) e o tempo
decorrido de cada uma é reportado ao final.

Pré-requisito para GPU (opcional, acelera a geração de embeddings):
`python -c "import torch; print(torch.cuda.is_available())"` deve imprimir True.

Uso:
    python modelagem/run_pipeline.py                    # roda todas as etapas
    python modelagem/run_pipeline.py --skip-embeddings  # pula a etapa 1 (embeddings já gerados)
    python modelagem/run_pipeline.py --skip-nested-cv   # pula as etapas 1 e 2 (usa os
                                                        # resultados do nested CV já salvos)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _run_step(name: str, cmd: list[str]) -> float:
    print("\n" + "=" * 70)
    print(name)
    print(f"$ {' '.join(cmd)}")
    print("=" * 70)
    start = time.monotonic()
    subprocess.run(cmd, check=True, cwd=ROOT)
    elapsed = time.monotonic() - start
    print(f"\n[{name}] concluído em {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="pula a geração de X_sbert_*.npy/X_codebert_*.npy (já gerados)")
    parser.add_argument("--skip-nested-cv", action="store_true",
                        help="pula a geração de embeddings e o nested CV (usa os resultados já salvos)")
    args = parser.parse_args()

    py = sys.executable
    steps = []
    if not (args.skip_embeddings or args.skip_nested_cv):
        steps.append(("geração de embeddings SBERT + CodeBERT",
                      [py, "preparacao/build_features.py", "--feature-sets", "sbert", "codebert"]))
    if not args.skip_nested_cv:
        steps.append(("validação cruzada aninhada com seleção completa",
                      [py, "modelagem/nested_cv_full_selection.py"]))
    steps += [
        ("comparações estatísticas (teste t corrigido)", [py, "modelagem/comparacoes_estatisticas.py"]),
        ("avaliação nas classes excluídas do treino", [py, "avaliacao/ood_analysis.py"]),
        ("métricas da revisão manual e calibração", [py, "extracao/compute_revisao_manual.py"]),
        ("figura das matrizes de confusão", [py, "modelagem/figura_matrizes_confusao.py"]),
        ("figura de confiança e calibração", [py, "avaliacao/figura_confianca_calibracao.py"]),
    ]

    timings: dict[str, float] = {}
    for i, (name, cmd) in enumerate(steps, 1):
        timings[name] = _run_step(f"{i}/{len(steps)} — {name}", cmd)

    print("\n" + "=" * 70)
    print("Resumo de tempos")
    print("=" * 70)
    for step, elapsed in timings.items():
        print(f"  {step:50s} {elapsed / 60:6.1f} min")


if __name__ == "__main__":
    main()
