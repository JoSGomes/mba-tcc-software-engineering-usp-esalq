"""Executa o pipeline de geração de embeddings e comparação de modelos de ponta a ponta.

Encadeia, em uma única chamada:

  1. preparacao/build_features.py --feature-sets sbert codebert
     (gera data/processed/X_{sbert,codebert}_*.npy)
  2. modelagem/compare_models.py --phase a
     (Grid Search nas 9 combinações representação x classificador)
  3. modelagem/compare_models.py --phase b
     (validação cruzada aninhada 10x5 entre as 3 representações + matriz de
      confusão agregada — etapa mais demorada)

Cada etapa roda como subprocesso (saída em tempo real no console) e o tempo
decorrido de cada uma é reportado ao final.

Pré-requisito para GPU (opcional, acelera a geração de embeddings):
`python -c "import torch; print(torch.cuda.is_available())"` deve imprimir True.

Uso:
    python modelagem/run_pipeline.py                    # roda as 3 etapas
    python modelagem/run_pipeline.py --skip-embeddings   # pula a etapa 1
                                                          # (embeddings já gerados)
    python modelagem/run_pipeline.py --only a            # roda só a etapa de seleção
    python modelagem/run_pipeline.py --only b            # roda só a comparação final
                                                          # (requer a etapa a já executada)
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
    parser.add_argument("--only", choices=["a", "b"], default=None,
                         help="roda apenas a etapa de seleção (a) ou a de comparação final (b) "
                              "de compare_models.py (implica --skip-embeddings)")
    args = parser.parse_args()

    py = sys.executable
    timings: dict[str, float] = {}

    if args.only is None and not args.skip_embeddings:
        timings["embeddings (sbert+codebert)"] = _run_step(
            "1/3 — geração de embeddings SBERT + CodeBERT",
            [py, "preparacao/build_features.py", "--feature-sets", "sbert", "codebert"],
        )

    if args.only in (None, "a"):
        timings["seleção (grid search)"] = _run_step(
            "2/3 — seleção de classificador por representação (Grid Search, 9 combinações)",
            [py, "modelagem/compare_models.py", "--phase", "a"],
        )

    if args.only in (None, "b"):
        timings["comparação final (nested CV)"] = _run_step(
            "3/3 — comparação estatística entre as 3 representações (nested 10x5 CV)",
            [py, "modelagem/compare_models.py", "--phase", "b"],
        )

    print("\n" + "=" * 70)
    print("Resumo de tempos")
    print("=" * 70)
    for step, elapsed in timings.items():
        print(f"  {step:35s} {elapsed / 60:6.1f} min")


if __name__ == "__main__":
    main()
