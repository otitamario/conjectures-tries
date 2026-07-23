"""
Ponto de entrada. Dado um paper do arXiv, extrai automaticamente os
teoremas/lemas provados dentro do escopo (Prompt 0) e roda o
pipeline completo (Prompt 1/2/3/6) para cada um.

Uso: python main.py 2401.01234
"""

import sys

from db import init_db, save_run, mark_explored
from llm_calls import run_prompt_0
from paper_fetch import get_arxiv_paper_text
from orchestrator import PipelineRun, run_analysis_and_conjectures, run_top_candidates, summarize

DB_PATH = "conjectures.db"


def process_paper(arxiv_id: str) -> bool:
    """Retorna True se pelo menos um resultado do paper foi processado
    e salvo no banco; False se não achou nada dentro do escopo (não é
    erro, só não rendeu nada aproveitável)."""
    init_db(DB_PATH)

    print(f"Baixando e extraindo texto de arXiv:{arxiv_id}...")
    paper_text = get_arxiv_paper_text(arxiv_id)
    print(f"Texto extraído ({len(paper_text)} caracteres). Rodando Prompt 0...")

    extracted = run_prompt_0(paper_text)
    in_scope_results = [r for r in extracted.results if r.in_scope and r.is_proved_in_paper]

    print(f"Encontrados {len(extracted.results)} resultados no total, "
          f"{len(in_scope_results)} provados e dentro do escopo.")

    if not in_scope_results:
        print("Nenhum resultado provado dentro do escopo encontrado. Parando.")
        mark_explored(DB_PATH, arxiv_id, source="arxiv",
                      in_scope_results_found=0, saved_anything=False)
        return False

    paper_link = f"https://arxiv.org/abs/{arxiv_id}"
    saved_anything = False

    for result in in_scope_results:
        print(f"\n{'=' * 70}\nProcessando {result.label}\n{'=' * 70}")

        run = PipelineRun(
            theorem_statement=result.statement,
            context=result.context,
            source="arxiv",
            external_id=arxiv_id,
            paper_title=result.label,  # troque por título real do paper se tiver via arXiv API
            paper_link=paper_link,
        )

        run_analysis_and_conjectures(run)
        if run.analysis and not run.analysis.in_scope:
            print(f"{result.label}: Prompt 1 discordou do escopo (Prompt 0 pode ter errado). Pulando.")
            continue

        print(f"Top 3 conjecturas: {run.conjectures.top_three}")
        results = run_top_candidates(run, attempt_proof=True)
        print(summarize(results))

        paper_id = save_run(run, results, db_path=DB_PATH)
        print(f"Salvo (paper_id={paper_id}).")
        saved_anything = True

    print(f"\nConcluído arXiv:{arxiv_id}.")
    mark_explored(DB_PATH, arxiv_id, source="arxiv",
                  in_scope_results_found=len(in_scope_results), saved_anything=saved_anything)
    return saved_anything


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py <arxiv_id>   (ex.: python main.py 2401.01234)")
        sys.exit(1)
    process_paper(sys.argv[1])
