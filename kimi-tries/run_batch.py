"""
Busca papers no arXiv por categoria/MSC + data limite, e roda o
pipeline completo (Prompt 0->1->2->3->6) para cada um encontrado.

Uso: python run_batch.py [--max-papers N] [--no-msc-filter]
Edite config.py para mudar categorias, MSC alvo e data limite.
"""

import argparse
import time

from config import ARXIV_CATEGORIES, TARGET_MSC_PREFIXES, DATE_FROM, DATE_TO, ARXIV_RATE_LIMIT_SECONDS
from arxiv_search import search_arxiv_paged
from main import process_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--no-msc-filter", action="store_true",
                         help="Não filtra por MSC declarado -- mantém tudo que bateu na categoria/data.")
    args = parser.parse_args()

    msc_prefixes = None if args.no_msc_filter else TARGET_MSC_PREFIXES

    print(f"Buscando no arXiv: categorias={ARXIV_CATEGORIES}, "
          f"data até {DATE_TO}, MSC alvo={msc_prefixes}")
    papers = search_arxiv_paged(
        categories=ARXIV_CATEGORIES,
        date_from=DATE_FROM,
        date_to=DATE_TO,
        total_max=args.max_papers * 5,  # busca uma margem maior, já que o filtro de MSC descarta parte
        msc_prefixes=msc_prefixes,
    )

    papers = papers[: args.max_papers]
    print(f"{len(papers)} papers selecionados para processar.\n")

    for i, paper in enumerate(papers, 1):
        print(f"\n{'#' * 70}")
        print(f"# PAPER {i}/{len(papers)}: {paper.title.strip()}")
        print(f"# arXiv:{paper.arxiv_id} | MSC declarado: {paper.msc_codes_found or 'nenhum'}")
        print(f"{'#' * 70}")
        try:
            process_paper(paper.arxiv_id)
        except Exception as e:
            print(f"[erro ao processar {paper.arxiv_id}]: {e}")
        time.sleep(ARXIV_RATE_LIMIT_SECONDS)

    print("\nLote concluído.")


if __name__ == "__main__":
    main()
