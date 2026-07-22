"""
Descoberta via zbMATH (MSC nativo, preciso) + processamento via
texto completo do arXiv. Processa até N papers BEM SUCEDIDOS (achou
pelo menos um resultado no escopo e salvou algo) -- não N tentativas.

Pula automaticamente qualquer arxiv_id já explorado em chamadas
anteriores (tabela explored_papers no banco), então rodar de novo
não reprocessa o que já foi visto, seja o resultado bem sucedido
ou não.

Uso: python run_batch_zbmath.py [--target-successes N]
Edite config.py para mudar TARGET_MSC_PREFIXES e DATE_TO/DATE_FROM.
"""

import argparse

from config import TARGET_MSC_PREFIXES, DATE_FROM, DATE_TO
from db import get_explored_ids
from zbmath_client import search_documents, filter_with_arxiv
from main import process_paper, DB_PATH

# quantas vezes buscar mais no zbMATH do que o alvo, já que uma parte
# vira dedup (já explorado) e outra não tem arXiv ou está fora do escopo
POOL_MULTIPLIER_START = 5
POOL_MULTIPLIER_MAX = 40


def _fetch_candidate_pool(pool_size: int) -> list:
    year_from = DATE_FROM[:4] if DATE_FROM else "1900"
    year_to = DATE_TO[:4]
    all_results = search_documents(
        msc_prefixes=TARGET_MSC_PREFIXES,
        year_from=year_from,
        year_to=year_to,
        max_results=pool_size,
    )
    return filter_with_arxiv(all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-successes", type=int, default=10,
                         help="Quantos papers BEM SUCEDIDOS processar nesta chamada.")
    args = parser.parse_args()

    already_explored = get_explored_ids(DB_PATH, source="arxiv")
    print(f"{len(already_explored)} papers já explorados em chamadas anteriores (serão pulados).")

    successes = 0
    attempts = 0
    tried_this_run: set[str] = set()
    multiplier = POOL_MULTIPLIER_START

    while successes < args.target_successes and multiplier <= POOL_MULTIPLIER_MAX:
        pool = _fetch_candidate_pool(args.target_successes * multiplier)
        candidates = [
            r for r in pool
            if r.arxiv_id not in already_explored and r.arxiv_id not in tried_this_run
        ]

        if not candidates:
            print(f"Nenhum candidato novo com pool de {args.target_successes * multiplier} "
                  f"resultados do zbMATH. Ampliando a busca...")
            multiplier *= 2
            continue

        print(f"{len(candidates)} candidatos novos disponíveis nesta rodada de busca.")

        for result in candidates:
            if successes >= args.target_successes:
                break

            tried_this_run.add(result.arxiv_id)
            attempts += 1
            print(f"\n{'#' * 70}")
            print(f"# TENTATIVA {attempts} (sucessos até agora: {successes}/{args.target_successes})")
            print(f"# {result.zbmath_title}")
            print(f"# MSC: {result.msc} | ano: {result.year} | arXiv:{result.arxiv_id}")
            print(f"{'#' * 70}")

            try:
                ok = process_paper(result.arxiv_id)
                if ok:
                    successes += 1
            except Exception as e:
                print(f"[erro ao processar {result.arxiv_id}]: {e}")
                # mesmo com erro, não marcamos como explorado aqui --
                # pode ter sido falha transitória (rede, rate limit),
                # então uma próxima chamada pode tentar de novo.

        if successes < args.target_successes:
            multiplier *= 2

    print(f"\nConcluído: {successes}/{args.target_successes} papers bem sucedidos "
          f"em {attempts} tentativas nesta chamada.")
    if successes < args.target_successes:
        print("Não atingiu a meta -- pool de busca esgotado dentro do limite "
              f"(multiplier máximo {POOL_MULTIPLIER_MAX}x). Considere ampliar "
              "TARGET_MSC_PREFIXES ou o intervalo de datas em config.py.")


if __name__ == "__main__":
    main()
