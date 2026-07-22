"""
Cliente zbMATH Open. Usado como camada de DESCOBERTA precisa por MSC
(o zbMATH tem MSC nativo; o arXiv não). Desde a extensão de 2024, o
zbMATH também indexa diretamente ~200k preprints do arXiv como
documentos próprios, então um resultado de busca por MSC pode já ser
um preprint do arXiv.

NOTA DE HONESTIDADE: não encontrei documentação acessível (bot
detection no zbmath.org no momento) confirmando o nome exato do
campo JSON onde o zbMATH guarda o identificador do arXiv dentro da
resposta do endpoint /document/_search. Por isso a extração abaixo
é defensiva: varre o registro inteiro (todos os campos) procurando
o padrão de id do arXiv (ex. "2401.01234"), em vez de depender de
um nome de campo específico que eu não confirmei. Isso é mais
robusto a mudanças de schema, mas também pode achar falsos positivos
raros (um id de arXiv citado dentro do abstract, por exemplo) --
vale checar manualmente os primeiros resultados antes de rodar em
lote grande.
"""

import json
import re
import time
from dataclasses import dataclass

import requests

ZBMATH_SEARCH_URL = "https://api.zbmath.org/document/_search"
ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
RATE_LIMIT_SECONDS = 1.0


@dataclass
class ZbmathResult:
    zbmath_title: str
    msc: list[str]
    year: str | None
    arxiv_id: str | None  # None se não achou nenhum id de arXiv no registro
    raw: dict  # registro completo, pra debug/auditoria


def _extract_arxiv_id(doc: dict) -> str | None:
    """Varre o registro inteiro (serializado como JSON) procurando um
    id no formato do arXiv. Best-effort -- ver nota no topo do arquivo."""
    blob = json.dumps(doc)
    match = ARXIV_ID_PATTERN.search(blob)
    return match.group(1) if match else None


def search_documents(
    msc_prefixes: list[str],
    year_from: str,
    year_to: str,
    max_results: int = 100,
    page_size: int = 100,
) -> list[ZbmathResult]:
    """
    Busca documentos no zbMATH por MSC (prefixo, ex. '05C' cobre
    '05C05', '05C69' etc. via wildcard) e intervalo de anos.
    Retorna TODOS os resultados encontrados (com ou sem arXiv id) --
    use filter_with_arxiv() pra manter só os que têm um id extraído.
    """
    msc_query = " OR ".join(f"msc:{prefix}*" for prefix in msc_prefixes)
    query = f"({msc_query}) AND py:[{year_from} TO {year_to}]"

    results: list[ZbmathResult] = []
    page = 0

    while len(results) < max_results:
        params = {
            "search_string": query,
            "page": page,
            "results_per_page": min(page_size, max_results - len(results)),
        }
        response = requests.get(ZBMATH_SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        docs = data.get("result", [])
        if not docs:
            break

        for doc in docs:
            results.append(ZbmathResult(
                zbmath_title=doc.get("title", {}).get("title", "") if isinstance(doc.get("title"), dict) else str(doc.get("title", "")),
                msc=doc.get("msc", []),
                year=str(doc.get("year")) if doc.get("year") else None,
                arxiv_id=_extract_arxiv_id(doc),
                raw=doc,
            ))

        page += 1
        time.sleep(RATE_LIMIT_SECONDS)

    return results


def filter_with_arxiv(results: list[ZbmathResult]) -> list[ZbmathResult]:
    """Mantém só os registros onde um id de arXiv foi encontrado --
    são esses que dá pra alimentar em paper_fetch.get_arxiv_paper_text."""
    return [r for r in results if r.arxiv_id is not None]
