"""
Busca no arXiv por categoria + intervalo de datas (filtro no servidor),
com refinamento opcional por MSC declarado no comentário/abstract
(filtro client-side, best-effort -- nem todo paper declara MSC).
"""

import re
import time
from dataclasses import dataclass

import feedparser
import requests

from config import (
    ARXIV_CATEGORIES, TARGET_MSC_PREFIXES, DATE_TO, DATE_FROM,
    MAX_RESULTS_PER_SEARCH, ARXIV_RATE_LIMIT_SECONDS,
)

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# MSC codes têm o formato "05C05", "11B39" etc. -- procura isso em
# qualquer lugar do comentário ou abstract.
MSC_PATTERN = re.compile(r"\b(\d{2}[A-Z]\d{2})\b")


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    published: str  # ISO 8601
    summary: str
    comment: str
    msc_codes_found: list[str]


def _format_arxiv_date(date_str: str, end_of_day: bool) -> str:
    """'2024-12-31' -> '202412312359' (formato exigido pelo submittedDate do arXiv)."""
    compact = date_str.replace("-", "")
    return compact + ("2359" if end_of_day else "0000")


def _build_query(categories: list[str], date_from: str | None, date_to: str | None) -> str:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cat_query})"
    if date_from or date_to:
        start = _format_arxiv_date(date_from, end_of_day=False) if date_from else "000101010000"
        end = _format_arxiv_date(date_to, end_of_day=True) if date_to else "999912312359"
        query += f" AND submittedDate:[{start} TO {end}]"
    return query


def _extract_msc(text: str) -> list[str]:
    return sorted(set(MSC_PATTERN.findall(text or "")))


def search_arxiv(
    categories: list[str] = ARXIV_CATEGORIES,
    date_from: str | None = DATE_FROM,
    date_to: str | None = DATE_TO,
    max_results: int = MAX_RESULTS_PER_SEARCH,
    msc_prefixes: list[str] | None = TARGET_MSC_PREFIXES,
) -> list[ArxivPaper]:
    """
    Busca papers no arXiv. Filtro de categoria+data é feito no servidor
    (confiável). Se `msc_prefixes` for passado, o resultado final só
    inclui papers cujo comentário/abstract declara um MSC que comece
    com um dos prefixos -- isso descarta papers que simplesmente não
    declaram MSC, então pode ser mais restritivo do que o desejado.
    Passe msc_prefixes=None para pular esse refinamento e manter tudo
    que bateu na categoria/data.
    """
    query = _build_query(categories, date_from, date_to)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()
    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        comment = getattr(entry, "arxiv_comment", "")
        summary = entry.summary
        msc_found = _extract_msc(f"{comment} {summary}")

        papers.append(ArxivPaper(
            arxiv_id=arxiv_id,
            title=entry.title,
            published=entry.published,
            summary=summary,
            comment=comment,
            msc_codes_found=msc_found,
        ))

    if msc_prefixes is None:
        return papers

    def matches(paper: ArxivPaper) -> bool:
        return any(code[:3] in msc_prefixes for code in paper.msc_codes_found)

    return [p for p in papers if matches(p)]


def search_arxiv_paged(
    categories: list[str] = ARXIV_CATEGORIES,
    date_from: str | None = DATE_FROM,
    date_to: str | None = DATE_TO,
    total_max: int = 500,
    page_size: int = 100,
    msc_prefixes: list[str] | None = TARGET_MSC_PREFIXES,
) -> list[ArxivPaper]:
    """Igual a search_arxiv, mas pagina além do limite de uma requisição,
    respeitando o rate limit pedido pelo arXiv (>= 3s entre chamadas)."""
    query = _build_query(categories, date_from, date_to)
    all_papers: list[ArxivPaper] = []
    start = 0

    while len(all_papers) < total_max:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(page_size, total_max - len(all_papers)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.text)

        if not feed.entries:
            break

        for entry in feed.entries:
            arxiv_id = entry.id.split("/abs/")[-1]
            comment = getattr(entry, "arxiv_comment", "")
            summary = entry.summary
            msc_found = _extract_msc(f"{comment} {summary}")
            all_papers.append(ArxivPaper(
                arxiv_id=arxiv_id, title=entry.title, published=entry.published,
                summary=summary, comment=comment, msc_codes_found=msc_found,
            ))

        start += page_size
        time.sleep(ARXIV_RATE_LIMIT_SECONDS)

    if msc_prefixes is None:
        return all_papers

    def matches(paper: ArxivPaper) -> bool:
        return any(code[:3] in msc_prefixes for code in paper.msc_codes_found)

    return [p for p in all_papers if matches(p)]
