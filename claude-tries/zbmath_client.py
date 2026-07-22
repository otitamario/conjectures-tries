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
o padrão de id do arXiv (ex. "2401.01234"). CONFIRMADO NA PRÁTICA:
esse padrão sozinho colide com o Zbl ID do próprio zbMATH (mesmo
formato "NNNN.NNNNN"), então a extração agora valida que os 4
primeiros dígitos formam um mês válido (01-12) e exige que a palavra
"arxiv" apareça perto do número no JSON bruto -- reduz bastante os
falsos positivos, mas ainda é heurístico; vale checar manualmente os
primeiros resultados antes de rodar em lote grande.

NOTAS DE SINTAXE DA QUERY (descobertas por tentativa e erro, já que a
documentação não estava acessível no momento):
  1. O endpoint é /v1/document/_search -- sem o "v1" a API responde
     303 See Other redirecionando pra cá, mas o redirect NÃO preserva
     a query string, gerando um 422 "field required" enganoso.
  2. O campo de MSC é "cc:", não "msc:".
  3. Intervalo de ano é "py:2020-2024" (hífen), não "py:[2020 TO 2024]".
  4. Cada termo individual "campo:valor" precisa de parênteses PRÓPRIOS
     quando combinado com AND/OR -- sem isso o wildcard (*) "vaza" pro
     operador seguinte e a busca retorna 0 resultados silenciosamente.
  5. Uma query com muitos termos OR (~9) também retorna 0 resultados,
     mesmo sintaticamente correta -- parece haver um limite não
     documentado de cláusulas por busca. Por isso este cliente faz UMA
     busca por prefixo de MSC e deduplica os resultados no cliente, em
     vez de combinar tudo numa OR gigante.
  6. Quando a busca não encontra nada, a API responde HTTP 404 com um
     corpo JSON informativo ({"status": {"internal_code": "successful
     access. No results found."}}) -- isso é uma resposta válida de
     "zero resultados", não um erro de rota. O código abaixo distingue
     esse caso (trata como lista vazia) de um erro de verdade.
"""

import json
import re
import time
from dataclasses import dataclass

import requests

ZBMATH_SEARCH_URL = "https://api.zbmath.org/v1/document/_search"
ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
RATE_LIMIT_SECONDS = 1.0
NO_RESULTS_INTERNAL_CODE = "successful access. No results found."


@dataclass
class ZbmathResult:
    zbmath_id: str | None
    zbmath_title: str
    msc: list[str]
    year: str | None
    arxiv_id: str | None  # None se não achou nenhum id de arXiv no registro
    raw: dict  # registro completo, pra debug/auditoria


def _extract_arxiv_id(doc: dict) -> str | None:
    """Varre o registro inteiro procurando um id no formato do arXiv
    (YYMM.NNNNN). Best-effort, com duas camadas de validação -- ver
    nota no topo do arquivo:

    1. Os 4 primeiros dígitos são interpretados como YYMM; se o "MM"
       não for um mês válido (01-12), não é um id de arXiv real --
       é quase certamente o Zbl ID do próprio zbMATH, que usa o MESMO
       formato visual "NNNN.NNNNN" mas sem essa restrição de mês.
    2. Exige que a palavra "arxiv" apareça no texto bruto perto do
       número encontrado, não só o padrão numérico isolado.
    """
    blob = json.dumps(doc)
    blob_lower = blob.lower()

    for match in ARXIV_ID_PATTERN.finditer(blob):
        candidate = match.group(1)
        month = int(candidate[2:4])
        if not (1 <= month <= 12):
            continue  # mês inválido -- provável Zbl ID, não arXiv

        window_start = max(0, match.start() - 60)
        window_end = min(len(blob), match.end() + 60)
        if "arxiv" in blob_lower[window_start:window_end]:
            return candidate

    return None


def _fetch_page(query: str, page: int, results_per_page: int) -> list[dict]:
    """Busca uma página. Retorna [] tanto para 'zero resultados' (404
    com internal_code conhecido) quanto para resultado vazio normal.
    Levanta exceção só para erros de verdade."""
    params = {
        "search_string": query,
        "page": page,
        "results_per_page": results_per_page,
    }
    response = requests.get(
        ZBMATH_SEARCH_URL,
        params=params,
        timeout=30,
        headers={"User-Agent": "conjecture-pipeline/0.1 (uso pessoal de pesquisa)"},
    )

    if response.status_code == 404:
        try:
            data = response.json()
        except ValueError:
            data = {}
        internal_code = data.get("status", {}).get("internal_code", "")
        if NO_RESULTS_INTERNAL_CODE in internal_code:
            return []  # zero resultados de verdade -- não é erro
        print(f"[zbMATH] ERRO 404 inesperado na URL: {response.url}")
        print(f"[zbMATH] Corpo da resposta: {response.text[:1000]}")
        response.raise_for_status()

    if not response.ok:
        print(f"[zbMATH] ERRO {response.status_code} na URL: {response.url}")
        print(f"[zbMATH] Corpo da resposta: {response.text[:1000]}")
    response.raise_for_status()

    return response.json().get("result", []) or []


def _search_single_prefix(
    prefix: str, year_from: str, year_to: str, max_results: int, page_size: int
) -> list[dict]:
    query = f"(cc:{prefix}*) AND (py:{year_from}-{year_to})"
    docs: list[dict] = []
    page = 0
    while len(docs) < max_results:
        batch = _fetch_page(query, page, min(page_size, max_results - len(docs)))
        if not batch:
            break
        docs.extend(batch)
        page += 1
        time.sleep(RATE_LIMIT_SECONDS)
    return docs


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

    Faz UMA busca por prefixo (não uma OR gigante -- ver nota 5 no
    topo do arquivo) e deduplica por zbmath_id, já que um documento
    pode ter mais de um MSC prefixo alvo entre suas classificações.

    max_results é o total combinado (após dedup), não por prefixo.
    Retorna TODOS os resultados encontrados (com ou sem arXiv id) --
    use filter_with_arxiv() pra manter só os que têm um id extraído.
    """
    seen_ids: set[str] = set()
    results: list[ZbmathResult] = []

    for prefix in msc_prefixes:
        if len(results) >= max_results:
            break

        remaining = max_results - len(results)
        docs = _search_single_prefix(prefix, year_from, year_to, remaining, page_size)

        for doc in docs:
            doc_id = str(doc.get("id")) if doc.get("id") is not None else None
            if doc_id is not None and doc_id in seen_ids:
                continue
            if doc_id is not None:
                seen_ids.add(doc_id)

            title = doc.get("title", {})
            results.append(ZbmathResult(
                zbmath_id=doc_id,
                zbmath_title=title.get("title", "") if isinstance(title, dict) else str(title),
                msc=doc.get("msc", []),
                year=str(doc.get("year")) if doc.get("year") else None,
                arxiv_id=_extract_arxiv_id(doc),
                raw=doc,
            ))

    return results


def filter_with_arxiv(results: list[ZbmathResult]) -> list[ZbmathResult]:
    """Mantém só os registros onde um id de arXiv foi encontrado --
    são esses que dá pra alimentar em paper_fetch.get_arxiv_paper_text."""
    return [r for r in results if r.arxiv_id is not None]