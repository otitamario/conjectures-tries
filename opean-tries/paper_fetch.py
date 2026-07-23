"""
Busca e extração de texto completo de papers.

IMPORTANTE: zbMATH Open é um serviço de indexação/resumo -- na
prática ele não hospeda o texto completo do paper (que costuma
estar atrás do paywall do editor). Pra extração automática do
enunciado dos teoremas você precisa do texto completo, então esse
módulo foca em arXiv (preprints, PDF aberto). Se seu paper só
existe no zbMATH/editora paga, baixe o PDF manualmente e use
`extract_text_from_pdf` direto.
"""

import re
import tempfile
import requests
from pypdf import PdfReader

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


def fetch_arxiv_pdf(arxiv_id: str) -> str:
    """Baixa o PDF do arXiv e retorna o caminho do arquivo temporário."""
    arxiv_id = re.sub(r"^arXiv:", "", arxiv_id.strip())
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(response.content)
        return f.name


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_arxiv_paper_text(arxiv_id: str) -> str:
    """Atalho: baixa e extrai o texto completo de um paper do arXiv pelo id (ex. '2401.01234')."""
    pdf_path = fetch_arxiv_pdf(arxiv_id)
    return extract_text_from_pdf(pdf_path)
