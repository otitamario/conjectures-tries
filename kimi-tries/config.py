"""
Configuração do escopo de busca. Como o arXiv não filtra por MSC
nativamente (só por categoria arXiv), usamos:
  1. categorias arXiv como filtro primário no lado do servidor
     (math.CO, math.NT cobrem a maior parte do escopo);
  2. os MSC codes como filtro client-side *opcional* sobre o campo
     de comentário/abstract, quando o autor declarou o MSC -- isso
     não é garantido (nem todo paper no arXiv declara MSC), então
     trate como refinamento, não como filtro exaustivo.
"""

ARXIV_CATEGORIES = ["math.CO", "math.NT"]

TARGET_MSC_PREFIXES = [
    "05A",  # Enumerative combinatorics
    "05C",  # Graph theory
    "05E",  # Algebraic combinatorics
    "11A",  # Elementary number theory
    "11B",  # Sequences and sets (Fibonacci/Lucas etc.)
    "11D",  # Diophantine equations
    "11N",  # Multiplicative number theory (primes)
    "11P",  # Additive number theory (partitions)
    "11T",  # Finite fields
]

DATE_TO = "2024-12-31"
DATE_FROM = "2000-01-01"  # None = sem limite inferior

MAX_RESULTS_PER_SEARCH = 100
ARXIV_RATE_LIMIT_SECONDS = 3  # arXiv pede pelo menos 3s entre requisições