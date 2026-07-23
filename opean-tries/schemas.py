"""
Schemas estruturados para as saídas de cada etapa do pipeline.
Usar com response_format / tool-use estruturado (mesmo padrão do
orquestrador Lean 4: forçar JSON e validar com pydantic antes de
avançar de estado).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Checkability(str, Enum):
    SYMBOLIC = "SYMBOLIC"
    BOUNDED_COMPUTATIONAL = "BOUNDED_COMPUTATIONAL"
    NOT_READILY_CHECKABLE = "NOT_READILY_CHECKABLE"


class Status(str, Enum):
    PLAUSIBLY_TRUE = "PLAUSIBLY_TRUE"
    PLAUSIBLY_FALSE = "PLAUSIBLY_FALSE"
    UNCERTAIN = "UNCERTAIN"


# ---- Prompt 0: extração de resultados provados do paper -----------------

class ExtractedResult(BaseModel):
    label: str  # ex. "Theorem 3.2", "Lemma 4.1"
    statement: str
    context: str  # definições/notação necessárias para entender o statement
    in_scope: bool  # combinatória e/ou teoria dos números elementar/enumerativa
    is_proved_in_paper: bool  # False se for citado de outro paper, ou for uma conjectura do próprio autor


class ExtractedResultsBatch(BaseModel):
    results: list[ExtractedResult]


# ---- Prompt 1: análise -------------------------------------------------

class ResultAnalysis(BaseModel):
    in_scope: bool
    restatement: str
    subareas: list[str]
    quantified_objects: list[str]
    explicit_hypotheses: list[str]
    implicit_assumptions: list[str]
    conclusion: str
    likely_proof_mechanism: str
    checkability: Checkability
    needs_networkx: bool
    ambiguities: list[str]
    formalized_statement: str


# ---- Prompt 2: conjecturas ----------------------------------------------

class Conjecture(BaseModel):
    name: str
    changed_component: str
    statement: str
    motivation: str
    status: Status
    status_reason: str
    counterexample_direction: Optional[str] = None
    checkability: Checkability
    needs_networkx: bool


class ConjectureBatch(BaseModel):
    candidates: list[Conjecture] = Field(max_length=10)
    top_three: list[str]  # nomes, na ordem de prioridade


# ---- Prompt 3: plano de verificação --------------------------------------

class VerificationPlan(BaseModel):
    objects_and_representation: str
    sympy_networkx_functions: list[str]
    reduces_to_symbolic_identity: bool
    ambiguities_to_resolve: list[str]
    code: str  # código Python/Sympy pronto para rodar
    correspondence_table: dict[str, str]
    verification_risks: list[str]


# ---- Resultado de execução real (não-LLM) --------------------------------

class ExecutionResult(BaseModel):
    passed: bool
    stdout: str
    stderr: str
    exit_code: int


# ---- Prompt 6: tentativa de prova -----------------------------------------

class ProofAttempt(BaseModel):
    reduced_to_symbolic_identity: bool
    proof_status: str  # "proved" | "bounded_evidence_only" | "neither"
    informal_proof: Optional[str] = None
    final_code: str
    techniques_used: list[str]
    unresolved_step: Optional[str] = None
