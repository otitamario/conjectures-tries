"""
Orquestrador: dado um resultado de paper, percorre o pipeline
Prompt 1 -> Prompt 2 -> Prompt 3 (para cada um dos 3 melhores
candidatos) -> execução em sandbox -> repair loop -> (opcional)
Prompt 6 para os candidatos que passaram.

Cada candidato do top_three roda de forma independente e isolada
(seu próprio estado, seu próprio código, seu próprio sandbox) -- a
falha de um não impede os outros de seguir. A única decisão que
permanece manual é aceitar ou não a saída do Prompt 6 como prova.
"""

import env  # noqa: F401  (carrega .env antes do os.environ.get abaixo)
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from pipeline_states import State, can_transition
from schemas import (
    ResultAnalysis, ConjectureBatch, Conjecture,
    VerificationPlan, ExecutionResult, ProofAttempt,
)
from llm_calls import (
    run_prompt_1, run_prompt_2, run_prompt_3, run_prompt_6,
    run_prompt_3_with_repair, print_stats, get_stats,  # ← adicione get_stats
)

from sandbox_executor import run_candidate_code

MAX_REPAIR_ATTEMPTS = 3
REPAIR_MODEL = os.environ.get("PIPELINE_REPAIR_MODEL", "claude-haiku-4-5-20251001")


@dataclass
class PipelineRun:
    """Fase 1: única por paper -- análise + geração de conjecturas."""
    theorem_statement: str
    context: str = ""
    # metadados de origem do paper, usados só para persistência/rastreio
    source: str | None = None          # "zbmath" | "arxiv" | "manual"
    external_id: str | None = None     # id na fonte (zbMATH doc id, arXiv id)
    paper_title: str | None = None
    paper_link: str | None = None
    state: State = State.FETCHED
    analysis: ResultAnalysis | None = None
    conjectures: ConjectureBatch | None = None
    log: list[str] = field(default_factory=list)

    def _transition(self, target: State):
        if not can_transition(self.state, target):
            raise RuntimeError(f"Transição inválida: {self.state} -> {target}")
        self.log.append(f"{self.state.name} -> {target.name}")
        self.state = target


@dataclass
class CandidateRun:
    """Fase 2: uma instância por candidato do top_three (ou pelo
    resultado original, se preferir verificar sem gerar variações)."""
    conjecture: Conjecture
    state: State = State.CONJECTURES_GENERATED
    plan: VerificationPlan | None = None
    execution: ExecutionResult | None = None
    proof: ProofAttempt | None = None
    repair_attempts: int = 0
    log: list[str] = field(default_factory=list)

    def _transition(self, target: State):
        if not can_transition(self.state, target):
            raise RuntimeError(f"Transição inválida: {self.state} -> {target}")
        self.log.append(f"{self.state.name} -> {target.name}")
        self.state = target


def run_analysis_and_conjectures(run: PipelineRun) -> PipelineRun:
    run.analysis = run_prompt_1(run.theorem_statement, run.context)
    if not run.analysis.in_scope:
        run._transition(State.OUT_OF_SCOPE)
        return run
    run._transition(State.ANALYZED)

    run.conjectures = run_prompt_2(run.analysis)
    run._transition(State.CONJECTURES_GENERATED)
    return run


def _select_top_three(batch: ConjectureBatch) -> list[Conjecture]:
    by_name = {c.name: c for c in batch.candidates}
    selected = [by_name[name] for name in batch.top_three if name in by_name]
    missing = set(batch.top_three) - by_name.keys()
    if missing:
        # o modelo às vezes erra o nome exato ao listar top_three;
        # não travamos o pipeline por isso, só avisamos.
        print(f"[aviso] nomes em top_three sem candidato correspondente: {missing}")
    return selected


def run_candidate(candidate: CandidateRun) -> CandidateRun:
    """
    Roda Prompt 3 -> execução -> repair loop com escalonamento automático.
    
    Estratégia de repair:
      - Tentativas 1-2: K2.5 (econômico)
      - Tentativa 3 (última): K2.6 (máxima qualidade)
    """
    statement = candidate.conjecture.statement

    # Primeira tentativa: K2.6
    candidate.plan = run_prompt_3(statement)
    candidate._transition(State.PLAN_GENERATED)

    candidate.execution = run_candidate_code(candidate.plan.code)
    candidate._transition(
        State.EXECUTED_PASS if candidate.execution.passed else State.EXECUTED_FAIL
    )

    # Repair loop com escalonamento automático
    while (not candidate.execution.passed) and candidate.repair_attempts < MAX_REPAIR_ATTEMPTS:
        candidate.repair_attempts += 1
        
        print(f"\n  [REPAIR] Tentativa {candidate.repair_attempts}/{MAX_REPAIR_ATTEMPTS} "
              f"para '{candidate.conjecture.name}'")
        
        # NOVO: usa run_prompt_3_with_repair com escalonamento automático
        candidate.plan = run_prompt_3_with_repair(
            selected_statement=statement,
            previous_error=candidate.execution.stderr,
            repair_attempt=candidate.repair_attempts,
            max_repairs=MAX_REPAIR_ATTEMPTS,
        )
        
        candidate.execution = run_candidate_code(candidate.plan.code)
        candidate._transition(State.REPAIR_ATTEMPTED)
        candidate._transition(
            State.EXECUTED_PASS if candidate.execution.passed else State.EXECUTED_FAIL
        )

    if not candidate.execution.passed:
        candidate._transition(State.MAX_REPAIRS_EXCEEDED)
    
    return candidate

def run_candidate_proof(candidate: CandidateRun, theorem_statement: str) -> CandidateRun:
    """Prompt 6 -- só sobre candidatos que já passaram (EXECUTED_PASS)."""
    if candidate.state != State.EXECUTED_PASS:
        raise RuntimeError(
            f"Candidato '{candidate.conjecture.name}' não passou na verificação; "
            "Prompt 6 não se aplica."
        )
    candidate.proof = run_prompt_6(candidate.plan.code, theorem_statement)
    candidate._transition(State.PROOF_ATTEMPTED)
    candidate._transition(State.DONE)
    return candidate


def _process_candidate(
    conjecture: Conjecture, theorem_statement: str, attempt_proof: bool
) -> CandidateRun:
    candidate = CandidateRun(conjecture=conjecture)
    try:
        run_candidate(candidate)
        if attempt_proof and candidate.state == State.EXECUTED_PASS:
            run_candidate_proof(candidate, theorem_statement)
    except Exception as e:
        candidate.log.append(f"ERRO não tratado: {e}")
    return candidate


def run_top_candidates(
    run: PipelineRun, attempt_proof: bool = False, max_workers: int = 3
) -> list[CandidateRun]:
    """
    Ponto de entrada principal para o fluxo automatizado: pega os 3
    melhores candidatos do Prompt 2 e roda Prompt 3 + execução +
    repair loop para cada um EM PARALELO (threads) -- as chamadas de
    API e o sandbox de cada candidato são independentes entre si.

    Se attempt_proof=True, roda também o Prompt 6 (como rascunho)
    para os que passaram -- trate proof.proof_status como rascunho a
    revisar, nunca como prova aceita automaticamente.

    A ordem dos resultados retornados não é garantida (depende de
    qual thread termina primeiro); use candidate.conjecture.name para
    identificar cada um.
    """
    if run.conjectures is None:
        raise RuntimeError("Rode run_analysis_and_conjectures antes.")

    candidates = _select_top_three(run.conjectures)
    results: list[CandidateRun] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_candidate, c, run.theorem_statement, attempt_proof): c
            for c in candidates
        }
        for future in as_completed(futures):
            conjecture = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                # só deveria acontecer por bug em _process_candidate, já que
                # ele mesmo captura exceções de run_candidate/run_candidate_proof
                failed = CandidateRun(conjecture=conjecture)
                failed.log.append(f"ERRO na thread: {e}")
                results.append(failed)

    return results


def summarize(results: list[CandidateRun]) -> str:
    """Versão melhorada com estatísticas de uso da API."""
    lines = []
    for c in results:
        status = c.state.name
        proof_note = ""
        if c.proof is not None:
            proof_note = f" | prova: {c.proof.proof_status}"
        lines.append(f"- {c.conjecture.name}: {status} (reparos: {c.repair_attempts}){proof_note}")
    
    # Adiciona estatísticas de uso da API
    lines.append("\n" + "=" * 50)
    lines.append("ESTATÍSTICAS DE USO DA API:")
    for model, stat in get_stats().items():
        if stat.calls > 0:
            lines.append(f"  {model}: {stat.calls} chamadas, "
                        f"${stat.estimated_cost:.4f} estimado")
    
    return "\n".join(lines)