"""
Persistência local em SQLite. Um arquivo .db, sem servidor, sem
dependência externa (usa sqlite3 da stdlib).

Schema:
  papers            -- um por PipelineRun (fonte, link, teorema original)
  analyses          -- resultado do Prompt 1 (1:1 com paper)
  conjectures       -- candidatos do Prompt 2 (N:1 com paper), com flag is_top_three
  verification_plans-- código gerado no Prompt 3 (N:1 com conjecture -- uma por tentativa/reparo)
  executions        -- resultado real de rodar o código (N:1 com plan)
  proofs            -- resultado do Prompt 6 (1:1 com conjecture, quando aplicável)

Campos que são listas/dicts nos schemas pydantic (subareas,
hypotheses, correspondence_table etc.) são guardados como JSON em
colunas TEXT -- simples de ler de volta com json.loads, sem precisar
de tabelas auxiliares para algo que você não vai consultar por
dentro do SQL.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from orchestrator import PipelineRun, CandidateRun

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    external_id TEXT,
    title TEXT,
    link TEXT,
    theorem_statement TEXT NOT NULL,
    context TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    in_scope INTEGER,
    restatement TEXT,
    subareas TEXT,
    quantified_objects TEXT,
    explicit_hypotheses TEXT,
    implicit_assumptions TEXT,
    conclusion TEXT,
    likely_proof_mechanism TEXT,
    checkability TEXT,
    needs_networkx INTEGER,
    ambiguities TEXT,
    formalized_statement TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conjectures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    name TEXT,
    changed_component TEXT,
    statement TEXT,
    motivation TEXT,
    status TEXT,
    status_reason TEXT,
    counterexample_direction TEXT,
    checkability TEXT,
    needs_networkx INTEGER,
    is_top_three INTEGER DEFAULT 0,
    final_state TEXT,
    repair_attempts INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conjecture_id INTEGER NOT NULL REFERENCES conjectures(id),
    attempt_number INTEGER NOT NULL,
    objects_and_representation TEXT,
    sympy_networkx_functions TEXT,
    reduces_to_symbolic_identity INTEGER,
    ambiguities_to_resolve TEXT,
    code TEXT,
    correspondence_table TEXT,
    verification_risks TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES verification_plans(id),
    passed INTEGER,
    stdout TEXT,
    stderr TEXT,
    exit_code INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proofs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conjecture_id INTEGER NOT NULL REFERENCES conjectures(id),
    reduced_to_symbolic_identity INTEGER,
    proof_status TEXT,
    informal_proof TEXT,
    final_code TEXT,
    techniques_used TEXT,
    unresolved_step TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS explored_papers (
    external_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    in_scope_results_found INTEGER NOT NULL,
    saved_anything INTEGER NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection(db_path: str = "conjectures.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = "conjectures.db") -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def save_paper(conn: sqlite3.Connection, run: PipelineRun) -> int:
    cur = conn.execute(
        """INSERT INTO papers (source, external_id, title, link, theorem_statement, context, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run.source, run.external_id, run.paper_title, run.paper_link,
         run.theorem_statement, run.context, _now()),
    )
    return cur.lastrowid


def save_analysis(conn: sqlite3.Connection, paper_id: int, run: PipelineRun) -> int | None:
    if run.analysis is None:
        return None
    a = run.analysis
    cur = conn.execute(
        """INSERT INTO analyses (paper_id, in_scope, restatement, subareas, quantified_objects,
               explicit_hypotheses, implicit_assumptions, conclusion, likely_proof_mechanism,
               checkability, needs_networkx, ambiguities, formalized_statement, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (paper_id, int(a.in_scope), a.restatement, json.dumps(a.subareas),
         json.dumps(a.quantified_objects), json.dumps(a.explicit_hypotheses),
         json.dumps(a.implicit_assumptions), a.conclusion, a.likely_proof_mechanism,
         a.checkability.value, int(a.needs_networkx), json.dumps(a.ambiguities),
         a.formalized_statement, _now()),
    )
    return cur.lastrowid


def save_candidate(conn: sqlite3.Connection, paper_id: int, candidate: CandidateRun) -> int:
    c = candidate.conjecture
    cur = conn.execute(
        """INSERT INTO conjectures (paper_id, name, changed_component, statement, motivation,
               status, status_reason, counterexample_direction, checkability, needs_networkx,
               is_top_three, final_state, repair_attempts, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
        (paper_id, c.name, c.changed_component, c.statement, c.motivation,
         c.status.value, c.status_reason, c.counterexample_direction,
         c.checkability.value, int(c.needs_networkx),
         candidate.state.name, candidate.repair_attempts, _now()),
    )
    conjecture_id = cur.lastrowid

    # cada tentativa de plano/execução vira uma linha própria (attempt_number
    # 0 = primeira tentativa, 1..N = reparos) -- assim o histórico de reparo
    # fica auditável, não só o resultado final.
    if candidate.plan is not None:
        p = candidate.plan
        cur2 = conn.execute(
            """INSERT INTO verification_plans (conjecture_id, attempt_number,
                   objects_and_representation, sympy_networkx_functions,
                   reduces_to_symbolic_identity, ambiguities_to_resolve, code,
                   correspondence_table, verification_risks, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conjecture_id, candidate.repair_attempts, p.objects_and_representation,
             json.dumps(p.sympy_networkx_functions), int(p.reduces_to_symbolic_identity),
             json.dumps(p.ambiguities_to_resolve), p.code,
             json.dumps(p.correspondence_table), json.dumps(p.verification_risks), _now()),
        )
        plan_id = cur2.lastrowid

        if candidate.execution is not None:
            e = candidate.execution
            conn.execute(
                """INSERT INTO executions (plan_id, passed, stdout, stderr, exit_code, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (plan_id, int(e.passed), e.stdout, e.stderr, e.exit_code, _now()),
            )

    if candidate.proof is not None:
        pr = candidate.proof
        conn.execute(
            """INSERT INTO proofs (conjecture_id, reduced_to_symbolic_identity, proof_status,
                   informal_proof, final_code, techniques_used, unresolved_step, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (conjecture_id, int(pr.reduced_to_symbolic_identity), pr.proof_status,
             pr.informal_proof, pr.final_code, json.dumps(pr.techniques_used),
             pr.unresolved_step, _now()),
        )

    return conjecture_id


def mark_explored(
    db_path: str, external_id: str, source: str, in_scope_results_found: int, saved_anything: bool
) -> None:
    """Registra que um paper já foi tentado, MESMO quando não rendeu
    nada aproveitável -- assim a próxima busca não paga de novo o
    custo de Prompt 0 nele."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO explored_papers (external_id, source, attempted_at,
                   in_scope_results_found, saved_anything)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(external_id) DO UPDATE SET
                   attempted_at=excluded.attempted_at,
                   in_scope_results_found=excluded.in_scope_results_found,
                   saved_anything=excluded.saved_anything""",
            (external_id, source, _now(), int(in_scope_results_found), int(saved_anything)),
        )


def get_explored_ids(db_path: str = "conjectures.db", source: str = "arxiv") -> set[str]:
    """Todo external_id já tentado (com ou sem sucesso) para uma fonte --
    use para filtrar antes de mandar um lote novo pro pipeline."""
    if not os.path.exists(db_path):
        return set()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT external_id FROM explored_papers WHERE source = ?", (source,)
        ).fetchall()
    return {row[0] for row in rows}


def get_processed_external_ids(db_path: str = "conjectures.db", source: str = "arxiv") -> set[str]:
    """Retorna o conjunto de external_id (ex. ids do arXiv) já salvos
    no banco para uma fonte -- usado para pular papers já explorados
    em chamadas futuras de descoberta (zbMATH/arXiv)."""
    if not os.path.exists(db_path):
        return set()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT external_id FROM papers WHERE source = ? AND external_id IS NOT NULL",
            (source,),
        ).fetchall()
    return {row[0] for row in rows}


def save_run(run: PipelineRun, results: list[CandidateRun], db_path: str = "conjectures.db") -> int:
    """Salva um PipelineRun completo (paper + análise + todos os
    candidatos processados) em uma única transação."""
    with get_connection(db_path) as conn:
        paper_id = save_paper(conn, run)
        save_analysis(conn, paper_id, run)
        for candidate in results:
            save_candidate(conn, paper_id, candidate)
        return paper_id
