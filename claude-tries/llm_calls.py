"""
Uma função por prompt do workflow. Cada função:
  1. monta o prompt (mesmo texto do documento original),
  2. injeta um bloco de instrução de saída (forçar JSON puro),
  3. chama a API,
  4. valida com o schema pydantic correspondente,
  5. levanta exceção clara se a validação falhar (para o orquestrador decidir re-tentar).

Ajuste `call_model` para o seu client real (Anthropic SDK).
"""

import env  # noqa: F401  (carrega .env antes de qualquer os.environ.get abaixo)
import json
import os
from anthropic import Anthropic
from schemas import (
    ExtractedResultsBatch, ResultAnalysis, ConjectureBatch, VerificationPlan, ProofAttempt,
)

JSON_ONLY_SUFFIX = (
    "\n\nResponda SOMENTE com um objeto JSON válido, sem texto antes ou depois, "
    "sem blocos de código markdown (sem ```)."
)

MODEL = os.environ.get("PIPELINE_MODEL", "claude-sonnet-5")
_client = Anthropic()  # lê ANTHROPIC_API_KEY do ambiente


def call_model(system_prompt: str, user_prompt: str, model: str = MODEL) -> str:
    response = _client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _parse(raw: str, schema):
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    return schema.model_validate(json.loads(cleaned))


def run_prompt_0(paper_text: str, max_chars: int = 150_000) -> ExtractedResultsBatch:
    """
    Varre o texto completo do paper e extrai enunciados de teoremas,
    lemas e proposições que o PRÓPRIO paper prova (não resultados
    citados de outros trabalhos, e não conjecturas em aberto),
    restritos a combinatória e/ou teoria dos números elementar.

    Trunca em max_chars por segurança (a maioria dos papers cabe
    tranquilamente; se o seu tiver apêndices enormes, considere
    passar só as seções de resultados principais).
    """
    truncated = paper_text[:max_chars]
    system = (
        "Você é um assistente de pesquisa em combinatória e teoria dos números. "
        "Sua tarefa é apenas localizar e extrair enunciados formais do texto — "
        "não analisá-los ainda, não gerar conjecturas."
    )
    user = f"""Aqui está o texto (possivelmente truncado) de um paper:

---
{truncated}
---

Extraia TODOS os teoremas, lemas ou proposições que:
1. são efetivamente PROVADOS neste paper (não citados de outro trabalho, não
   deixados como conjectura em aberto pelos próprios autores);
2. pertencem a combinatória e/ou teoria dos números elementar/enumerativa
   (enumeração, funções geradoras, partições, grafos, permutações,
   congruências, primos, sequências tipo Fibonacci/Lucas, equações
   diofantinas). Marque in_scope=false para os que não pertencerem, mas
   ainda assim inclua-os na lista.

Para cada um, preencha: label (ex. "Theorem 3.2"), statement (o enunciado
completo, autocontido), context (definições/notação necessárias para
entender o statement sem olhar o resto do paper), in_scope,
is_proved_in_paper.

Não inclua lemas técnicos triviais de manipulação algébrica sem interesse
combinatório/aritmético próprio. Priorize os resultados principais do paper."""
    raw = call_model(system, user + JSON_ONLY_SUFFIX)
    return _parse(raw, ExtractedResultsBatch)


def run_prompt_1(theorem_statement: str, context: str = "") -> ResultAnalysis:
    system = "Você é um assistente de pesquisa em combinatória e teoria dos números, com verificação computacional via Sympy."
    user = f"""Aqui está o resultado:

{theorem_statement}

Contexto opcional:
{context}

Analise o resultado antes de tentar formalizá-lo, retornando os campos do schema ResultAnalysis
(restatement, subareas, quantified_objects, explicit_hypotheses, implicit_assumptions,
conclusion, likely_proof_mechanism, checkability, needs_networkx, ambiguities,
formalized_statement, in_scope)."""
    raw = call_model(system, user + JSON_ONLY_SUFFIX)
    return _parse(raw, ResultAnalysis)


def run_prompt_2(analysis: ResultAnalysis) -> ConjectureBatch:
    system = "Você é um assistente de pesquisa em combinatória e teoria dos números."
    user = f"""A partir desta análise:

{analysis.model_dump_json(indent=2)}

Gere no máximo dez conjecturas candidatas (campo `candidates`), cada uma com name,
changed_component, statement, motivation, status, status_reason,
counterexample_direction, checkability, needs_networkx. Ao final, preencha
`top_three` com os nomes dos três candidatos mais promissores, em ordem."""
    raw = call_model(system, user + JSON_ONLY_SUFFIX)
    return _parse(raw, ConjectureBatch)


def run_prompt_3(selected_statement: str, definitions: str = "", model: str = MODEL) -> VerificationPlan:
    system = "Você é um assistente de pesquisa que formaliza conjecturas como verificações em Sympy/networkx."
    user = f"""Formalize esta afirmação como algo checável com Sympy (e networkx se envolver grafos):

{selected_statement}

Definições informais usadas:
{definitions}

Preencha o schema VerificationPlan: objects_and_representation, sympy_networkx_functions,
reduces_to_symbolic_identity, ambiguities_to_resolve, code (código Python completo e
executável, com print/assert de PASS/FAIL), correspondence_table, verification_risks."""
    raw = call_model(system, user + JSON_ONLY_SUFFIX, model=model)
    return _parse(raw, VerificationPlan)


def run_prompt_6(working_code: str, mathematical_statement: str) -> ProofAttempt:
    system = "Você é um assistente de pesquisa tentando elevar evidência computacional a uma prova genuína."
    user = f"""O seguinte código tem uma verificação Sympy/networkx funcionando:

{working_code}

O resultado informal correspondente é:
{mathematical_statement}

Tente avançar de evidência computacional para uma prova real. NÃO declare "provado" com
base apenas em checagem finita. Preencha o schema ProofAttempt: reduced_to_symbolic_identity,
proof_status ("proved" | "bounded_evidence_only" | "neither"), informal_proof, final_code,
techniques_used, unresolved_step."""
    raw = call_model(system, user + JSON_ONLY_SUFFIX)
    return _parse(raw, ProofAttempt)
