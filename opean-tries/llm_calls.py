"""
Uma função por prompt do workflow. Cada função:
  1. monta o prompt (mesmo texto do documento original),
  2. injeta um bloco de instrução de saída (forçar JSON puro),
  3. Chama a API da OpenAI.
  4. valida com o schema pydantic correspondente,
  5. levanta exceção clara se a validação falhar (para o orquestrador decidir re-tentar).


Otimizado para context caching e escalonamento automático de modelos.

CORREÇÕES APLICADAS:
- Tratamento de JSON truncado (max_tokens atingido no meio da string)
- Tratamento de respostas vazias (prompt muito grande)
- Aumento de max_tokens para 32768
- Redução do max_chars no Prompt 0 para 15000
- Heurística agressiva de reparo de JSON truncado
- Logging detalhado de finish_reason
- NORMALIZAÇÃO DE CAMPOS: likely_proof_mechanism (lista→string), checkability (texto→enum)
"""

import env  # noqa: F401  (carrega .env antes de qualquer os.environ.get abaixo)
import json
import os
import random
import time
from dataclasses import dataclass
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

class EmptyModelResponseError(RuntimeError):
    """A API respondeu, mas sem conteúdo útil."""
    pass

from schemas import (
    ExtractedResultsBatch, ResultAnalysis, ConjectureBatch, VerificationPlan, ProofAttempt,
)

JSON_ONLY_SUFFIX = (
    "\n\nResponda SOMENTE com um objeto JSON válido (não uma lista), "
    "sem texto antes ou depois, sem blocos de código markdown (sem ```). "
    "O JSON deve ser um objeto com as chaves especificadas no schema."
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO OPENAI - ESTRATÉGIA HÍBRIDA COM ESCALONAMENTO
# ─────────────────────────────────────────────────────────────────────────────

# Modelos por tarefa (configurável via variáveis de ambiente)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY não encontrada no arquivo .env"
    )


# Modelos disponíveis na SUA conta (pelo teste)
MODEL_EXTRACTION = os.environ.get(
    "PIPELINE_MODEL_P0",
    "gpt-5-nano",
)

MODEL_ANALYSIS = os.environ.get(
    "PIPELINE_MODEL_P1",
    "gpt-5-mini",
)

MODEL_CONJECTURES = os.environ.get(
    "PIPELINE_MODEL_P2",
    "gpt-5-mini",
)

MODEL_VERIFICATION = os.environ.get(
    "PIPELINE_MODEL_P3",
    "gpt-5-mini",
)

MODEL_REPAIR_PRIMARY = os.environ.get(
    "PIPELINE_MODEL_REPAIR",
    "gpt-5-nano",
)

MODEL_REPAIR_ESCALATION = os.environ.get(
    "PIPELINE_MODEL_REPAIR_LAST",
    "gpt-5-mini",
)

MODEL_PROOF = os.environ.get(
    "PIPELINE_MODEL_P6",
    "gpt-5.1",
)

MODEL_EXTRACTION_FALLBACK = os.environ.get(
    "PIPELINE_MODEL_P0_FALLBACK",
    "gpt-5-mini",
)

MODEL_ANALYSIS_FALLBACK = os.environ.get(
    "PIPELINE_MODEL_P1_FALLBACK",
    "gpt-5-mini",
)

MODEL_VERIFICATION_FALLBACK = os.environ.get(
    "PIPELINE_MODEL_P3_FALLBACK",
    "gpt-5.1",
)

# Número de repairs antes de escalar para o modelo mais forte
REPAIR_ESCALATION_THRESHOLD = int(os.environ.get("REPAIR_ESCALATION_THRESHOLD", "2"))

# Cliente sem retries internos: os retries são controlados abaixo, com logging claro.
_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "300")),
    max_retries=0,
)

# Parâmetros de resiliência
API_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "300"))
API_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
API_RETRY_BASE_SECONDS = float(os.environ.get("OPENAI_RETRY_BASE_SECONDS", "5"))
API_RETRY_MAX_SECONDS = float(os.environ.get("OPENAI_RETRY_MAX_SECONDS", "60"))

# Parâmetros padrão de geração
DEFAULT_MAX_TOKENS = 8_000

# Limites por etapa
MAX_TOKENS_EXTRACTION = int(os.environ.get("MAX_TOKENS_EXTRACTION", "8000"))
MAX_TOKENS_ANALYSIS = int(os.environ.get("MAX_TOKENS_ANALYSIS", "6000"))
MAX_TOKENS_CONJECTURES = int(os.environ.get("MAX_TOKENS_CONJECTURES", "10000"))
MAX_TOKENS_VERIFICATION = int(os.environ.get("MAX_TOKENS_VERIFICATION", "6000"))
MAX_TOKENS_REPAIR = int(os.environ.get("MAX_TOKENS_REPAIR", "4000"))
MAX_TOKENS_PROOF = int(os.environ.get("MAX_TOKENS_PROOF", "12000"))


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS OTIMIZADOS PARA CONTEXT CACHING
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_EXTRACTOR = (
    "Você é um assistente de pesquisa em combinatória e teoria dos números. "
    "Sua tarefa é apenas localizar e extrair enunciados formais do texto — "
    "não analisá-los ainda, não gerar conjecturas."
)

SYSTEM_MATH_RESEARCH = (
    "Você é um assistente de pesquisa em combinatória e teoria dos números, "
    "com verificação computacional via Sympy."
)

SYSTEM_VERIFICATION = (
    "Você é um assistente de pesquisa que formaliza conjecturas como verificações em Sympy/networkx."
)

SYSTEM_PROVER = (
    "Você é um assistente de pesquisa tentando elevar evidência computacional a uma prova genuína."
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTATÍSTICAS DE USO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelUsageStats:
    """Estatísticas de uso por modelo para análise de custo-efetividade."""
    model_name: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    successes: int = 0
    failures: int = 0

    @property
    def estimated_cost(self) -> float:
        """Custo estimado em USD (sem cache)."""
        prices = {
            # Atualize estes valores quando os preços oficiais mudarem.
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5.1": {"input": 1.25, "output": 10.00},
        }
        price = prices.get(self.model_name)

        if price is None:
            return 0.0
        return (self.input_tokens / 1_000_000 * price["input"] + 
                self.output_tokens / 1_000_000 * price["output"])


# Estatísticas globais
_stats: dict[str, ModelUsageStats] = {
    model: ModelUsageStats(model_name=model)
    for model in {
        MODEL_EXTRACTION,
        MODEL_ANALYSIS,
        MODEL_CONJECTURES,
        MODEL_VERIFICATION,
        MODEL_REPAIR_PRIMARY,
        MODEL_REPAIR_ESCALATION,
        MODEL_PROOF,
        MODEL_EXTRACTION_FALLBACK,
        MODEL_ANALYSIS_FALLBACK,
        MODEL_VERIFICATION_FALLBACK,
    }
}


def get_stats() -> dict[str, ModelUsageStats]:
    """Retorna estatísticas de uso para análise."""
    return _stats


def print_stats() -> None:
    """Imprime relatório de uso e custo estimado."""
    print("\n" + "=" * 70)
    print("RELATÓRIO DE USO DA API OPENAI")
    print("=" * 70)
    total_cost = 0
    for model, stat in _stats.items():
        if stat.calls == 0:
            continue
        cost = stat.estimated_cost
        total_cost += cost
        success_rate = stat.successes / stat.calls * 100 if stat.calls > 0 else 0
        print(f"\n{model}:")
        print(f"  Chamadas: {stat.calls}")
        print(f"  Input tokens: {stat.input_tokens:,}")
        print(f"  Output tokens: {stat.output_tokens:,}")
        print(f"  Sucessos: {stat.successes} | Falhas: {stat.failures}")
        print(f"  Taxa de sucesso: {success_rate:.1f}%")
        print(f"  Custo estimado: ${cost:.4f}")
    print(f"\nCUSTO TOTAL ESTIMADO: ${total_cost:.4f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO BASE DE CHAMADA COM LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def call_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    track_stats: bool = True,
    stage: str = "indefinida",
    fallback_model: str | None = None,
) -> str:
    """
    Chama a Chat Completions API com timeout, logging e retries explícitos.

    Política:
    - resposta vazia com fallback configurado: escala imediatamente;
    - timeout, conexão, rate limit e erro interno: retry no mesmo modelo;
    - resposta vazia sem fallback: retry normal até o limite.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    system_chars = len(system_prompt)
    user_chars = len(user_prompt)
    total_chars = system_chars + user_chars
    estimated_input_tokens = max(1, total_chars // 3)

    retryable_errors = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        EmptyModelResponseError,
    )

    print(
        f"  [PROMPT] etapa={stage} | "
        f"system={system_chars:,} chars | "
        f"user={user_chars:,} chars | "
        f"total={total_chars:,} chars | "
        f"tokens_estimados≈{estimated_input_tokens:,}"
    )

    for attempt in range(1, API_MAX_RETRIES + 1):
        start_time = time.time()
        print(
            f"  [API] etapa={stage} | modelo={model} | "
            f"tentativa={attempt}/{API_MAX_RETRIES} | "
            f"timeout={API_TIMEOUT_SECONDS:.0f}s | max_tokens={max_tokens}"
        )

        try:
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=API_TIMEOUT_SECONDS,
            )

            result = response.choices[0].message.content
            if not result or not result.strip():
                raise EmptyModelResponseError(
                    f"Resposta vazia na etapa {stage}, usando o modelo {model}."
                )

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(
                    f"  [AVISO] etapa={stage}: resposta truncada por limite "
                    f"de tokens; o JSON pode estar incompleto."
                )

            if track_stats and model in _stats:
                stat = _stats[model]
                stat.calls += 1
                if response.usage:
                    stat.input_tokens += response.usage.prompt_tokens
                    stat.output_tokens += response.usage.completion_tokens
                stat.successes += 1

            elapsed = time.time() - start_time
            actual_input = response.usage.prompt_tokens if response.usage else estimated_input_tokens
            actual_output = response.usage.completion_tokens if response.usage else len(result) // 3

            print(
                f"  [API] etapa={stage} | modelo={model} | OK em {elapsed:.1f}s | "
                f"input={actual_input} | output={actual_output} | finish={finish_reason}"
            )
            return result

        except retryable_errors as exc:
            elapsed = time.time() - start_time

            if track_stats and model in _stats:
                _stats[model].failures += 1

            if (
                isinstance(exc, EmptyModelResponseError)
                and fallback_model
                and fallback_model != model
            ):
                print(
                    f"  [API] etapa={stage} | modelo={model} retornou vazio "
                    f"após {elapsed:.1f}s. Escalando imediatamente para {fallback_model}."
                )
                return call_model(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=fallback_model,
                    max_tokens=max_tokens,
                    track_stats=track_stats,
                    stage=f"{stage}_fallback",
                    fallback_model=None,
                )

            if attempt >= API_MAX_RETRIES:
                print(
                    f"  [API] etapa={stage} | modelo={model} | "
                    f"FALHA DEFINITIVA após {attempt} tentativas: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            delay = min(
                API_RETRY_MAX_SECONDS,
                API_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            delay += random.uniform(0, min(2.0, delay * 0.2))
            print(
                f"  [API] etapa={stage} | erro transitório após {elapsed:.1f}s: "
                f"{type(exc).__name__}: {exc}. Nova tentativa em {delay:.1f}s."
            )
            time.sleep(delay)

        except Exception as exc:
            if track_stats and model in _stats:
                _stats[model].failures += 1
            elapsed = time.time() - start_time
            print(
                f"  [API] etapa={stage} | modelo={model} | "
                f"erro não transitório após {elapsed:.1f}s: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

    raise RuntimeError("Fluxo de retry terminou sem resposta nem exceção.")


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO DE REPAIR COM ESCALONAMENTO AUTOMÁTICO
# ─────────────────────────────────────────────────────────────────────────────

def run_prompt_3_with_repair(
    selected_statement: str,
    definitions: str = "",
    previous_error: str = "",
    repair_attempt: int = 0,
    max_repairs: int = 3,
) -> VerificationPlan:
    """
    Gera plano de verificação com escalonamento automático de modelo em repairs.
    """
    is_repair = repair_attempt > 0

    if not is_repair:
        model = MODEL_VERIFICATION
        user = f"""Formalize esta afirmação como algo checável com Sympy (e networkx se envolver grafos):

{selected_statement}

Definições informais usadas:
{definitions}

Preencha o schema VerificationPlan: objects_and_representation, sympy_networkxFunctions,
reduces_to_symbolic_identity, ambiguities_to_resolve, code (código Python completo e
executável, com print/assert de PASS/FAIL), correspondence_table, verification_risks."""

    else:
        remaining = max_repairs - repair_attempt + 1

        if remaining <= 1 and repair_attempt >= REPAIR_ESCALATION_THRESHOLD:
            model = MODEL_REPAIR_ESCALATION
            print(f"  [REPAIR] Escalonando para {model} (tentativa {repair_attempt}, "
                  f"threshold={REPAIR_ESCALATION_THRESHOLD})")
        else:
            model = MODEL_REPAIR_PRIMARY
            print(f"  [REPAIR] Usando {model} (tentativa {repair_attempt}, "
                  f"escala em {REPAIR_ESCALATION_THRESHOLD})")

        user = f"""A seguinte conjectura precisa ser verificada em Sympy/networkx:

{selected_statement}

Definições informais usadas:
{definitions}

[Código anterior falhou com:]
{previous_error}

[Corrija o código sem enfraquecer a afirmação nem estreitar o range testado.
Gere um novo código Python completo e executável.]

Preencha o schema VerificationPlan: objects_and_representation, sympy_networkxFunctions,
reduces_to_symbolic_identity, ambiguities_to_resolve, code, correspondence_table, verification_risks."""

    token_limit = (
        MAX_TOKENS_REPAIR
        if is_repair
        else MAX_TOKENS_VERIFICATION
    )

    raw = call_model(
        SYSTEM_VERIFICATION,
        user + JSON_ONLY_SUFFIX,
        model=model,
        max_tokens=token_limit,
        stage="prompt_3_repair" if is_repair else "prompt_3_verification",
        fallback_model=(
            None if is_repair else MODEL_VERIFICATION_FALLBACK
        ),
    )
    return _parse(raw, VerificationPlan)


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE PARSE ROBUSTO + NORMALIZAÇÃO DE CAMPOS
# ─────────────────────────────────────────────────────────────────────────────

def _fix_truncated_json(text: str) -> str:
    """
    Heurística agressiva para tentar consertar JSONs truncados pelo max_tokens.
    Completa strings não fechadas, arrays/objetos não fechados, etc.
    """
    result = text.strip()
    if not result:
        return "{}"

    # Se já termina com } ou ], pode ser válido
    if result.rstrip().endswith(("}", "]")):
        try:
            json.loads(result)
            return result
        except json.JSONDecodeError:
            pass  # continua tentando consertar

    # Completa strings não terminadas
    in_string = False
    escaped = False
    for i, ch in enumerate(result):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        result += '"'

    # Remove trailing vírgulas, dois-pontos soltos, etc.
    result = result.rstrip()
    while result and result[-1] in ",:":
        result = result[:-1].rstrip()

    # Fecha estruturas abertas usando stack (ordem LIFO)
    stack = []
    in_str = False
    esc = False
    for ch in result:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    # Fecha no sentido inverso
    for opener in reversed(stack):
        if opener == "{":
            result += "}"
        else:
            result += "]"

    # Se ainda não é um objeto JSON válido, tenta extrair objeto interno
    try:
        json.loads(result)
    except json.JSONDecodeError:
        start = result.find("{")
        end = result.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                inner = result[start:end+1]
                json.loads(inner)
                result = inner
            except json.JSONDecodeError:
                pass

    return result


def _normalize_field(data: dict, field_name: str, target_type: str = "string") -> None:
    """Normaliza um campo para o tipo esperado pelo schema."""
    if field_name not in data:
        return

    value = data[field_name]

    if target_type == "string" and isinstance(value, list):
        # Converte lista para string concatenada
        data[field_name] = " | ".join(str(v) for v in value)
        print(f"  [NORMALIZAÇÃO] {field_name}: lista → string")

    elif target_type == "list" and isinstance(value, str):
        # Converte string para lista com um elemento
        data[field_name] = [value]
        print(f"  [NORMALIZAÇÃO] {field_name}: string → lista")


def _normalize_checkability(data: dict) -> None:
    """Normaliza o campo checkability para um dos valores do enum válidos."""
    if "checkability" not in data:
        return

    raw = str(data["checkability"]).strip().upper()

    # Mapeamento de sinônimos/comportamentos para valores do enum
    mappings = {
        "SYMBOLIC": ["SYMBOLIC", "SYMBOLICALLY", "ALGEBRAIC", "CLOSED FORM", "ANALYTIC"],
        "BOUNDED_COMPUTATIONAL": [
            "BOUNDED", "COMPUTATIONAL", "COMPUTABLE", "BRUTE FORCE", 
            "FINITE", "ENUMERATIVE", "EXHAUSTIVE", "ALGORITHMIC"
        ],
        "NOT_READILY_CHECKABLE": [
            "NOT READILY", "NOT_CHECKABLE", "UNCHECKABLE", "DIFFICULT",
            "HARD", "COMPLEX", "NON-COMPUTABLE", "THEORETICAL"
        ],
    }

    for canonical, synonyms in mappings.items():
        if any(syn in raw for syn in synonyms):
            if data["checkability"] != canonical:
                print(f"  [NORMALIZAÇÃO] checkability: '{data['checkability']}' → '{canonical}'")
                data["checkability"] = canonical
            return

    # Se não conseguiu mapear, força para BOUNDED_COMPUTATIONAL como default seguro
    print(f"  [NORMALIZAÇÃO] checkability: '{data['checkability']}' não mapeado → 'BOUNDED_COMPUTATIONAL' (default)")
    data["checkability"] = "BOUNDED_COMPUTATIONAL"


def _normalize_status(data: dict) -> None:
    """Normaliza o campo status para um dos valores do enum válidos."""
    if "status" not in data:
        return

    raw = str(data["status"]).strip().upper()

    mappings = {
        "PLAUSIBLY_TRUE": ["PLAUSIBLY_TRUE", "TRUE", "LIKELY", "PROBABLE", "PLAUSIBLE"],
        "PLAUSIBLY_FALSE": ["PLAUSIBLY_FALSE", "FALSE", "UNLIKELY", "IMPROBABLE"],
        "UNCERTAIN": ["UNCERTAIN", "UNKNOWN", "OPEN", "UNSURE", "NEUTRAL"],
    }

    for canonical, synonyms in mappings.items():
        if any(syn in raw for syn in synonyms):
            if data["status"] != canonical:
                print(f"  [NORMALIZAÇÃO] status: '{data['status']}' → '{canonical}'")
                data["status"] = canonical
            return

    # Default seguro
    print(f"  [NORMALIZAÇÃO] status: '{data['status']}' não mapeado → 'UNCERTAIN' (default)")
    data["status"] = "UNCERTAIN"


def _normalize_result_analysis(data: dict) -> None:
    """Aplica todas as normalizações necessárias para ResultAnalysis."""
    _normalize_field(data, "likely_proof_mechanism", "string")
    _normalize_field(data, "restatement", "string")
    _normalize_field(data, "conclusion", "string")
    _normalize_field(data, "formalized_statement", "string")
    _normalize_checkability(data)

    # Garante que campos de lista sejam listas
    for list_field in ["subareas", "quantified_objects", "explicit_hypotheses", 
                       "implicit_assumptions", "ambiguities"]:
        _normalize_field(data, list_field, "list")


def _normalize_conjecture(data: dict) -> None:
    """Aplica normalizações para Conjecture."""
    _normalize_checkability(data)
    _normalize_status(data)


def _normalize_all(data: dict, schema_name: str) -> None:
    """Aplica normalizações específicas baseadas no schema."""
    if schema_name == "ResultAnalysis":
        _normalize_result_analysis(data)
    elif schema_name == "ConjectureBatch":
        if "candidates" in data and isinstance(data["candidates"], list):
            for c in data["candidates"]:
                _normalize_conjecture(c)
    elif schema_name == "Conjecture":
        _normalize_conjecture(data)


def _parse(raw: str, schema):
    """
    Parseia JSON com tratamento de:
    - JSON truncado (max_tokens atingido no meio da string)
    - Markdown code blocks (```json ... ```)
    - Respostas vazias ou não-JSON
    - Listas embrulhadas em objeto
    - NORMALIZAÇÃO DE CAMPOS para compatibilidade com schema Pydantic
    """
    if not raw or not raw.strip():
        raise ValueError("Resposta da API está vazia (raw='' ou whitespace-only)")

    # Remove markdown code blocks
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Tenta parsear o JSON
    data = None
    last_error = None
    used_repair = False

    # Tentativa 1: JSON direto
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e1:
        last_error = e1

        # Tentativa 2: Extrai objeto JSON mais completo possível
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(cleaned[start:end+1])
        except json.JSONDecodeError as e2:
            last_error = e2

            # Tentativa 3: Tenta truncar no último ponto válido
            try:
                for i in range(len(cleaned), 0, -1):
                    try:
                        data = json.loads(cleaned[:i])
                        break
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass

    if data is None:
        # Último recurso: heurística de reparo de JSON truncado
        try:
            fixed = _fix_truncated_json(cleaned)
            data = json.loads(fixed)
            used_repair = True
            print(f"  [REPARO] JSON truncado consertado com heurística.")
        except Exception as e3:
            raise ValueError(
                f"Não foi possível parsear JSON da resposta da API. "
                f"Erro original: {last_error}. "
                f"Primeiros 500 chars da resposta: {cleaned[:500]!r}"
            ) from last_error

    # ── Normalização de formatos alternativos ──
    if isinstance(data, dict):
        if "theorems" in data and "results" not in data:
            data["results"] = data.pop("theorems")
        if "candidates" in data and "top_three" not in data:
            data["top_three"] = [c.get("name", "") for c in data["candidates"][:3]]

    elif isinstance(data, list):
        if schema.__name__ == 'ExtractedResultsBatch':
            data = {"results": data}
        elif schema.__name__ == 'ConjectureBatch':
            data = {
                "candidates": data,
                "top_three": [c.get("name", "") for c in data[:3]]
            }

    # ── NORMALIZAÇÃO DE CAMPOS para compatibilidade com schema ──
    if isinstance(data, dict):
        _normalize_all(data, schema.__name__)

        # Normaliza também campos aninhados (ex: candidates dentro de ConjectureBatch)
        if "candidates" in data and isinstance(data["candidates"], list):
            for c in data["candidates"]:
                if isinstance(c, dict):
                    _normalize_all(c, "Conjecture")
        if "results" in data and isinstance(data["results"], list):
            for r in data["results"]:
                if isinstance(r, dict):
                    _normalize_all(r, "ExtractedResult")

    return schema.model_validate(data)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS DO WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def run_prompt_0(paper_text: str, max_chars: int = 15_000) -> ExtractedResultsBatch:
    """
    Extrai teoremas do paper. REDUZIDO para 15.000 chars para evitar
    que o prompt ocupe todo o contexto e deixe 0 tokens para a resposta.
    """
    truncated = paper_text[:max_chars]
    user = f"""Aqui está o texto (possivelmente truncado) de um paper:

---
{truncated}
---

Extraia os PRINCIPAIS teoremas, lemas ou proposições (máximo 15) que:
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
combinatório/aritmético próprio. Priorize os resultados principais do paper.

IMPORTANTE: Retorne um objeto JSON com a chave EXATA "results" (não "theorems"), assim:
{{"results": [{{"label": "Theorem 1.1", "statement": "...", "context": "...", "in_scope": true, "is_proved_in_paper": true}}]}}"""

    raw = call_model(
        SYSTEM_EXTRACTOR,
        user + JSON_ONLY_SUFFIX,
        model=MODEL_EXTRACTION,
        max_tokens=MAX_TOKENS_EXTRACTION,
        stage="prompt_0_extraction",
        fallback_model=MODEL_EXTRACTION_FALLBACK,
    )
    return _parse(raw, ExtractedResultsBatch)


def run_prompt_1(theorem_statement: str, context: str = "") -> ResultAnalysis:
    user = f"""Aqui está o resultado:

{theorem_statement}

Contexto opcional:
{context}

Analise o resultado antes de tentar formalizá-lo, retornando os campos do schema ResultAnalysis.

IMPORTANTE: Siga EXATAMENTE estes tipos:
- likely_proof_mechanism: UMA string (ex: "Induction on n"), NÃO uma lista
- checkability: EXATAMENTE uma destas strings: "SYMBOLIC", "BOUNDED_COMPUTATIONAL", ou "NOT_READILY_CHECKABLE"
- subareas, quantified_objects, explicit_hypotheses, implicit_assumptions, ambiguities: listas de strings
- restatement, conclusion, formalized_statement: strings
- needs_networkx: true ou false
- in_scope: true ou false

Retorne um objeto JSON com TODOS estes campos."""

    raw = call_model(
        SYSTEM_MATH_RESEARCH,
        user + JSON_ONLY_SUFFIX,
        model=MODEL_ANALYSIS,
        max_tokens=MAX_TOKENS_ANALYSIS,
        stage="prompt_1_analysis",
        fallback_model=MODEL_ANALYSIS_FALLBACK,
    )
    return _parse(raw, ResultAnalysis)


def run_prompt_2(analysis: ResultAnalysis) -> ConjectureBatch:
    user = f"""A partir desta análise:

{analysis.model_dump_json(indent=2)}

Gere no máximo dez conjecturas candidatas (campo `candidates`), cada uma com name,
changed_component, statement, motivation, status, status_reason,
counterexample_direction, checkability, needs_networkx.

IMPORTANTE: Siga EXATAMENTE estes tipos:
- status: EXATAMENTE uma destas strings: "PLAUSIBLY_TRUE", "PLAUSIBLY_FALSE", ou "UNCERTAIN"
- checkability: EXATAMENTE uma destas strings: "SYMBOLIC", "BOUNDED_COMPUTATIONAL", ou "NOT_READILY_CHECKABLE"
- needs_networkx: true ou false
- counterexample_direction: string ou null

Ao final, preencha `top_three` com os nomes dos três candidatos mais promissores, em ordem."""

    raw = call_model(
        SYSTEM_MATH_RESEARCH,
        user + JSON_ONLY_SUFFIX,
        model=MODEL_CONJECTURES,
        max_tokens=MAX_TOKENS_CONJECTURES,
        stage="prompt_2_conjectures",
    )
    return _parse(raw, ConjectureBatch)


def run_prompt_3(selected_statement: str, definitions: str = "", model: str = None) -> VerificationPlan:
    actual_model = model or MODEL_VERIFICATION
    user = f"""Formalize esta afirmação como algo checável com Sympy (e networkx se envolver grafos):

{selected_statement}

Definições informais usadas:
{definitions}

Preencha o schema VerificationPlan: objects_and_representation, sympy_networkxFunctions,
reduces_to_symbolic_identity, ambiguities_to_resolve, code (código Python completo e
executável, com print/assert de PASS/FAIL), correspondence_table, verification_risks."""

    raw = call_model(
        SYSTEM_VERIFICATION,
        user + JSON_ONLY_SUFFIX,
        model=actual_model,
        max_tokens=MAX_TOKENS_VERIFICATION,
        stage="prompt_3_verification",
        fallback_model=(
            MODEL_VERIFICATION_FALLBACK
            if actual_model != MODEL_VERIFICATION_FALLBACK
            else None
        ),
    )
    return _parse(raw, VerificationPlan)


def run_prompt_6(working_code: str, mathematical_statement: str) -> ProofAttempt:
    max_code_chars = int(os.environ.get("PROMPT_6_MAX_CODE_CHARS", "20000"))
    bounded_code = working_code[:max_code_chars]
    if len(working_code) > max_code_chars:
        print(
            f"  [PROMPT 6] Código truncado de {len(working_code)} para "
            f"{max_code_chars} caracteres para reduzir latência."
        )

    user = f"""O seguinte código tem uma verificação Sympy/networkx funcionando:

{bounded_code}

O resultado informal correspondente é:
{mathematical_statement}

Tente avançar de evidência computacional para uma prova real. NÃO declare "provado" com
base apenas em checagem finita. Preencha o schema ProofAttempt: reduced_to_symbolic_identity,
proof_status ("proved" | "bounded_evidence_only" | "neither"), informal_proof, final_code,
techniques_used, unresolved_step."""

    raw = call_model(
        SYSTEM_PROVER,
        user + JSON_ONLY_SUFFIX,
        model=MODEL_PROOF,
        max_tokens=MAX_TOKENS_PROOF,
        stage="prompt_6_proof",
    )
    return _parse(raw, ProofAttempt)