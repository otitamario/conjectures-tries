"""
Uma função por prompt do workflow. Cada função:
  1. monta o prompt (mesmo texto do documento original),
  2. injeta um bloco de instrução de saída (forçar JSON puro),
  3. chama a API,
  4. valida com o schema pydantic correspondente,
  5. levanta exceção clara se a validação falhar (para o orquestrador decidir re-tentar).

Adaptado para API Kimi (Moonshot AI) - compatível com OpenAI SDK.
Otimizado para context caching e escalonamento automático de modelos.
"""

import env  # noqa: F401  (carrega .env antes de qualquer os.environ.get abaixo)
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable
from openai import OpenAI
from schemas import (
    ExtractedResultsBatch, ResultAnalysis, ConjectureBatch, VerificationPlan, ProofAttempt,
)

JSON_ONLY_SUFFIX = (
    "\n\nResponda SOMENTE com um objeto JSON válido, sem texto antes ou depois, "
    "sem blocos de código markdown (sem ```)."
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO KIMI - ESTRATÉGIA HÍBRIDA COM ESCALONAMENTO
# ─────────────────────────────────────────────────────────────────────────────

# Modelos por tarefa (configurável via variáveis de ambiente)
MODEL_EXTRACTION = os.environ.get("PIPELINE_MODEL_P0", "kimi-k2-5")        # Prompt 0: extração
MODEL_ANALYSIS = os.environ.get("PIPELINE_MODEL_P1", "kimi-k2-6")           # Prompt 1: análise
MODEL_CONJECTURES = os.environ.get("PIPELINE_MODEL_P2", "kimi-k2-6")      # Prompt 2: conjecturas
MODEL_VERIFICATION = os.environ.get("PIPELINE_MODEL_P3", "kimi-k2-6")      # Prompt 3: verificação
MODEL_REPAIR_PRIMARY = os.environ.get("PIPELINE_MODEL_REPAIR", "kimi-k2-5")  # Repair: 1ª tentativa
MODEL_REPAIR_ESCALATION = os.environ.get("PIPELINE_MODEL_REPAIR_LAST", "kimi-k2-6")  # Repair: última
MODEL_PROOF = os.environ.get("PIPELINE_MODEL_P6", "kimi-k2-6")            # Prompt 6: prova

# Limite de tentativas com K2.5 antes de escalar para K2.6
REPAIR_ESCALATION_THRESHOLD = int(os.environ.get("REPAIR_ESCALATION_THRESHOLD", "2"))

# Cliente Kimi (API compatível com OpenAI)
_client = OpenAI(
    api_key=os.environ.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)

# Parâmetros padrão de geração
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2

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
# ESTATÍSTICAS DE USO (para otimização contínua)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelUsageStats:
    """Estatísticas de uso por modelo para análise de custo-efetividade."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    successes: int = 0
    failures: int = 0
    
    @property
    def estimated_cost(self) -> float:
        """Custo estimado em USD (sem cache)."""
        # Preços por 1M tokens
        prices = {
            "kimi-k2-5": {"input": 0.60, "output": 3.00},
            "kimi-k2-6": {"input": 0.95, "output": 4.00},
        }
        price = prices.get(self.model_name, {"input": 0.95, "output": 4.00})
        return (self.input_tokens / 1_000_000 * price["input"] + 
                self.output_tokens / 1_000_000 * price["output"])
    
    def __post_init__(self):
        self.model_name = ""  # será setado externamente


# Estatísticas globais (persistidas em memória durante a execução)
_stats: dict[str, ModelUsageStats] = {
    "kimi-k2-5": field(default_factory=lambda: _create_stat("kimi-k2-5")),
    "kimi-k2-6": field(default_factory=lambda: _create_stat("kimi-k2-6")),
}

def _create_stat(name: str) -> ModelUsageStats:
    stat = ModelUsageStats()
    stat.model_name = name
    return stat


def get_stats() -> dict[str, ModelUsageStats]:
    """Retorna estatísticas de uso para análise."""
    return _stats


def print_stats() -> None:
    """Imprime relatório de uso e custo estimado."""
    print("\n" + "=" * 70)
    print("RELATÓRIO DE USO DA API KIMI")
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
    temperature: float = DEFAULT_TEMPERATURE,
    track_stats: bool = True,
) -> str:
    """
    Chama a API Kimi com formato compatível OpenAI.
    
    Args:
        system_prompt: Instruções de sistema
        user_prompt: Prompt do usuário
        model: Nome do modelo (kimi-k2-5 ou kimi-k2-6)
        max_tokens: Máximo de tokens de saída
        temperature: Temperatura de amostragem
        track_stats: Se True, registra estatísticas de uso
    
    Returns:
        Texto da resposta do modelo
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Estimativa de tokens de input (aproximada: 1 token ≈ 4 chars em inglês, 2 em chinês,
    # mas para português/técnico usamos fator conservador)
    estimated_input_tokens = len(system_prompt + user_prompt) // 3

    start_time = time.time()
    try:
        response = _client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        
        result = response.choices[0].message.content
        
        # Atualiza estatísticas
        if track_stats and model in _stats:
            stat = _stats[model]
            stat.calls += 1
            stat.input_tokens += estimated_input_tokens
            # Estimativa conservadora: output é metade do max_tokens em média
            stat.output_tokens += len(result) // 3
            stat.successes += 1
        
        elapsed = time.time() - start_time
        print(f"  [API] {model} | {elapsed:.1f}s | input ~{estimated_input_tokens} tokens")
        
        return result
        
    except Exception as e:
        # Registra falha
        if track_stats and model in _stats:
            _stats[model].failures += 1
        
        elapsed = time.time() - start_time
        print(f"  [API] {model} | FALHA após {elapsed:.1f}s | {type(e).__name__}: {e}")
        raise


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
    
    Estratégia:
      - Tentativa 0 (original): K2.6
      - Repairs 1-N: K2.5 nas primeiras tentativas, K2.6 na última
    """
    is_repair = repair_attempt > 0
    
    if not is_repair:
        # Primeira tentativa: sempre K2.6 para máxima qualidade
        model = MODEL_VERIFICATION
        user = f"""Formalize esta afirmação como algo checável com Sympy (e networkx se envolver grafos):

{selected_statement}

Definições informais usadas:
{definitions}

Preencha o schema VerificationPlan: objects_and_representation, sympy_networkxFunctions,
reduces_to_symbolic_identity, ambiguities_to_resolve, code (código Python completo e
executável, com print/assert de PASS/FAIL), correspondence_table, verification_risks."""
    
    else:
        # Repair: decide modelo baseado na tentativa
        remaining = max_repairs - repair_attempt + 1
        
        if remaining <= 1 and repair_attempt >= REPAIR_ESCALATION_THRESHOLD:
            # Última tentativa ou já falhou várias vezes com K2.5: escala para K2.6
            model = MODEL_REPAIR_ESCALATION
            print(f"  [REPAIR] Escalonando para {model} (tentativa {repair_attempt}, "
                  f"threshold={REPAIR_ESCALATION_THRESHOLD})")
        else:
            # Ainda tem tentativas sobrando: usa K2.5 para economizar
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
    
    raw = call_model(SYSTEM_MATH_RESEARCH, user + JSON_ONLY_SUFFIX, model=model)
    return _parse(raw, VerificationPlan)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS DO WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def _parse(raw: str, schema):
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    return schema.model_validate(json.loads(cleaned))


def run_prompt_0(paper_text: str, max_chars: int = 150_000) -> ExtractedResultsBatch:
    """
    Prompt 0: Extração de teoremas do paper.
    Modelo: K2.5 (tarefa simples, economia de 37% vs K2.6)
    """
    truncated = paper_text[:max_chars]
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
    
    raw = call_model(SYSTEM_EXTRACTOR, user + JSON_ONLY_SUFFIX, model=MODEL_EXTRACTION)
    return _parse(raw, ExtractedResultsBatch)


def run_prompt_1(theorem_statement: str, context: str = "") -> ResultAnalysis:
    """
    Prompt 1: Análise do resultado.
    Modelo: K2.6 (requer raciocínio matemático profundo)
    """
    user = f"""Aqui está o resultado:

{theorem_statement}

Contexto opcional:
{context}

Analise o resultado antes de tentar formalizá-lo, retornando os campos do schema ResultAnalysis
(restatement, subareas, quantified_objects, explicit_hypotheses, implicit_assumptions,
conclusion, likely_proof_mechanism, checkability, needs_networkx, ambiguities,
formalized_statement, in_scope)."""
    
    raw = call_model(SYSTEM_MATH_RESEARCH, user + JSON_ONLY_SUFFIX, model=MODEL_ANALYSIS)
    return _parse(raw, ResultAnalysis)


def run_prompt_2(analysis: ResultAnalysis) -> ConjectureBatch:
    """
    Prompt 2: Geração de conjecturas.
    Modelo: K2.6 (criatividade + rigor matemático)
    """
    user = f"""A partir desta análise:

{analysis.model_dump_json(indent=2)}

Gere no máximo dez conjecturas candidatas (campo `candidates`), cada uma com name,
changed_component, statement, motivation, status, status_reason,
counterexample_direction, checkability, needs_networkx. Ao final, preencha
`top_three` com os nomes dos três candidatos mais promissores, em ordem."""
    
    raw = call_model(SYSTEM_MATH_RESEARCH, user + JSON_ONLY_SUFFIX, model=MODEL_CONJECTURES)
    return _parse(raw, ConjectureBatch)


def run_prompt_3(selected_statement: str, definitions: str = "", model: str = None) -> VerificationPlan:
    """
    Prompt 3: Plano de verificação (primeira tentativa).
    Modelo: K2.6 (geração de código complexo)
    
    Nota: Para repairs, use run_prompt_3_with_repair() no orquestrador.
    """
    actual_model = model or MODEL_VERIFICATION
    user = f"""Formalize esta afirmação como algo checável com Sympy (e networkx se envolver grafos):

{selected_statement}

Definições informais usadas:
{definitions}

Preencha o schema VerificationPlan: objects_and_representation, sympy_networkxFunctions,
reduces_to_symbolic_identity, ambiguities_to_resolve, code (código Python completo e
executável, com print/assert de PASS/FAIL), correspondence_table, verification_risks."""
    
    raw = call_model(SYSTEM_MATH_RESEARCH, user + JSON_ONLY_SUFFIX, model=actual_model)
    return _parse(raw, VerificationPlan)


def run_prompt_6(working_code: str, mathematical_statement: str) -> ProofAttempt:
    """
    Prompt 6: Tentativa de prova genuína.
    Modelo: K2.6 (maior capacidade de reasoning)
    """
    user = f"""O seguinte código tem uma verificação Sympy/networkx funcionando:

{working_code}

O resultado informal correspondente é:
{mathematical_statement}

Tente avançar de evidência computacional para uma prova real. NÃO declare "provado" com
base apenas em checagem finita. Preencha o schema ProofAttempt: reduced_to_symbolic_identity,
proof_status ("proved" | "bounded_evidence_only" | "neither"), informal_proof, final_code,
techniques_used, unresolved_step."""
    
    raw = call_model(SYSTEM_MATH_RESEARCH, user + JSON_ONLY_SUFFIX, model=MODEL_PROOF)
    return _parse(raw, ProofAttempt)