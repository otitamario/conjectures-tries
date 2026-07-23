"""
Estados do pipeline de conjectura->verificação.
Mesmo espírito do state machine do orquestrador Lean 4: cada
transição corresponde a UMA chamada de API (ou UMA execução local),
nunca as duas ao mesmo tempo, para que falhas sejam isoláveis e
repetíveis (retry) sem re-executar etapas já bem-sucedidas.
"""

from enum import Enum, auto


class State(Enum):
    FETCHED = auto()            # paper obtido (zbMATH/arXiv), ainda não analisado
    ANALYZED = auto()           # Prompt 1 concluído
    OUT_OF_SCOPE = auto()       # Prompt 1 determinou que não é combinatória/teoria dos números
    CONJECTURES_GENERATED = auto()  # Prompt 2 concluído
    PLAN_GENERATED = auto()     # Prompt 3 concluído (código proposto, ainda não rodado)
    EXECUTED_PASS = auto()      # código rodou e a asserção passou
    EXECUTED_FAIL = auto()      # código rodou e falhou ou deu erro
    REPAIR_ATTEMPTED = auto()   # nova versão do código gerada após falha
    PROOF_ATTEMPTED = auto()    # Prompt 6 concluído (revisão manual recomendada a partir daqui)
    DONE = auto()

    MAX_REPAIRS_EXCEEDED = auto()  # estado terminal de falha


# Transições válidas — usado para validar que o orquestrador não pula etapas
VALID_TRANSITIONS: dict[State, set[State]] = {
    State.FETCHED: {State.ANALYZED, State.OUT_OF_SCOPE},
    State.ANALYZED: {State.CONJECTURES_GENERATED},
    State.CONJECTURES_GENERATED: {State.PLAN_GENERATED},
    State.PLAN_GENERATED: {State.EXECUTED_PASS, State.EXECUTED_FAIL},
    State.EXECUTED_FAIL: {State.REPAIR_ATTEMPTED, State.MAX_REPAIRS_EXCEEDED},
    State.REPAIR_ATTEMPTED: {State.EXECUTED_PASS, State.EXECUTED_FAIL},
    State.EXECUTED_PASS: {State.PROOF_ATTEMPTED, State.DONE},
    State.PROOF_ATTEMPTED: {State.DONE},
}


def can_transition(current: State, target: State) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())
