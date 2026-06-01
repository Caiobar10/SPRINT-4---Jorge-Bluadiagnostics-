"""
supervisor.py — Supervisor que orquestra agentes via LangGraph.
Implementa roteamento condicional, gerenciamento de estado e fluxo inteligente.
"""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.schemas import TriageOutput, ClinicalGuidance, EmergencyResponse, RiskLevel
from src.agents.triage_agent import run_triage, format_triage_response
from src.agents.prescription_agent import run_clinical_guidance, format_guidance_response
from src.guardrails import (
    input_guard, output_guard,
    CRISIS_RESPONSE, OUT_OF_SCOPE_RESPONSE, INJECTION_BLOCKED_RESPONSE
)
from src.rag.rag_engine import rag_engine


# ============================================================
# ESTADO DO GRAFO LANGGRAPH
# ============================================================

class AgentState(TypedDict):
    """Estado compartilhado entre todos os agentes do grafo."""
    messages: Annotated[List[BaseMessage], add_messages]
    patient_input: str
    triage_result: Optional[Dict]
    clinical_guidance: Optional[Dict]
    final_response: Optional[str]
    risk_level: Optional[str]
    requires_emergency: bool
    guardrail_blocked: bool
    guardrail_reason: Optional[str]
    rag_context: str
    conversation_complete: bool
    iteration_count: int


# ============================================================
# NÓS DO GRAFO
# ============================================================

def input_validation_node(state: AgentState) -> AgentState:
    """Nó 1: Valida entrada com guardrails (Camada 1)."""
    patient_input = state["patient_input"]
    guard_result = input_guard(patient_input)

    if not guard_result.is_safe:
        # Determina resposta adequada por tipo de violação
        if guard_result.risk_category == "crisis_mental_health":
            response = CRISIS_RESPONSE
        elif guard_result.risk_category == "injection":
            response = INJECTION_BLOCKED_RESPONSE
        elif guard_result.risk_category == "out_of_scope":
            response = OUT_OF_SCOPE_RESPONSE
        else:
            response = OUT_OF_SCOPE_RESPONSE

        return {
            **state,
            "guardrail_blocked": True,
            "guardrail_reason": guard_result.risk_category,
            "final_response": response,
            "conversation_complete": True,
        }

    return {
        **state,
        "guardrail_blocked": False,
        "patient_input": guard_result.sanitized_content or patient_input,
    }


def rag_retrieval_node(state: AgentState) -> AgentState:
    """Nó 2: Recupera documentos clínicos relevantes via RAG."""
    query = state["patient_input"]

    # Enriquece query com histórico recente
    recent_messages = state.get("messages", [])[-3:]
    history_text = " ".join([
        m.content for m in recent_messages
        if isinstance(m, HumanMessage)
    ])
    full_query = f"{query} {history_text}".strip()

    rag_docs = rag_engine.retrieve(full_query, top_k=2)
    rag_context = rag_engine.format_context(rag_docs)

    return {**state, "rag_context": rag_context}


def triage_node(state: AgentState) -> AgentState:
    """Nó 3: Executa triagem do paciente (Agente de Triagem)."""
    conversation_history = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in state.get("messages", [])[-6:]  # Últimas 6 mensagens
    ]

    triage = run_triage(
        patient_message=state["patient_input"],
        conversation_history=conversation_history,
        rag_context=state.get("rag_context", ""),
    )

    return {
        **state,
        "triage_result": triage.model_dump(),
        "risk_level": triage.risk_level.value,
        "requires_emergency": triage.requires_emergency,
        "messages": state["messages"] + [
            AIMessage(content=f"[TRIAGEM] Risco: {triage.risk_level.value}")
        ],
    }


def prescription_node(state: AgentState) -> AgentState:
    """Nó 4: Orientação clínica condicional (Agente de Prescrição)."""
    triage_dict = state["triage_result"]
    triage = TriageOutput(**triage_dict)

    conversation_history = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in state.get("messages", [])[-6:]
    ]

    guidance = run_clinical_guidance(
        triage_result=triage,
        patient_message=state["patient_input"],
        conversation_history=conversation_history,
    )

    return {
        **state,
        "clinical_guidance": guidance.model_dump(),
    }


def emergency_node(state: AgentState) -> AgentState:
    """Nó 5: Resposta de emergência (ativado apenas para casos críticos)."""
    # Já tratado dentro do prescription_node via lógica condicional
    # Este nó garante que a resposta de emergência seja priorizada
    return {**state, "conversation_complete": False}


def response_synthesis_node(state: AgentState) -> AgentState:
    """Nó 6: Supervisor sintetiza a resposta final e valida com output guard."""
    triage_dict = state.get("triage_result")
    guidance_dict = state.get("clinical_guidance")

    # Monta resposta
    response_parts = []

    if triage_dict:
        triage = TriageOutput(**triage_dict)
        triage_response = format_triage_response(triage)
        response_parts.append(triage_response)

    if guidance_dict:
        # Tenta EmergencyResponse primeiro
        try:
            guidance = EmergencyResponse(**guidance_dict)
        except Exception:
            try:
                guidance = ClinicalGuidance(**guidance_dict)
            except Exception:
                guidance = None

        if guidance:
            guidance_response = format_guidance_response(guidance)
            response_parts.append(guidance_response)

    final_response = "\n\n---\n\n".join(response_parts) if response_parts else (
        "Entendo seus sintomas. Pode me contar mais detalhes para que eu possa ajudá-lo melhor?"
    )

    # CAMADA 3: Output Guard
    output_check = output_guard(final_response)
    safe_response = output_check.sanitized_content or final_response

    # Adiciona footer padrão Care Plus
    safe_response += "\n\n*BluaDiagnostics — Care Plus | Para emergências: SAMU 192*"

    return {
        **state,
        "final_response": safe_response,
        "messages": state["messages"] + [AIMessage(content=safe_response)],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


# ============================================================
# ROTEAMENTO CONDICIONAL
# ============================================================

def should_block(state: AgentState) -> str:
    """Decide se bloqueia na camada de guardrail ou continua."""
    if state.get("guardrail_blocked"):
        return "blocked"
    return "continue"


def route_after_triage(state: AgentState) -> str:
    """Roteamento condicional baseado no resultado da triagem."""
    risk = state.get("risk_level", "medio")
    requires_emergency = state.get("requires_emergency", False)

    if requires_emergency or risk == "critico":
        return "emergency"
    else:
        return "prescription"


# ============================================================
# CONSTRUÇÃO DO GRAFO LANGGRAPH
# ============================================================

def build_graph() -> StateGraph:
    """Constrói o grafo de agentes com LangGraph."""
    graph = StateGraph(AgentState)

    # Adiciona nós
    graph.add_node("input_validation", input_validation_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("triage", triage_node)
    graph.add_node("prescription", prescription_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("response_synthesis", response_synthesis_node)

    # Define ponto de entrada
    graph.set_entry_point("input_validation")

    # Aresta condicional após validação
    graph.add_conditional_edges(
        "input_validation",
        should_block,
        {
            "blocked": END,  # Guardrail bloqueou — encerra
            "continue": "rag_retrieval",  # Continua para RAG
        }
    )

    # RAG → Triagem
    graph.add_edge("rag_retrieval", "triage")

    # Triagem → Roteamento condicional (prescrição ou emergência)
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "prescription": "prescription",
            "emergency": "emergency",
        }
    )

    # Ambos os ramos convergem para síntese
    graph.add_edge("prescription", "response_synthesis")
    graph.add_edge("emergency", "prescription")  # Emergency passa por prescription também
    graph.add_edge("response_synthesis", END)

    return graph.compile()


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

class BluaDiagnosticsAgent:
    """
    Agente principal do BluaDiagnostics.
    Orquestra todos os agentes via LangGraph com gerenciamento de estado.
    """

    def __init__(self):
        self.graph = build_graph()
        self.conversation_history: List[BaseMessage] = []

    def chat(self, user_message: str) -> str:
        """
        Processa mensagem do usuário pelo pipeline completo.

        Args:
            user_message: Mensagem do paciente

        Returns:
            Resposta do sistema
        """
        self.conversation_history.append(HumanMessage(content=user_message))

        initial_state = AgentState(
            messages=self.conversation_history.copy(),
            patient_input=user_message,
            triage_result=None,
            clinical_guidance=None,
            final_response=None,
            risk_level=None,
            requires_emergency=False,
            guardrail_blocked=False,
            guardrail_reason=None,
            rag_context="",
            conversation_complete=False,
            iteration_count=0,
        )

        result = self.graph.invoke(initial_state)

        response = result.get("final_response", "Desculpe, não consegui processar sua mensagem.")
        self.conversation_history.append(AIMessage(content=response))

        return response

    def reset(self):
        """Reinicia a conversa."""
        self.conversation_history = []
