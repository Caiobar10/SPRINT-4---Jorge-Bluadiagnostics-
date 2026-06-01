"""
schemas.py — Structured Output com Pydantic v2
Todos os outputs dos agentes são validados por esses schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "baixo"
    MEDIUM = "medio"
    HIGH = "alto"
    CRITICAL = "critico"


class TriageOutput(BaseModel):
    """Output do Agente de Triagem — Etapa 1 do pipeline."""
    patient_summary: str = Field(
        description="Resumo objetivo dos sintomas relatados pelo paciente"
    )
    primary_symptoms: List[str] = Field(
        description="Lista dos sintomas principais identificados na conversa"
    )
    risk_level: RiskLevel = Field(
        description="Classificação inicial de risco: baixo, medio, alto ou critico"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Sinais de alerta graves que requerem atenção imediata (dor no peito, falta de ar severa, etc)"
    )
    requires_emergency: bool = Field(
        description="True se o paciente deve ir imediatamente a uma UPA/PS"
    )
    routing_intent: str = Field(
        description="Para qual agente rotear: 'prescription' para orientação clínica, 'emergency' para emergência, 'general' para consulta geral"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Nível de confiança da triagem entre 0 e 1"
    )


class ClinicalGuidance(BaseModel):
    """Output do Agente de Prescrição/Orientação Clínica — Etapa 2."""
    diagnosis_hypothesis: str = Field(
        description="Hipótese diagnóstica baseada nos sintomas (NÃO é diagnóstico definitivo)"
    )
    recommended_actions: List[str] = Field(
        description="Ações recomendadas em ordem de prioridade"
    )
    care_level: str = Field(
        description="Nível de atendimento recomendado: autocuidado, UBS, UPA, PS, SAMU"
    )
    retrieved_protocols: List[str] = Field(
        default_factory=list,
        description="Protocolos clínicos recuperados pelo RAG relevantes ao caso"
    )
    disclaimer: str = Field(
        description="Aviso legal obrigatório sobre limitações do sistema"
    )


class EmergencyResponse(BaseModel):
    """Output do Agente de Emergência — ativado quando risk_level=critico."""
    urgency_message: str = Field(
        description="Mensagem urgente e clara para o paciente"
    )
    emergency_contacts: List[str] = Field(
        description="Contatos de emergência relevantes (SAMU 192, PS mais próximo, etc)"
    )
    immediate_instructions: List[str] = Field(
        description="Instruções imediatas que o paciente deve seguir AGORA"
    )
    do_not_do: List[str] = Field(
        description="O que o paciente NÃO deve fazer enquanto aguarda atendimento"
    )


class SupervisorDecision(BaseModel):
    """Output do Supervisor — coordena os agentes."""
    next_agent: str = Field(
        description="Próximo agente a ser acionado: triage, prescription, emergency, human"
    )
    reasoning: str = Field(
        description="Justificativa da decisão de roteamento"
    )
    conversation_complete: bool = Field(
        description="True se o fluxo conversacional foi concluído adequadamente"
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Score de qualidade da resposta gerada"
    )


class GuardrailResult(BaseModel):
    """Resultado da validação de guardrails."""
    is_safe: bool = Field(description="True se o conteúdo passou em todos os guardrails")
    violated_rules: List[str] = Field(
        default_factory=list,
        description="Lista de regras violadas, se houver"
    )
    sanitized_content: Optional[str] = Field(
        default=None,
        description="Conteúdo sanitizado após filtragem, se necessário"
    )
    risk_category: Optional[str] = Field(
        default=None,
        description="Categoria do risco detectado: injection, out_of_scope, medical_advice, etc"
    )
