"""
triage_agent.py — Agente de Triagem (Etapa 1 do pipeline multi-agente).
Responsável por: coletar sintomas, classificar risco e definir roteamento.
"""
from src.llm_client import call_llm_structured, call_llm
from src.schemas import TriageOutput, RiskLevel
from src.rag.rag_engine import rag_engine
from typing import Dict, List, Optional


TRIAGE_AGENT_CONTEXT = """
<agent_role>
Você é o Agente de Triagem do BluaDiagnostics — Care Plus.

Experiência: 15 anos em triagem clínica de pronto-socorro e telemedicina
Especialidade: Classificação de Manchester e protocolo ABCDE
Tom de voz: Empático, calmo e objetivo. Nunca alarmista, mas sempre honesto.
Limitações: Não realiza diagnósticos definitivos. Não prescreve medicamentos.
</agent_role>

<triage_framework>
Você utiliza o Protocolo de Classificação de Manchester adaptado:
1. CRITICO (vermelho): risco imediato à vida → emergência
2. ALTO (laranja): risco potencial → avaliação urgente em até 1h
3. MÉDIO (amarelo): situação aguda mas estável → UBS em até 4h
4. BAIXO (verde): situação não urgente → agendamento em 48h
</triage_framework>

<output_instruction>
Analise os sintomas descritos e retorne APENAS JSON estruturado conforme schema.
Seja preciso na classificação de risco — errar para baixo pode custar uma vida.
</output_instruction>
"""


def run_triage(
    patient_message: str,
    conversation_history: Optional[List[Dict]] = None,
    rag_context: str = "",
) -> TriageOutput:
    """
    Executa a triagem do paciente.

    Args:
        patient_message: Mensagem atual do paciente com sintomas
        conversation_history: Histórico da conversa
        rag_context: Contexto recuperado pelo RAG

    Returns:
        TriageOutput com risco classificado e roteamento definido
    """
    context = TRIAGE_AGENT_CONTEXT
    if rag_context:
        context += f"\n\n{rag_context}"

    prompt = f"""
Analise os seguintes sintomas relatados pelo paciente e realize a triagem:

<patient_report>
{patient_message}
</patient_report>

{"<conversation_context>" + str(conversation_history[-3:]) + "</conversation_context>" if conversation_history else ""}

Execute a classificação de risco e determine o roteamento adequado.
Retorne o JSON estruturado com todos os campos obrigatórios.
"""

    result_dict = call_llm_structured(
        prompt,
        output_schema=TriageOutput,
        conversation_history=conversation_history,
        agent_context=context,
    )

    # Fallback se structured output falhar
    if not result_dict:
        return _fallback_triage(patient_message)

    try:
        return TriageOutput(**result_dict)
    except Exception:
        return _fallback_triage(patient_message)


def _fallback_triage(patient_message: str) -> TriageOutput:
    """Triagem de fallback quando o structured output falha."""
    text = patient_message.lower()

    # Detecção heurística de emergência
    emergency_keywords = [
        "dor no peito", "falta de ar", "desmaiei", "não consigo respirar",
        "dormência no braço", "face caída", "vomitando sangue"
    ]
    is_emergency = any(kw in text for kw in emergency_keywords)

    return TriageOutput(
        patient_summary=f"Paciente relata: {patient_message[:100]}...",
        primary_symptoms=["sintomas não classificados — avaliação manual necessária"],
        risk_level=RiskLevel.HIGH if is_emergency else RiskLevel.MEDIUM,
        red_flags=["avaliação automática falhou — revisar manualmente"],
        requires_emergency=is_emergency,
        routing_intent="emergency" if is_emergency else "prescription",
        confidence_score=0.3,
    )


def format_triage_response(triage: TriageOutput) -> str:
    """Formata o resultado da triagem para exibição ao paciente."""
    risk_emoji = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRITICAL: "🔴",
    }

    emoji = risk_emoji.get(triage.risk_level, "⚪")

    response = f"{emoji} **Classificação inicial: {triage.risk_level.value.upper()}**\n\n"
    response += f"📋 {triage.patient_summary}\n"

    if triage.red_flags:
        response += "\n⚠️ **Sinais de alerta identificados:**\n"
        for flag in triage.red_flags:
            response += f"  • {flag}\n"

    if triage.requires_emergency:
        response += "\n🚨 **ATENÇÃO: Recomendamos atendimento de emergência imediato.**\n"
        response += "Ligue SAMU **192** ou vá ao PS mais próximo.\n"

    return response
