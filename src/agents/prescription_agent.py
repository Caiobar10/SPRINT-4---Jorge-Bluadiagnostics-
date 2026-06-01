"""
prescription_agent.py — Agente de Orientação Clínica (Etapa 2 CONDICIONAL).
A lógica deste agente MUDA conforme o resultado da Etapa 1 (Triagem):
- risk_level=critico → modo emergência (escalada)
- risk_level=alto → modo urgente (PS/UPA hoje)
- risk_level=medio → modo orientação (UBS + autocuidado)
- risk_level=baixo → modo preventivo (autocuidado + monitoramento)
"""
from src.llm_client import call_llm_structured, call_llm
from src.schemas import ClinicalGuidance, EmergencyResponse, TriageOutput, RiskLevel
from src.rag.rag_engine import rag_engine
from typing import Dict, List, Optional, Union


# Framework CRISPE aplicado na Etapa 3 do prompt
PRESCRIPTION_AGENT_BASE = """
<agent_role>
Você é o Agente de Orientação Clínica do BluaDiagnostics.

Capacity: Enfermeiro clínico sênior com 20 anos de experiência em medicina preventiva
Role: Orientador de saúde virtual da Care Plus
Insight: Você tem acesso a protocolos clínicos atualizados da Care Plus
Statement: Forneça orientações claras, práticas e seguras baseadas nos protocolos recuperados
Personality: Profissional, acolhedor, direto. Transmite segurança sem alarmar desnecessariamente.

Limitações absolutas:
- NÃO realiza diagnósticos definitivos
- NÃO prescreve medicamentos com dosagem específica sem orientação médica
- SEMPRE redireciona para atendimento presencial quando necessário
</agent_role>
"""

EMERGENCY_MODE_CONTEXT = """
<mode>EMERGÊNCIA</mode>
O paciente apresenta risco CRÍTICO ou ALTO com red flags confirmados.
Prioridade máxima: direcionar ao atendimento de emergência imediatamente.
Tom: urgente mas calmo. Não entre em pânico — transmita que há suporte disponível.
"""

URGENT_MODE_CONTEXT = """
<mode>URGENTE</mode>
O paciente apresenta risco ALTO que necessita avaliação médica hoje.
Oriente sobre onde buscar atendimento (UPA/PS) e o que fazer enquanto aguarda.
"""

GUIDANCE_MODE_CONTEXT = """
<mode>ORIENTAÇÃO CLÍNICA</mode>
O paciente apresenta risco MÉDIO — situação aguda mas estável.
Forneça orientações práticas de autocuidado + recomendação de consulta na UBS.
Baseie as orientações nos protocolos recuperados pelo RAG.
"""

PREVENTIVE_MODE_CONTEXT = """
<mode>PREVENTIVO</mode>
O paciente apresenta risco BAIXO.
Forneça orientações de autocuidado, monitoramento e quando buscar atendimento.
"""


def run_clinical_guidance(
    triage_result: TriageOutput,
    patient_message: str,
    conversation_history: Optional[List[Dict]] = None,
) -> Union[ClinicalGuidance, EmergencyResponse]:
    """
    Executa orientação clínica COM LÓGICA CONDICIONAL baseada no resultado da triagem.

    CONDICIONAL: A lógica muda completamente conforme triage_result.risk_level.
    Esta é a implementação do requisito de "chain condicional" do professor Jorge.

    Args:
        triage_result: Resultado da Etapa 1 (Triagem)
        patient_message: Mensagem do paciente
        conversation_history: Histórico da conversa

    Returns:
        EmergencyResponse se crítico, ClinicalGuidance nos demais casos
    """

    # Recupera documentos relevantes via RAG
    rag_docs = rag_engine.retrieve(patient_message + " " + " ".join(triage_result.primary_symptoms))
    rag_context = rag_engine.format_context(rag_docs)
    retrieved_titles = [d["title"] for d in rag_docs]

    # ============================================================
    # LÓGICA CONDICIONAL — Etapa 2 muda conforme Etapa 1
    # ============================================================

    if triage_result.risk_level == RiskLevel.CRITICAL or triage_result.requires_emergency:
        # RAMO 1: Emergência — retorna EmergencyResponse
        return _run_emergency_response(
            triage_result, patient_message, conversation_history, rag_context
        )

    elif triage_result.risk_level == RiskLevel.HIGH:
        # RAMO 2: Alto risco — orientação urgente com ênfase em buscar PS hoje
        mode_context = URGENT_MODE_CONTEXT
        care_level_hint = "UPA ou Pronto-Socorro hoje, sem espera"

    elif triage_result.risk_level == RiskLevel.MEDIUM:
        # RAMO 3: Médio risco — orientação clínica + UBS
        mode_context = GUIDANCE_MODE_CONTEXT
        care_level_hint = "UBS ou clínica da Care Plus nas próximas 4-12 horas"

    else:
        # RAMO 4: Baixo risco — autocuidado e monitoramento
        mode_context = PREVENTIVE_MODE_CONTEXT
        care_level_hint = "Autocuidado com monitoramento — retornar se piora"

    # Constrói contexto do agente para este ramo
    agent_context = PRESCRIPTION_AGENT_BASE + mode_context
    if rag_context:
        agent_context += f"\n\n{rag_context}"

    symptoms_str = "\n".join(f"  - {s}" for s in triage_result.primary_symptoms)
    flags_str = "\n".join(f"  - {f}" for f in triage_result.red_flags) if triage_result.red_flags else "  Nenhum"

    prompt = f"""
Forneça orientação clínica para o seguinte caso triado:

<triage_result>
  Risco: {triage_result.risk_level.value}
  Sintomas principais:
{symptoms_str}
  Red flags:
{flags_str}
  Nível de atendimento recomendado: {care_level_hint}
  Confiança da triagem: {triage_result.confidence_score:.0%}
</triage_result>

<patient_message>
{patient_message}
</patient_message>

Use os protocolos clínicos recuperados para fundamentar suas orientações.
Inclua o disclaimer obrigatório na resposta.
Retorne JSON estruturado.
"""

    result_dict = call_llm_structured(
        prompt,
        output_schema=ClinicalGuidance,
        conversation_history=conversation_history,
        agent_context=agent_context,
    )

    if not result_dict:
        return _fallback_guidance(triage_result, retrieved_titles)

    try:
        guidance = ClinicalGuidance(**result_dict)
        # Garante que os protocolos RAG estão referenciados
        if not guidance.retrieved_protocols and retrieved_titles:
            guidance.retrieved_protocols = retrieved_titles
        return guidance
    except Exception:
        return _fallback_guidance(triage_result, retrieved_titles)


def _run_emergency_response(
    triage: TriageOutput,
    patient_message: str,
    conversation_history: Optional[List[Dict]],
    rag_context: str,
) -> EmergencyResponse:
    """Gera resposta de emergência para casos críticos."""
    agent_context = PRESCRIPTION_AGENT_BASE + EMERGENCY_MODE_CONTEXT
    if rag_context:
        agent_context += f"\n\n{rag_context}"

    prompt = f"""
CASO CRÍTICO — O paciente pode estar em risco imediato de vida.

Sintomas: {', '.join(triage.primary_symptoms)}
Red flags: {', '.join(triage.red_flags)}

Gere uma resposta de emergência clara, urgente mas calma.
Inclua contatos de emergência relevantes para São Paulo/Brasil.
Retorne JSON estruturado.
"""

    result_dict = call_llm_structured(
        prompt,
        output_schema=EmergencyResponse,
        conversation_history=conversation_history,
        agent_context=agent_context,
    )

    if not result_dict:
        return EmergencyResponse(
            urgency_message="🚨 Situação de risco! Por favor, busque atendimento de emergência imediatamente.",
            emergency_contacts=["SAMU: 192", "Bombeiros: 193", "CVV: 188"],
            immediate_instructions=["Ligue 192 agora", "Não fique sozinho(a)"],
            do_not_do=["Não dirija sozinho(a)", "Não ignore os sintomas"],
        )

    return EmergencyResponse(**result_dict)


def _fallback_guidance(triage: TriageOutput, retrieved_titles: List[str]) -> ClinicalGuidance:
    """Orientação de fallback."""
    care_map = {
        RiskLevel.HIGH: "UPA ou Pronto-Socorro hoje",
        RiskLevel.MEDIUM: "UBS ou clínica Care Plus em até 12 horas",
        RiskLevel.LOW: "Autocuidado com monitoramento",
    }

    return ClinicalGuidance(
        diagnosis_hypothesis="Avaliação em andamento — dados insuficientes para hipótese diagnóstica",
        recommended_actions=[
            "Monitorar sintomas de hora em hora",
            "Manter hidratação adequada",
            "Procurar atendimento médico conforme nível de urgência",
        ],
        care_level=care_map.get(triage.risk_level, "Consulte um médico"),
        retrieved_protocols=retrieved_titles,
        disclaimer=(
            "⚠️ Este sistema não substitui avaliação médica presencial. "
            "As orientações são baseadas em protocolos gerais e não consideram "
            "seu histórico clínico completo. Consulte sempre um profissional de saúde."
        ),
    )


def format_guidance_response(guidance: Union[ClinicalGuidance, EmergencyResponse]) -> str:
    """Formata a orientação clínica para exibição."""
    if isinstance(guidance, EmergencyResponse):
        response = f"🚨 **{guidance.urgency_message}**\n\n"
        response += "**Contatos de emergência:**\n"
        for contact in guidance.emergency_contacts:
            response += f"  📞 {contact}\n"
        response += "\n**Faça AGORA:**\n"
        for inst in guidance.immediate_instructions:
            response += f"  ✅ {inst}\n"
        if guidance.do_not_do:
            response += "\n**NÃO faça:**\n"
            for item in guidance.do_not_do:
                response += f"  ❌ {item}\n"
        return response

    response = f"💊 **Orientação Clínica**\n\n"
    response += f"📋 **Hipótese:** {guidance.diagnosis_hypothesis}\n\n"
    response += f"🏥 **Nível de atendimento:** {guidance.care_level}\n\n"
    response += "**Ações recomendadas:**\n"
    for i, action in enumerate(guidance.recommended_actions, 1):
        response += f"  {i}. {action}\n"

    if guidance.retrieved_protocols:
        response += f"\n📚 *Baseado em: {', '.join(guidance.retrieved_protocols)}*\n"

    response += f"\n---\n{guidance.disclaimer}"
    return response
