"""
clinical_tools.py — Function Calling Tools para o LangGraph.
Implementa ferramentas especializadas que os agentes podem chamar.
"""
from langchain_core.tools import tool
from typing import Optional
import json


@tool
def assess_risk_level(
    symptoms: str,
    duration_days: int = 1,
    has_fever: bool = False,
    fever_temp: float = 37.0,
    has_chest_pain: bool = False,
    has_shortness_of_breath: bool = False,
) -> str:
    """
    Avalia o nível de risco clínico baseado em sintomas e parâmetros vitais.
    Use esta ferramenta quando precisar classificar urgência clínica objetivamente.

    Args:
        symptoms: Descrição dos sintomas em texto livre
        duration_days: Há quantos dias os sintomas estão presentes
        has_fever: Se o paciente tem febre
        fever_temp: Temperatura corporal em graus Celsius
        has_chest_pain: Se há dor no peito
        has_shortness_of_breath: Se há falta de ar

    Returns:
        JSON com classificação de risco e recomendação
    """
    risk_score = 0
    red_flags = []
    recommendations = []

    # Sinais críticos imediatos
    if has_chest_pain:
        risk_score += 40
        red_flags.append("Dor no peito detectada — risco cardiovascular")
        recommendations.append("Avaliar sinais de IAM: irradiação, sudorese, náusea")

    if has_shortness_of_breath:
        risk_score += 35
        red_flags.append("Falta de ar — possível comprometimento respiratório")
        recommendations.append("Verificar frequência respiratória e saturação O2")

    # Febre
    if has_fever:
        if fever_temp >= 40.0:
            risk_score += 30
            red_flags.append(f"Febre muito alta: {fever_temp}°C — atenção imediata")
        elif fever_temp >= 39.0:
            risk_score += 20
            red_flags.append(f"Febre alta: {fever_temp}°C")
        elif fever_temp >= 38.0:
            risk_score += 10
            recommendations.append("Antitérmico e hidratação adequada")

    # Duração
    if duration_days > 7:
        risk_score += 15
        recommendations.append("Sintomas persistentes > 7 dias — investigação necessária")
    elif duration_days > 3:
        risk_score += 5

    # Palavras-chave críticas nos sintomas
    critical_keywords = [
        "desmaiei", "não consigo respirar", "braço dormendo",
        "boca torta", "confusão mental", "sangue"
    ]
    for kw in critical_keywords:
        if kw in symptoms.lower():
            risk_score += 25
            red_flags.append(f"Sintoma crítico detectado: {kw}")

    # Classificação
    if risk_score >= 60 or has_chest_pain or has_shortness_of_breath:
        risk_level = "critico"
        care_level = "PS/SAMU imediatamente"
    elif risk_score >= 35:
        risk_level = "alto"
        care_level = "UPA ou PS em até 1 hora"
    elif risk_score >= 15:
        risk_level = "medio"
        care_level = "UBS ou telemedicina em até 4 horas"
    else:
        risk_level = "baixo"
        care_level = "Autocuidado — retornar se piora"

    return json.dumps({
        "risk_level": risk_level,
        "risk_score": risk_score,
        "red_flags": red_flags,
        "recommendations": recommendations,
        "care_level": care_level,
    }, ensure_ascii=False)


@tool
def get_patient_history(patient_id: str = "current") -> str:
    """
    Recupera histórico clínico simplificado do paciente para contextualizar a triagem.
    Em produção, conectaria ao prontuário eletrônico da Care Plus.

    Args:
        patient_id: ID do paciente (use 'current' para paciente atual)

    Returns:
        JSON com histórico resumido
    """
    # Simulação de dados do prontuário — em produção viria do sistema Care Plus
    simulated_history = {
        "patient_id": patient_id,
        "age_range": "35-45",
        "chronic_conditions": ["Hipertensão arterial (controlada)", "Rinite alérgica"],
        "current_medications": ["Losartana 50mg 1x/dia"],
        "allergies": ["Penicilina"],
        "last_consultation": "2025-03-15",
        "last_consultation_reason": "Check-up anual",
        "care_plus_plan": "Premium",
        "preferred_specialists": ["Clínica Geral", "Cardiologia"],
        "note": "SIMULADO — Em produção, dados reais do prontuário Care Plus"
    }

    return json.dumps(simulated_history, ensure_ascii=False, indent=2)


@tool
def check_drug_interactions(medications: str) -> str:
    """
    Verifica interações medicamentosas básicas para orientação de segurança.
    NÃO substitui avaliação farmacêutica ou médica.

    Args:
        medications: Lista de medicamentos separados por vírgula

    Returns:
        JSON com alertas de interação conhecidos
    """
    med_list = [m.strip().lower() for m in medications.split(",")]

    # Base simplificada de interações — em produção: Anvisa/Micromedex
    known_interactions = {
        ("aspirina", "ibuprofeno"): "MODERADO: AINEs combinados aumentam risco de sangramento GI",
        ("losartana", "espironolactona"): "MODERADO: Risco de hipercalemia — monitorar potássio",
        ("paracetamol", "álcool"): "ALTO: Hepatotoxicidade — evitar combinação",
        ("metformina", "álcool"): "MODERADO: Risco de acidose lática",
    }

    alerts = []
    for (med1, med2), description in known_interactions.items():
        if med1 in med_list and med2 in med_list:
            alerts.append({
                "medications": [med1, med2],
                "severity": description.split(":")[0],
                "description": description,
            })

    return json.dumps({
        "medications_checked": med_list,
        "interactions_found": len(alerts),
        "alerts": alerts,
        "disclaimer": "Verificação básica. Consulte sempre um farmacêutico ou médico.",
    }, ensure_ascii=False, indent=2)


@tool
def find_care_plus_service(
    service_type: str,
    urgency: str = "routine",
    specialty: str = "clinica_geral"
) -> str:
    """
    Encontra o serviço adequado na rede Care Plus baseado no tipo e urgência.

    Args:
        service_type: Tipo: 'telemedicina', 'ubs', 'upa', 'ps', 'especialista'
        urgency: Urgência: 'emergency', 'urgent', 'routine'
        specialty: Especialidade médica desejada

    Returns:
        JSON com instruções de acesso ao serviço
    """
    services = {
        "telemedicina": {
            "name": "Telemedicina Care Plus — App Blua",
            "access": "App Blua (iOS/Android) ou portal careplus.com.br",
            "wait_time": "Até 2 horas para consulta",
            "available_24h": True,
            "specialties": ["Clínica Geral", "Pediatria", "Dermatologia", "Ginecologia",
                           "Psicologia", "Nutrição", "Fisioterapia"],
            "how_to": "1. Baixe o App Blua | 2. Faça login com dados do plano | 3. Selecione especialidade | 4. Agende ou consulte agora"
        },
        "upa": {
            "name": "UPA — Unidade de Pronto Atendimento",
            "access": "Busque a UPA mais próxima ou ligue 192",
            "wait_time": "Variável — triagem por gravidade",
            "available_24h": True,
            "note": "Indicado para casos MODERADOS a ALTOS que não configuram emergência imediata"
        },
        "ps": {
            "name": "Pronto-Socorro",
            "access": "Hospital credenciado Care Plus mais próximo",
            "wait_time": "Triagem Manchester — priorizados por gravidade",
            "available_24h": True,
            "note": "Para emergências use o SAMU 192 ao invés de ir dirigindo"
        }
    }

    service = services.get(service_type.lower(), services["telemedicina"])
    service["care_plus_central"] = "0800 xxx-xxxx (consulte seu cartão)"

    return json.dumps(service, ensure_ascii=False, indent=2)


# Lista de todas as ferramentas disponíveis para os agentes
ALL_TOOLS = [assess_risk_level, get_patient_history, check_drug_interactions, find_care_plus_service]
