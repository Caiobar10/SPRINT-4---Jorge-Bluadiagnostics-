"""
guardrails.py — 3 Camadas de Guardrails contra Prompt Injection e conteúdo inseguro.

Camada 1: INPUT GUARD — sanitiza e valida entrada do usuário
Camada 2: SYSTEM GUARD — system prompt injetado em toda requisição (em llm_client.py)
Camada 3: OUTPUT GUARD — valida e filtra respostas do LLM antes de exibir ao paciente
"""
import re
from typing import Tuple
from src.schemas import GuardrailResult


# ============================================================
# PATTERNS DE DETECÇÃO
# ============================================================

INJECTION_PATTERNS = [
    # Tentativas clássicas de prompt injection
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(everything|all|your)\s+(you|previously|above)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|another|evil|unrestricted)",
    r"act\s+as\s+(if\s+you\s+(are|were)|a\s+different)",
    r"(pretend|imagine)\s+(you\s+are|to\s+be)\s+(a\s+)?(different|evil|unrestricted|jailbreak)",
    r"jailbreak|DAN\s+mode|developer\s+mode|god\s+mode",
    r"override\s+(your\s+)?(instructions?|rules?|guidelines?|training)",
    # DAN e variações em inglês
    r"\bDAN\b.*\b(without\s+restrictions?|no\s+rules?|unrestricted)",
    r"you\s+are\s+now\s+DAN",
    r"without\s+restrictions?\s+(AI|model|assistant)",
    # Em português
    r"ignore\s+(todas\s+as\s+)?(instruções|regras)\s+anteriores",
    r"você\s+agora\s+é\s+(um?\s+)?(outro|diferente|sem\s+restrições)",
    r"assistente\s+sem\s+restrições",
    r"sem\s+restrições\s+éticas",
    r"finja\s+(que\s+você\s+é|ser)\s+(um?\s+)?(médico|doutor|especialista\s+sem\s+ética)",
    r"(esqueça|ignore)\s+tudo\s+(que\s+você|o\s+que)\s+(aprendeu|foi\s+treinado)",
]

OUT_OF_SCOPE_PATTERNS = [
    # Assuntos completamente fora do domínio Care Plus
    r"\b(bitcoin|crypto|criptomoeda|investimento|bolsa\s+de\s+valores)\b",
    r"\b(receita\s+de\s+(bolo|culinária|comida))\b",
    r"\b(horóscopo|signo|tarô|astrologia)\b",
    r"\b(política|eleição|partido|presidente|governo)\b",
    r"\b(programação|código|python|javascript|html)\b",
    r"\b(piada|joke|entretenimento)\b",
]

MEDICAL_ADVICE_ABUSE_PATTERNS = [
    # Tentativas de obter prescrição médica real
    r"(me\s+receite|prescr[eia]+\s+(me\s+)?o\s+remédio|qual\s+remédio\s+(devo\s+)?tomar)",
    r"(posso\s+tomar|qual\s+a\s+dose|quanto\s+de)\s+\w+\s+(mg|ml|comprimido)",
    r"(combine[i]?\s+(remédios?|medicamentos?))",
    r"(diagnóstico\s+definitivo|você\s+(sabe|confirma)\s+que\s+tenho)",
]

HARMFUL_CONTENT_PATTERNS = [
    r"\b(suicídio|suicida|me\s+matar|quero\s+morrer|não\s+quero\s+viver)\b",
    r"\b(overdose\s+intencional|tomar\s+muitos\s+remédios\s+de\s+uma\s+vez)\b",
    r"\b(automutilação|me\s+machucar|cortar\s+os?\s+punhos?)\b",
]

SAFE_OUTPUT_PATTERNS = [
    # O LLM não deve gerar diagnósticos definitivos
    r"(você\s+tem\s+certamente|definitivamente\s+(você\s+tem|é)|diagnos[e]?\s+definitiv|diagnóstico\s+definitiv)",
    # O LLM não deve prescrever medicamentos específicos com dosagem
    r"\d+\s*(mg|ml|g)\s+de\s+\w+\s+(por|a\s+cada)\s+\d+\s*(hora|dia|horas)",
    # O LLM não deve dar informações de suicídio ou automutilação
    r"(como\s+se\s+machucar|método\s+de\s+suicídio)",
]


# ============================================================
# CAMADA 1: INPUT GUARD
# ============================================================

def input_guard(user_input: str) -> GuardrailResult:
    """
    Camada 1: Valida e sanitiza a entrada do usuário.
    Bloqueia injeções de prompt, conteúdo fora do escopo e tentativas de abuso.
    """
    text_lower = user_input.lower()
    violated_rules = []
    risk_category = None

    # Verifica injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            violated_rules.append(f"Prompt injection detectado: {pattern[:40]}...")
            risk_category = "injection"
            break

    # Verifica conteúdo de automutilação (prioridade máxima — redirecionar para emergência)
    for pattern in HARMFUL_CONTENT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            violated_rules.append("Conteúdo relacionado a automutilação ou ideação suicida detectado")
            risk_category = "crisis_mental_health"
            break

    # Verifica out-of-scope
    if not violated_rules:
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violated_rules.append(f"Assunto fora do escopo Care Plus: {pattern[:40]}...")
                risk_category = "out_of_scope"
                break

    # Verifica tentativas de abuso médico
    if not violated_rules:
        for pattern in MEDICAL_ADVICE_ABUSE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violated_rules.append("Tentativa de obter prescrição médica indevida")
                risk_category = "medical_prescription_abuse"
                break

    is_safe = len(violated_rules) == 0

    return GuardrailResult(
        is_safe=is_safe,
        violated_rules=violated_rules,
        sanitized_content=_sanitize_input(user_input) if is_safe else None,
        risk_category=risk_category,
    )


def _sanitize_input(text: str) -> str:
    """Remove caracteres suspeitos sem alterar o sentido clínico."""
    # Remove sequências de escape e caracteres de controle
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Remove tentativas de XML/tag injection
    text = re.sub(r"<\s*(script|system|instruction|prompt)\s*>", "", text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================
# CAMADA 3: OUTPUT GUARD
# ============================================================

def output_guard(llm_response: str) -> GuardrailResult:
    """
    Camada 3: Valida a resposta do LLM antes de exibir ao paciente.
    Garante que nenhum conteúdo perigoso ou diagnóstico definitivo chegue ao usuário.
    """
    text_lower = llm_response.lower()
    violated_rules = []
    risk_category = None

    for pattern in SAFE_OUTPUT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            violated_rules.append(f"Output inseguro detectado: {pattern[:50]}...")
            risk_category = "unsafe_output"
            break

    # Garante que disclaimer está presente em respostas clínicas
    clinical_keywords = ["sintoma", "doença", "tratamento", "medicamento", "diagnós"]
    has_clinical_content = any(kw in text_lower for kw in clinical_keywords)
    has_disclaimer = any(kw in text_lower for kw in [
        "não sou médico", "procure um", "consulte", "não substitui",
        "orientação médica", "atendimento profissional"
    ])

    if has_clinical_content and not has_disclaimer:
        violated_rules.append("Resposta clínica sem disclaimer obrigatório")
        risk_category = "missing_disclaimer"

    is_safe = len(violated_rules) == 0

    # Se houver violação, substitui por resposta segura
    sanitized = llm_response
    if not is_safe and risk_category == "missing_disclaimer":
        sanitized = llm_response + (
            "\n\n⚠️ *Importante: Este assistente virtual não substitui consulta médica. "
            "Para avaliação definitiva, procure um profissional de saúde.*"
        )
    elif not is_safe:
        sanitized = (
            "Posso ajudá-lo com orientações gerais sobre saúde dentro do contexto da Care Plus. "
            "Para diagnósticos e prescrições, é fundamental consultar um médico. "
            "Deseja que eu explique como agendar uma consulta pela rede Care Plus?"
        )

    return GuardrailResult(
        is_safe=is_safe,
        violated_rules=violated_rules,
        sanitized_content=sanitized,
        risk_category=risk_category,
    )


# ============================================================
# RESPOSTA PARA CASOS ESPECIAIS
# ============================================================

CRISIS_RESPONSE = """
🆘 **Atenção — Isso é importante.**

Percebi que você pode estar passando por um momento muito difícil.
Você não está sozinho(a) e há pessoas prontas para ajudar agora.

**CVV — Centro de Valorização da Vida**
📞 188 (24h, gratuito)
💬 cvv.org.br (chat online)

**CAPS — Centro de Atenção Psicossocial**
Procure o CAPS mais próximo da sua casa.

**Se estiver em perigo imediato:**
📞 SAMU: 192 | Bombeiros: 193

Por favor, entre em contato com alguém agora. ❤️
"""

OUT_OF_SCOPE_RESPONSE = """
Sou o BluaDiagnostics, assistente clínico virtual da Care Plus, e estou aqui para ajudar com questões de saúde.

Para esse tipo de assunto, infelizmente não tenho como ajudar. 😊

Posso auxiliá-lo com:
• Orientações sobre sintomas
• Triagem inicial de risco
• Informações sobre serviços da rede Care Plus
• Protocolos de cuidado preventivo

Como posso ajudá-lo com sua saúde hoje?
"""

INJECTION_BLOCKED_RESPONSE = """
Detectei algo incomum na sua mensagem. Sou um assistente clínico com foco exclusivo em saúde,
e não posso alterar minhas diretrizes de segurança.

Se precisar de ajuda médica, estou aqui para auxiliá-lo normalmente.
Para emergências: SAMU 192.
"""
