"""
rag_engine.py — Pipeline RAG completo com embeddings, vector store FAISS e retrieval.
Recupera protocolos clínicos relevantes para enriquecer o contexto dos agentes.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_PATH = Path(os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base"))
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))


# ============================================================
# DOCUMENTOS CLÍNICOS EMBUTIDOS (knowledge base)
# ============================================================

CLINICAL_DOCUMENTS = [
    {
        "id": "proto_dor_peito",
        "title": "Protocolo — Dor no Peito",
        "content": """
Protocolo Clínico Care Plus — Dor Torácica
Classificação: VERMELHO — Emergência Potencial

Sintomas de Alerta (encaminhar imediatamente para PS):
- Dor precordial com irradiação para braço esquerdo, mandíbula ou costas
- Dor associada a sudorese fria, náusea e vômito
- Dor em repouso com duração > 20 minutos
- Síncope ou pré-síncope associada

Conduta inicial:
1. Avaliar sinais vitais imediatamente
2. ECG em até 10 minutos se disponível
3. Acesso venoso periférico
4. O2 suplementar se SatO2 < 94%
5. Notificar médico plantonista imediatamente

Diagnósticos diferenciais:
- Síndrome Coronariana Aguda (IAM, Angina Instável)
- Tromboembolismo Pulmonar
- Dissecção Aórtica
- Pneumotórax Espontâneo
- Costocondrite (causa musculoesquelética — menor urgência)

Recomendação ao paciente: Ligar SAMU 192 ou ir ao PS imediatamente.
        """,
        "tags": ["dor", "peito", "torácica", "cardíaco", "emergência", "IAM"],
        "risk_level": "alto"
    },
    {
        "id": "proto_febre",
        "title": "Protocolo — Febre em Adultos",
        "content": """
Protocolo Clínico Care Plus — Manejo de Febre em Adultos

Definição: Temperatura axilar > 37,5°C ou temperatura retal > 38°C

Classificação por temperatura:
- 37,5°C - 38,5°C: Febre baixa (subfebril) — monitorar
- 38,5°C - 39,5°C: Febre moderada — investigar causa
- > 39,5°C: Febre alta — avaliar urgência
- > 40°C: Febre muito alta — atenção imediata

Sinais de alarme (encaminhar ao PS):
- Febre > 40°C sem resposta a antitérmicos
- Febre associada a rigidez de nuca, petéquias ou confusão mental
- Febre em imunossuprimidos, idosos > 80 anos ou neonatos
- Febre persistente > 7 dias sem diagnóstico

Orientações gerais:
1. Hidratação adequada (2-3L de água/dia)
2. Repouso relativo
3. Antitérmicos: Paracetamol 500-1000mg a cada 6h (máx 4g/dia)
4. Monitorar temperatura a cada 4 horas
5. Retornar se febre > 3 dias ou surgimento de novos sintomas

Causas mais comuns:
- Infecções respiratórias virais (60-70% dos casos)
- Infecções urinárias
- Infecções gastrointestinais
        """,
        "tags": ["febre", "temperatura", "antitérmico", "infecção", "viral"],
        "risk_level": "medio"
    },
    {
        "id": "proto_respiratorio",
        "title": "Protocolo — Sintomas Respiratórios",
        "content": """
Protocolo Clínico Care Plus — Avaliação Respiratória

Sinais de alerta respiratório (emergência):
- Frequência respiratória > 30 irpm ou < 10 irpm
- Saturação O2 < 92% em ar ambiente
- Uso de musculatura acessória
- Cianose central (lábios azulados)
- Estridor laríngeo
- Incapacidade de falar frases completas

Condições com falta de ar — triagem:
URGENTE (PS imediato):
- Broncoespasmo grave não responsivo a broncodilatador
- Pneumonia com comprometimento de todo o lobo
- Derrame pleural volumoso
- Exacerbação grave de DPOC

MODERADO (UBS em 24h):
- Tosse produtiva > 3 semanas
- Expectoração esverdeada com febre
- Piora progressiva da dispneia em asmáticos controlados

BAIXO RISCO (autocuidado + retorno se piora):
- Tosse seca pós-viral
- Rinorreia anterior com tosse
- Laringite viral

Orientações para pacientes asmáticos:
1. Usar broncodilatador de resgate conforme prescrito
2. Manter posição semi-sentada
3. Evitar esforço físico durante crise
4. Procurar PS se não melhorar em 20 minutos
        """,
        "tags": ["respiratório", "tosse", "falta de ar", "dispneia", "asma", "satO2"],
        "risk_level": "variavel"
    },
    {
        "id": "proto_neurologico",
        "title": "Protocolo — Sintomas Neurológicos",
        "content": """
Protocolo Clínico Care Plus — Sintomas Neurológicos

EMERGÊNCIA NEUROLÓGICA — Chamar SAMU 192 imediatamente:
- AVC: Face caída + Braço fraco + Fala alterada (teste FAST)
- Crise convulsiva pela primeira vez ou duração > 5 minutos
- Cefaleia "em trovoada" (a pior dor de cabeça da vida, início súbito)
- Perda de consciência súbita sem causa aparente
- Confusão mental aguda em idosos

Protocolo FAST para AVC:
F — Face: Peça para sorrir. Um lado caiu?
A — Arm: Levante os dois braços. Um cai?
S — Speech: Repita uma frase. A fala está arrastada?
T — Time: Tempo é essencial. Ligue SAMU 192 AGORA.

Cefaleia:
- Primária (tensional, enxaqueca): geralmente responde a analgésicos
- Secundária (meningite, AVC, tumor): requer investigação urgente
        """,
        "tags": ["neurológico", "AVC", "convulsão", "cefaleia", "desmaio", "confusão"],
        "risk_level": "alto"
    },
    {
        "id": "proto_gastrointestinal",
        "title": "Protocolo — Sintomas Gastrointestinais",
        "content": """
Protocolo Clínico Care Plus — Sintomas Gastrointestinais

Sinais de alarme (PS urgente):
- Sangramento digestivo (vômito com sangue, fezes escuras/melena)
- Dor abdominal intensa, contínua, associada a rigidez de parede
- Distensão abdominal progressiva com ausência de evacuações > 48h
- Icterícia progressiva com dor abdominal

Diarreia:
- Aguda (< 14 dias): geralmente autolimitada
- Sinais de desidratação: hidratação oral imediata
- Critérios de internação: diarreia > 10x/dia, febre > 38,5°C, sangue nas fezes

Hidratação oral (OMS):
- Solução: 1L de água + 1 colher de sopa de açúcar + 1 pitada de sal
- Ou: Pedialyte/Gatorade diluído
- Oferecer 200ml após cada evacuação líquida
        """,
        "tags": ["diarreia", "vômito", "dor abdominal", "gastrointestinal", "náusea"],
        "risk_level": "variavel"
    },
    {
        "id": "info_careplus",
        "title": "Informações — Rede Care Plus",
        "content": """
Care Plus — Plano de Saúde Premium
Sobre: Empresa brasileira fundada há 30+ anos, parte do grupo Bupa Internacional.
Beneficiários: 600.000+ em São Paulo | 8 clínicas próprias | Rede credenciada: +28.000 unidades

Serviços disponíveis:
- Telemedicina: App Blua — consultas em até 8 especialidades
- TytoCare: dispositivo de exame físico remoto
- Chatbot de triagem: disponível 24/7
- Rede credenciada: clínicas, hospitais, laboratórios em todo o Brasil

Como agendar:
1. App Blua (iOS/Android): agendamento em até 2h
2. Portal careplus.com.br: agendamento online
3. Central de atendimento: 0800 xxx-xxxx (24h)
4. WhatsApp: disponível para empresas

Especialidades pela telemedicina:
Clínica Geral, Pediatria, Dermatologia, Ginecologia,
Psicologia, Nutrição, Fisioterapia, Cardiologia
        """,
        "tags": ["careplus", "plano", "telemedicina", "blua", "agendamento"],
        "risk_level": "informativo"
    }
]


class RAGEngine:
    """
    Motor RAG (Retrieval-Augmented Generation) do BluaDiagnostics.
    Usa embeddings simples baseados em TF-IDF para recuperação sem dependência de GPU.
    Em produção, substituir por FAISS + sentence-transformers.
    """

    def __init__(self):
        self.documents = CLINICAL_DOCUMENTS
        self._build_index()

    def _build_index(self):
        """Constrói índice de busca invertido simples (fallback sem GPU)."""
        self.index = {}
        for doc in self.documents:
            # Indexa por tags e palavras do conteúdo
            all_terms = doc["tags"] + doc["title"].lower().split()
            content_words = doc["content"].lower().split()[:50]  # Primeiras 50 palavras
            all_terms.extend(content_words)

            for term in all_terms:
                term = term.strip(",.;:!?()[]")
                if len(term) > 2:
                    if term not in self.index:
                        self.index[term] = []
                    if doc["id"] not in self.index[term]:
                        self.index[term].append(doc["id"])

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Recupera os documentos mais relevantes para a query.

        Args:
            query: Texto da consulta (sintomas do paciente)
            top_k: Número máximo de documentos a retornar

        Returns:
            Lista de documentos relevantes ordenados por relevância
        """
        query_terms = query.lower().split()
        scores = {}

        for term in query_terms:
            term = term.strip(",.;:!?()[]")
            if term in self.index:
                for doc_id in self.index[term]:
                    scores[doc_id] = scores.get(doc_id, 0) + 1

        # Ordena por score e retorna top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc:
                result.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "relevance_score": score,
                    "risk_level": doc["risk_level"],
                })

        return result

    def format_context(self, documents: List[Dict]) -> str:
        """Formata documentos recuperados como contexto para o LLM."""
        if not documents:
            return ""

        context_parts = ["<retrieved_protocols>"]
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"""
<protocol_{i}>
<title>{doc['title']}</title>
<relevance>{doc['relevance_score']} pontos</relevance>
<content>
{doc['content'].strip()}
</content>
</protocol_{i}>
""")
        context_parts.append("</retrieved_protocols>")
        return "\n".join(context_parts)

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Busca documento por ID."""
        return next((d for d in self.documents if d["id"] == doc_id), None)


# Instância global do RAG
rag_engine = RAGEngine()
