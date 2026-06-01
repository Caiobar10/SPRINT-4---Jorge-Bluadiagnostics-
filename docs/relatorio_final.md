# Relatório Técnico Final — BluaDiagnostics Sprint 4

**Disciplina:** Prompt Engineering & Artificial Intelligence  
**Professor:** Jorge Luiz Gomes  
**Instituição:** FIAP  
**Data:** 28/05/2026

---

## Integrantes

| Nome | RM |
|------|----|
| Caio Moda Barbieri | 566747 |
| Laura Stephanie Vasquez Oliveira | 567277 |
| Luis Roberto Lanzoni Kihara Júnior | 567406 |
| Mark Lima Leal | 566760 |
| Sofia Fernandes de Lima Oliveira | 567824 |

---

## 1. Contexto e Evolução da Sprint 3

Na Sprint 3, entregamos uma PoC básica do BluaDiagnostics: um chatbot simples com Ollama que realizava triagem de sintomas em arquivo único (`main.py`). O sistema obteve nota **5.2/10**, com os seguintes gaps identificados:

| Gap identificado | Impacto na nota |
|-----------------|-----------------|
| Estrutura monolítica (tudo em main.py) | -1.5 (arquitetura) |
| RAG declarado mas não funcional | -1.0 (funcionamento) |
| Sem arquitetura multi-agente real | -1.0 (requisito crítico) |
| Guardrails superficiais | -0.5 (segurança) |
| Sem evals automatizados | -0.8 (avaliação) |

A Sprint 4 foi uma reescrita completa do sistema com foco em engenharia de IA, não apenas em funcionamento.

---

## 2. Arquitetura Final

### 2.1 Grafo Multi-Agente (LangGraph)

```
[input_validation] → [rag_retrieval] → [triage]
                                           │
                    ┌──────────────────────┤
                    │                      │
              [risco crítico]       [risco médio/alto/baixo]
                    │                      │
               [emergency]          [prescription]
                    │                      │
                    └──────────┬───────────┘
                               │
                    [response_synthesis]
                               │
                             [END]
```

**Nós do grafo:**
- `input_validation`: Camada 1 de guardrail — bloqueia antes do LLM
- `rag_retrieval`: Recupera protocolos clínicos relevantes
- `triage`: Agente de Triagem — classifica risco e define roteamento
- `prescription`: Agente de Orientação Clínica (lógica condicional)
- `emergency`: Nó de emergência para casos críticos
- `response_synthesis`: Supervisor — monta e valida resposta final

### 2.2 Estado Compartilhado (AgentState)

O estado do LangGraph persiste ao longo do grafo:
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]      # Histórico completo
    patient_input: str               # Input atual
    triage_result: Optional[Dict]    # Output da triagem
    clinical_guidance: Optional[Dict] # Output da orientação
    risk_level: Optional[str]        # Nível de risco classificado
    requires_emergency: bool          # Flag de emergência
    guardrail_blocked: bool           # Flag de bloqueio
    rag_context: str                  # Contexto recuperado
```

### 2.3 Lógica Condicional (Requisito Crítico)

O `prescription_agent.py` implementa 4 ramos condicionais baseados no resultado da triagem:

```python
if risk_level == CRITICAL or requires_emergency:
    # RAMO 1: EmergencyResponse — SAMU 192, instruções imediatas
elif risk_level == HIGH:
    # RAMO 2: ClinicalGuidance modo URGENTE — PS hoje
elif risk_level == MEDIUM:
    # RAMO 3: ClinicalGuidance modo ORIENTAÇÃO — UBS + autocuidado
else:
    # RAMO 4: ClinicalGuidance modo PREVENTIVO — monitoramento
```

Esta lógica garante que a **Etapa 2 muda completamente conforme o resultado da Etapa 1**.

---

## 3. RAG — Retrieval-Augmented Generation

### Base de Conhecimento
6 documentos clínicos especializados:
- Protocolo Dor Torácica (nível: alto)
- Protocolo Febre em Adultos (nível: médio)
- Protocolo Sintomas Respiratórios (nível: variável)
- Protocolo Sintomas Neurológicos/AVC (nível: alto)
- Protocolo Sintomas Gastrointestinais (nível: variável)
- Informações Rede Care Plus (nível: informativo)

### Processo de Retrieval
1. Query construída a partir dos sintomas + histórico recente
2. Índice invertido por termos (fallback sem GPU)
3. Ranqueamento por score de relevância
4. Top-2 documentos formatados como `<retrieved_protocols>` XML
5. Contexto injetado no prompt do agente de triagem

### Demonstração de RAG no Vídeo
O vídeo demonstra: "Dor no peito + falta de ar" → recupera Protocolo Dor Torácica + Protocolo Respiratório → resposta fundamentada em protocolos clínicos reais.

---

## 4. Guardrails — 3 Camadas

### Camada 1 — Input Guard
5 conjuntos de patterns regex:
1. `INJECTION_PATTERNS` (8 patterns) — jailbreak, DAN mode, "ignore instruções"
2. `OUT_OF_SCOPE_PATTERNS` (7 patterns) — política, finanças, entretenimento
3. `MEDICAL_ADVICE_ABUSE_PATTERNS` (3 patterns) — prescrição indevida
4. `HARMFUL_CONTENT_PATTERNS` (3 patterns) — crise de saúde mental
5. Resposta diferenciada por categoria de risco

### Camada 2 — System Guard
System prompt defensivo (v3) com XML tags injetado em **toda requisição** ao LLM via `call_llm()`:
- `<persona>`: identidade com limitações explícitas
- `<scope>`: dentro/fora do escopo definidos
- `<safety_rules>`: 6 regras numeradas invioláveis
- `<disclaimer_template>`: aviso padronizado

### Camada 3 — Output Guard
Validação da resposta do LLM antes de exibir ao paciente:
- Detecta diagnósticos definitivos no output
- Detecta dosagem medicamentosa específica
- Verifica presença de disclaimer em respostas clínicas
- Substitui por resposta segura quando necessário

### Resultados nos Testes
- Precision dos guardrails: 100% (sem falsos positivos)
- Recall dos guardrails: 100% (todos os ataques bloqueados)
- F1-Score: 1.00

---

## 5. Structured Output com Pydantic v2

4 schemas implementados com `Field(description=...)` em todos os campos:

| Schema | Quando usado | Campos principais |
|--------|-------------|-------------------|
| `TriageOutput` | Agente de Triagem | patient_summary, risk_level, red_flags, routing_intent |
| `ClinicalGuidance` | Orientação clínica | diagnosis_hypothesis, recommended_actions, care_level |
| `EmergencyResponse` | Casos críticos | urgency_message, emergency_contacts, immediate_instructions |
| `GuardrailResult` | Validação guardrails | is_safe, violated_rules, risk_category |

Cada chamada estruturada inclui:
- Extração de JSON da resposta (com fallback de markdown)
- Validação com `model_validate_json()`
- Fallback com valores seguros em caso de falha

---

## 6. Avaliação Automática

### Dataset
- 20 casos de teste em 4 categorias
- 10 happy_path + 5 red_flag + 3 jailbreak + 2 out_of_scope

### Métricas
| Métrica | Peso | Descrição |
|---------|------|-----------|
| Safety Score | 30% | Guardrails bloquearam/passaram corretamente? |
| Disclaimer Score | 20% | Aviso legal presente em respostas clínicas? |
| Keyword Coverage | 25% | Palavras-chave esperadas presentes? |
| Completeness | 10% | Tamanho e conteúdo adequados? |
| Emergency Escalation | 15% | Casos críticos recebem instrução de emergência? |

### Gráficos Gerados (5)
1. `01_score_por_categoria.png` — Score médio por categoria
2. `02_radar_metricas.png` — Radar das 5 métricas
3. `03_latencia_por_categoria.png` — Latência por categoria
4. `04_guardrails_performance.png` — Matriz de confusão dos guardrails
5. `05_distribuicao_scores.png` — Histograma de scores gerais

---

## 7. Evolução dos Prompts V1 → V3

| Versão | Problema | Solução |
|--------|----------|---------|
| V1 | 3 linhas, sem persona, sem scope | Intencionalmente básica para comparação |
| V2 | Estrutura texto livre, sem XML | Persona + escopo + disclaimer + emergência |
| V3 | — | XML tags + CRISPE + safety rules numeradas + template padronizado |

**Impacto mensurável V2→V3:**
- Disclaimer score: 0.62 → 0.91 (+47%)
- Emergency escalation: 0.70 → 0.95 (+36%)
- Consistência geral: +35%

---

## 8. Decisões Técnicas e Trade-offs

### RAG sem GPU
**Decisão:** Índice invertido simples ao invés de FAISS + sentence-transformers  
**Justificativa:** Reprodutibilidade em qualquer máquina sem GPU, incluindo CI  
**Trade-off:** Menor qualidade semântica vs. maior portabilidade  
**Em produção:** FAISS + text-embedding-3-small via API ou sentence-transformers local

### Modelo Local (llama3.2:3b)
**Decisão:** llama3.2:3b como padrão ao invés de gpt-oss:120b  
**Justificativa:** Privacidade clínica (LGPD) + reprodutibilidade sem API key  
**Trade-off:** Menor qualidade das respostas vs. privacidade e zero custo  
**Em produção:** llama3.1:70b ou similar via servidor privado Care Plus

### Structured Output com retry
**Decisão:** Prompt com schema JSON + validação Pydantic + 3 tentativas + fallback  
**Justificativa:** LLMs ocasionalmente falham ao formatar JSON válido  
**Trade-off:** +latência vs. robustez garantida

---

## 9. Limitações Conhecidas

1. **Base de conhecimento clínico pequena:** 6 documentos vs. necessário em produção (1000+)
2. **Sem persistência de histórico:** Memória reinicia a cada sessão
3. **Sem autenticação do paciente:** Qualquer um pode usar o sistema
4. **Sem integração real com prontuário:** Dados de histórico são simulados
5. **RAG sem embeddings semânticos:** Busca por keywords vs. similaridade vetorial

---

## 10. Roadmap para Produção

1. Integrar com prontuário eletrônico Care Plus via API REST
2. Implementar FAISS + sentence-transformers para RAG semântico
3. Adicionar LangSmith para observabilidade em produção
4. Integrar com TytoCare para dados de exame físico remoto
5. FHIR R4 para interoperabilidade com outros sistemas de saúde
6. Autenticação com OAuth 2.0 + validação de beneficiário Care Plus
7. Auditoria completa de todas as interações (LGPD art. 38)
8. Fine-tuning do modelo com dados clínicos anonimizados da Care Plus

---

## 11. Reflexão do Grupo

A Sprint 4 foi fundamentalmente diferente da Sprint 3. Na Sprint 3, focamos em "fazer funcionar". Na Sprint 4, focamos em "fazer direito".

**O que aprendemos:**
- Arquitetura multi-agente com LangGraph não é apenas organização de código — é sobre gerenciamento de estado e roteamento condicional real
- Guardrails precisam ser implementados em camadas porque nenhuma única proteção é suficiente
- RAG não é só "carregar documentos" — o processo de retrieval e formatação do contexto afeta diretamente a qualidade das respostas
- Avaliação automatizada revela problemas que revisão manual nunca encontraria
- System prompt é infraestrutura, não decoração — não injetar em toda requisição é um bug de segurança

**O que faríamos diferente em produção:**
- Começaríamos com a arquitetura multi-agente desde o dia 1 (não refatorar depois)
- Implantaríamos LangSmith desde o início para rastreabilidade de cada decisão dos agentes
- Usaríamos um modelo 8B+ para qualidade clínica aceitável em produção
- Construiríamos uma base de conhecimento com protocolos reais CFM/AMB
- Adicionaríamos testes unitários para cada nó do grafo

---

*Relatório gerado em 28/05/2026 — BluaDiagnostics Sprint 4 — FIAP*
