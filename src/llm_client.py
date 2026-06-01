"""
llm_client.py — Cliente LLM com system prompt injetado em TODAS as requisições.
O system prompt é carregado do arquivo e injetado automaticamente — Camada 2 de guardrail.

Suporta dois modos de operação:
  Local:  OLLAMA_BASE_URL=http://localhost:11434  (sem API key)
  Cloud:  OLLAMA_BASE_URL=https://ollama.com/api + OLLAMA_API_KEY=sua_chave
          OLLAMA_MODEL=gpt-oss:120b-cloud
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Optional, List, Dict
import json
import time

load_dotenv()

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"


def load_system_prompt() -> str:
    """Carrega o system prompt do arquivo. OBRIGATÓRIO em toda requisição ao LLM."""
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"System prompt não encontrado em {SYSTEM_PROMPT_PATH}")


def get_llm(temperature: Optional[float] = None) -> ChatOllama:
    """
    Retorna instância configurada do LLM.
    Detecta automaticamente se é modo local ou cloud pela presença de OLLAMA_API_KEY.
    """
    api_key = os.getenv("OLLAMA_API_KEY", "")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    kwargs = dict(
        model=model,
        base_url=base_url,
        temperature=temperature or float(os.getenv("MODEL_TEMPERATURE", "0.1")),
        num_predict=int(os.getenv("MODEL_MAX_TOKENS", "2048")),
    )

    # Ollama Cloud exige API key no header Authorization
    if api_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}

    return ChatOllama(**kwargs)


def call_llm(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    temperature: Optional[float] = None,
    inject_system_prompt: bool = True,
    agent_context: str = "",
    max_retries: int = 3,
) -> str:
    """
    Chama o LLM com system prompt SEMPRE injetado (Camada 2 de guardrail).
    Inclui retry automático e fallback.
    """
    llm = get_llm(temperature)
    messages = []

    # CAMADA 2: System prompt sempre injetado
    if inject_system_prompt:
        base_system = load_system_prompt()
        full_system = base_system
        if agent_context:
            full_system += f"\n\n<agent_context>\n{agent_context}\n</agent_context>"
        messages.append(SystemMessage(content=full_system))

    # Histórico de conversa
    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    # Retry com backoff exponencial
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[LLM] Tentativa {attempt + 1} falhou. Aguardando {wait_time}s...")
                time.sleep(wait_time)

    # Fallback: resposta segura quando LLM não está disponível
    print(f"[LLM] Todas as tentativas falharam: {last_error}")
    return (
        "Desculpe, estou com dificuldades técnicas no momento. "
        "Para questões urgentes de saúde, ligue para o SAMU (192) ou "
        "dirija-se ao pronto-socorro mais próximo."
    )


def call_llm_structured(
    user_message: str,
    output_schema: type,
    conversation_history: Optional[List[Dict]] = None,
    agent_context: str = "",
    max_retries: int = 3,
) -> dict:
    """
    Chama o LLM e força output estruturado em JSON compatível com schema Pydantic.
    Inclui validação e retry.
    """
    schema_str = json.dumps(output_schema.model_json_schema(), indent=2)
    structured_prompt = f"""
{user_message}

<output_format>
Responda EXCLUSIVAMENTE com um JSON válido seguindo este schema exato.
Não adicione texto antes ou depois do JSON.
Schema:
{schema_str}
</output_format>
"""

    for attempt in range(max_retries):
        try:
            raw_response = call_llm(
                structured_prompt,
                conversation_history=conversation_history,
                agent_context=agent_context,
            )

            # Extrai JSON da resposta
            json_str = raw_response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Valida com Pydantic
            parsed = output_schema.model_validate_json(json_str)
            return parsed.model_dump()

        except Exception as e:
            print(f"[STRUCTURED] Tentativa {attempt + 1} falhou: {e}")

    return {}
