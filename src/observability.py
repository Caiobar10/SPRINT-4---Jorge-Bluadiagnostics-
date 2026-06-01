"""
observability.py — Integração com LangSmith para rastreabilidade de agentes.
Diferencial reconhecido pelo professor: grupos que adicionarem observabilidade
terão reconhecimento especial.

Para habilitar:
  1. Crie conta em smith.langchain.com
  2. Gere API key
  3. No .env:
     LANGCHAIN_TRACING_V2=true
     LANGCHAIN_API_KEY=sua_chave
     LANGCHAIN_PROJECT=bluadiagnostics-sprint4
"""
import os
from dotenv import load_dotenv

load_dotenv()


def setup_langsmith():
    """
    Configura LangSmith se as variáveis de ambiente estiverem presentes.
    Ativa rastreamento de cada nó do grafo, tools chamadas e documentos RAG recuperados.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()

    if api_key and tracing == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv(
            "LANGCHAIN_PROJECT", "bluadiagnostics-sprint4"
        )
        print(f"✅ LangSmith ativo — projeto: {os.environ['LANGCHAIN_PROJECT']}")
        print(f"   Dashboard: https://smith.langchain.com")
        return True
    else:
        print("ℹ️  LangSmith desativado (configure LANGCHAIN_API_KEY e LANGCHAIN_TRACING_V2=true no .env)")
        return False


def is_tracing_enabled() -> bool:
    """Verifica se o tracing está ativo."""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        and bool(os.getenv("LANGCHAIN_API_KEY"))
    )
