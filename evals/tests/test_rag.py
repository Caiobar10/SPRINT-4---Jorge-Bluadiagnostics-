"""
test_rag.py — Testes unitários para o motor RAG.
Execute com: pytest evals/tests/ -v
"""
import pytest
import sys
sys.path.insert(0, '.')

from src.rag.rag_engine import RAGEngine, CLINICAL_DOCUMENTS


class TestRAGRetrieval:
    """Testa o retrieval do RAG."""

    @pytest.fixture
    def rag(self):
        return RAGEngine()

    def test_retrieves_chest_pain_protocol(self, rag):
        docs = rag.retrieve("dor no peito irradiação braço")
        assert len(docs) > 0
        titles = [d["title"] for d in docs]
        assert any("Dor no Peito" in t or "Torácica" in t for t in titles)

    def test_retrieves_fever_protocol(self, rag):
        docs = rag.retrieve("febre temperatura alta criança")
        assert len(docs) > 0
        titles = [d["title"] for d in docs]
        assert any("Febre" in t for t in titles)

    def test_retrieves_neurological_protocol(self, rag):
        docs = rag.retrieve("AVC convulsão boca torta")
        assert len(docs) > 0
        titles = [d["title"] for d in docs]
        assert any("Neurológico" in t or "AVC" in t for t in titles)

    def test_top_k_respected(self, rag):
        docs = rag.retrieve("sintomas gerais", top_k=1)
        assert len(docs) <= 1

    def test_top_k_default(self, rag):
        docs = rag.retrieve("febre dor peito respiração")
        assert len(docs) <= 2  # top_k padrão é 2

    def test_documents_have_required_fields(self, rag):
        docs = rag.retrieve("febre")
        for doc in docs:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc
            assert "relevance_score" in doc
            assert "risk_level" in doc

    def test_relevance_score_positive(self, rag):
        docs = rag.retrieve("febre temperatura")
        for doc in docs:
            assert doc["relevance_score"] > 0

    def test_empty_query_returns_empty(self, rag):
        docs = rag.retrieve("xyzabc123nonsense")
        assert len(docs) == 0 or docs[0]["relevance_score"] == 0

    def test_format_context_returns_xml(self, rag):
        docs = rag.retrieve("dor peito")
        context = rag.format_context(docs)
        assert "<retrieved_protocols>" in context
        assert "<title>" in context

    def test_format_context_empty_list(self, rag):
        context = rag.format_context([])
        assert context == ""

    def test_get_document_by_id(self, rag):
        doc = rag.get_document_by_id("proto_dor_peito")
        assert doc is not None
        assert doc["id"] == "proto_dor_peito"

    def test_get_nonexistent_document(self, rag):
        doc = rag.get_document_by_id("id_que_nao_existe")
        assert doc is None


class TestKnowledgeBase:
    """Testa a integridade da base de conhecimento."""

    def test_minimum_document_count(self):
        assert len(CLINICAL_DOCUMENTS) >= 5, "Deve ter pelo menos 5 documentos clínicos"

    def test_all_documents_have_required_fields(self):
        required_fields = ["id", "title", "content", "tags", "risk_level"]
        for doc in CLINICAL_DOCUMENTS:
            for field in required_fields:
                assert field in doc, f"Documento {doc.get('id', '?')} sem campo '{field}'"

    def test_all_documents_have_content(self):
        for doc in CLINICAL_DOCUMENTS:
            assert len(doc["content"]) > 50, f"Documento {doc['id']} com conteúdo muito curto"

    def test_all_documents_have_tags(self):
        for doc in CLINICAL_DOCUMENTS:
            assert len(doc["tags"]) > 0, f"Documento {doc['id']} sem tags"

    def test_unique_ids(self):
        ids = [doc["id"] for doc in CLINICAL_DOCUMENTS]
        assert len(ids) == len(set(ids)), "IDs duplicados na knowledge base"

    def test_risk_levels_valid(self):
        valid_levels = {"baixo", "medio", "alto", "critico", "variavel", "informativo"}
        for doc in CLINICAL_DOCUMENTS:
            assert doc["risk_level"] in valid_levels, (
                f"Nível de risco inválido: {doc['risk_level']} em {doc['id']}"
            )
