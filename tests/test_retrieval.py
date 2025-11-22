from src.agent.rag import RAGStore

def test_rag_retrieve():
    store = RAGStore("data/index", "sentence-transformers/all-MiniLM-L6-v2")
    hits = store.retrieve("What is RAG?", k=2)
    assert len(hits) == 2
    assert all("text" in h for h in hits)
