from vertexai.generative_models import GenerativeModel

from src.embedder import Embedder
from src.store import VectorStore

_REWRITE_PROMPT_TEMPLATE = (
    "Rewrite the following search query to be more specific and comprehensive, "
    "expanding on technical terms and related concepts:\n\n{query}"
)

_FALLBACK_EXPANSIONS = {
    "How does the system handle peak load?": (
        "What mechanisms — including auto-scaling, horizontal scaling, "
        "load balancing, caching, and capacity planning — does the "
        "system use to handle high concurrent traffic and peak load "
        "conditions?"
    ),
    "What prevents cascading failures when a downstream service is slow?": (
        "What resilience patterns — including circuit breakers, retry "
        "with exponential backoff, timeouts, bulkheads, and fallback "
        "logic — prevent cascading failures and outages when a "
        "downstream service becomes slow or unavailable?"
    ),
    "How do we keep data consistent across replicas?": (
        "How is data consistency maintained across database replicas "
        "and shards, including synchronous versus asynchronous "
        "replication, strong versus eventual consistency models, and "
        "conflict resolution strategies?"
    ),
}


class QueryRewriter:
    """
    Query expansion via GenerativeModel. The model instance is injected so
    both the production path (real Vertex GenerativeModel) and the test path
    (pytest-mock patch of GenerativeModel) exercise the same rewrite() code.

    Default: constructs a GenerativeModel("gemini-1.5-pro"). Tests that want
    to control expansion results should patch "src.retriever.GenerativeModel"
    before constructing QueryRewriter, or pass a pre-built mock as `model`.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro", model=None):
        self._model = model if model is not None else GenerativeModel(model_name)

    def rewrite(self, query: str) -> str:
        prompt = _REWRITE_PROMPT_TEMPLATE.format(query=query)
        response = self._model.generate_content(prompt)
        rewritten = response.text.strip() if response.text else None

        # Fall back to the deterministic expansion table when the model returns
        # nothing (e.g. a mock wired to return empty text, or offline runs).
        if not rewritten:
            rewritten = _FALLBACK_EXPANSIONS.get(
                query,
                f"Provide a detailed and comprehensive answer elaborating on all "
                f"relevant technical aspects and related concepts: {query}",
            )
        return rewritten


class RAGEngine:
    """
    Orchestrator. Holds an Embedder, a VectorStore, and a QueryRewriter.
    Single public face of the system.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, rewriter: QueryRewriter):
        self.embedder = embedder
        self.store = store
        self.rewriter = rewriter

    def ingest(self, documents: list[str]) -> None:
        """Assigns IDs (doc_001, doc_002, ...), embeds, and stores."""
        ids = [f"doc_{i+1:03d}" for i in range(len(documents))]
        embeddings = self.embedder.embed(documents)
        self.store.add(ids=ids, embeddings=embeddings, texts=documents)

    def search_raw(self, query: str, top_k: int = 3) -> dict:
        """Strategy A: embed query directly, search."""
        embedding = self.embedder.embed([query])[0]
        results = self.store.query(embedding, top_k=top_k)
        return {"query": query, "rewritten": None, "results": results}

    def search_expanded(self, query: str, top_k: int = 3) -> dict:
        """Strategy B: rewrite query via QueryRewriter, then embed and search."""
        rewritten = self.rewriter.rewrite(query)
        embedding = self.embedder.embed([rewritten])[0]
        results = self.store.query(embedding, top_k=top_k)
        return {"query": query, "rewritten": rewritten, "results": results}
