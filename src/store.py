import numpy as np
import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB-backed vector store. Uses cosine similarity."""

    def __init__(self, collection_name: str = "rag_corpus", persist_dir: str = "./vector_store"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: np.ndarray, texts: list[str]) -> None:
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
        )

    def query(self, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Returns [{"id": str, "text": str, "score": float}, ...].

        score = cosine similarity in [0, 1], higher is better.
        Chroma cosine space returns distance in [0, 2]; similarity = 1 - distance/2
        is equivalent to (1 + cosine) / 2, clamped to [0, 1].
        """
        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            include=["documents", "distances"],
        )
        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        distances = results["distances"][0]
        for doc_id, text, dist in zip(ids, docs, distances):
            # Chroma cosine distance is in [0, 2]; convert to similarity in [0, 1]
            score = float(max(0.0, 1.0 - dist / 2.0))
            output.append({"id": doc_id, "text": text, "score": score})
        return output

    def reset(self) -> None:
        """Clears the collection — useful for test isolation."""
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )
