from vertexai.language_models import TextEmbeddingModel  # noqa: F401 — patched in tests

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps the embedding model. Public shape matches Vertex AI's
    TextEmbeddingModel.get_embeddings() so production migration is a swap.
    Locally backed by sentence-transformers (all-MiniLM-L6-v2, 384-dim).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_vertex: bool = False):
        self.use_vertex = use_vertex
        self.model_name = model_name
        if not use_vertex:
            self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns L2-normalized vectors, shape (len(texts), 384)."""
        if self.use_vertex:
            # Production path — calls Vertex TextEmbeddingModel
            vertex_model = TextEmbeddingModel.from_pretrained(self.model_name)
            embeddings_response = vertex_model.get_embeddings(texts)
            vectors = np.array([e.values for e in embeddings_response], dtype=np.float32)
        else:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            vectors = vectors.astype(np.float32)

        # L2-normalize so cosine == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
