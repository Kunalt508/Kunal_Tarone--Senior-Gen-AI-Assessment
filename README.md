# Context-Aware RAG — Retrieval Strategy Benchmark

## Overview

This project implements a local Retrieval-Augmented Generation (RAG) retrieval engine that ingests a seven-paragraph technical corpus, stores embeddings in a local vector store, and benchmarks two retrieval strategies: **Strategy A** (raw vector search — embed the query directly) and **Strategy B** (AI-enhanced retrieval — rewrite the query via a `QueryRewriter` before embedding and searching). The pipeline ends at top-K retrieved chunks; there is no generation step. Running `pytest` executes all unit tests, integration tests, and produces `retrieval_benchmark.md` at the repo root.

## Setup

```bash
pip install -r requirements.txt
pytest
```

After `pytest` completes, `retrieval_benchmark.md` is written to the repo root with per-query metrics and a summary. Use `pytest -s` to also see the structured comparison table printed to stdout.

## Architecture

| File | Responsibility |
|------|---------------|
| `src/embedder.py` | `Embedder` — wraps `sentence-transformers` (`all-MiniLM-L6-v2`) locally; public interface mirrors Vertex AI `TextEmbeddingModel.get_embeddings()` for a production swap. L2-normalises all vectors before returning. Accepts `use_vertex=True` to call the real Vertex SDK (exercised by mocked unit tests). |
| `src/store.py` | `VectorStore` — ChromaDB-backed persistent collection using cosine space. Exposes `add()`, `query()` (returns cosine similarity scores in `[0, 1]`), and `reset()` for test isolation. `persist_dir` is configurable so tests use a separate directory from production data. |
| `src/retriever.py` | `QueryRewriter` — calls `GenerativeModel.generate_content()` to rewrite queries; accepts a pre-built `model` for dependency injection (used by tests to inject a mock). Falls back to a deterministic `_FALLBACK_EXPANSIONS` table when the model returns empty text. `RAGEngine` — orchestrates ingest, Strategy A (`search_raw`), and Strategy B (`search_expanded`). |
| `tests/test_pipeline.py` | Unit tests (Vertex SDK mocking via `pytest-mock`), pipeline integration tests, and a session-scoped benchmark that writes `retrieval_benchmark.md` and prints a structured comparison table to stdout. |

## Similarity Metric: Cosine vs Euclidean

### Why cosine?

`sentence-transformers` models are trained with a **cosine similarity objective** — the loss function explicitly minimises the angle between semantically similar texts in embedding space. This means the geometry the model "thinks in" is angular, not positional. The *direction* of a vector encodes meaning; the *magnitude* carries no semantic signal and varies with tokenisation, sentence length, and other surface properties.

### Euclidean on normalised vectors is equivalent — but misleading

After L2-normalisation (which `Embedder.embed()` always applies), every vector lies on the unit hypersphere. On the unit sphere, Euclidean distance and cosine distance are **monotonically equivalent**:

```
euclidean(u, v)² = 2 − 2·cos(u, v)
```

This means both metrics produce *identical rankings*. However, reporting a Euclidean distance on normalised vectors is still the wrong label: it obscures the underlying geometry and confuses readers who expect magnitude to matter.

### Conclusion: cosine

Cosine similarity is the honest label for this embedding space. It directly measures the angular relationship the model was trained to capture, it is scale-invariant by construction, and it maps naturally to the `[0, 1]` similarity scores reported in the benchmark.

ChromaDB is configured with `hnsw:space=cosine`, so distances from `collection.query()` are cosine distances in `[0, 2]`. The `VectorStore.query()` method converts them to similarities via `score = 1 − distance / 2`.

---

## Production Migration to Vertex AI Vector Search (Matching Engine)

The three abstraction boundaries (`Embedder`, `QueryRewriter`, `VectorStore`) are designed so that migrating to fully managed GCP services requires only swapping each component — `RAGEngine` and the benchmark code are untouched.

### Embedder swap

Replace the `sentence-transformers` call with:

```python
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embeddings = model.get_embeddings(batch)          # ingestion — batched for efficiency
embedding  = model.get_embeddings([query])[0]     # query — single call, low latency
```

The `Embedder` class already imports `TextEmbeddingModel` and has a `use_vertex=True` branch exercised by the mocked tests.

### QueryRewriter swap

`QueryRewriter` already calls `GenerativeModel.generate_content()` to rewrite queries — no code change is needed for the production path. The `_FALLBACK_EXPANSIONS` table only activates when the model returns empty text (e.g. in offline or mocked runs). To go live, supply valid Vertex AI credentials and remove the mock injection in tests:

```python
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content(prompt)
rewritten = response.text
```

`GenerativeModel` is already imported in `retriever.py` and patched in the unit tests.

### VectorStore swap

Replace ChromaDB with a Vertex AI Matching Engine client:

1. **Build index:** Export embeddings as a JSONL to GCS, then call `aiplatform.MatchingEngineIndex.create_tree_ah_index()`. Configure `DOT_PRODUCT_DISTANCE` on L2-normalised vectors — dot product on unit vectors is mathematically equivalent to cosine similarity, giving cosine semantics at billion-scale.
2. **Deploy:** `aiplatform.MatchingEngineIndexEndpoint.deploy_index()`.
3. **Query:** `index_endpoint.match(deployed_index_id, queries, num_neighbors=top_k)`.

### Index updates

- **Periodic corpus refresh:** batch `upsert_datapoints()` or rebuild the index from a fresh JSONL export.
- **Low-latency additions:** streaming `upsert_datapoints()` — changes propagate within minutes without a full rebuild.

### What does not change

`RAGEngine.ingest()`, `search_raw()`, and `search_expanded()` are unmodified. The benchmark test is unmodified. The abstractions earned their keep.
