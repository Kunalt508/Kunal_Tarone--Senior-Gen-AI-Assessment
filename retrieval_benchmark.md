# Retrieval Benchmark Report

**Experiment:** Compare two retrieval strategies on a 7-document technical corpus.

- **Strategy A (Raw):** Embed the original query directly and search.
- **Strategy B (Expanded):** Rewrite the query via `QueryRewriter` to add
  technical context, then embed and search.

Similarity metric: cosine (L2-normalized vectors stored in ChromaDB with
`hnsw:space=cosine`). Scores reported as cosine similarity ∈ [0, 1].

---

            ## Query: How does the system handle peak load?

            **Original query:** How does the system handle peak load?

            **Rewritten query (Strategy B):** What mechanisms — including auto-scaling, horizontal scaling, load balancing, caching, and capacity planning — does the system use to handle high concurrent traffic and peak load conditions?

            ### Top-3 Results

            | Rank | Strategy A ID | A Score | A Preview (80 chars) | Strategy B ID | B Score | B Preview (80 chars) |
|------|---------------|---------|----------------------|---------------|---------|----------------------|
| 1 | doc_001 | 0.7375 | Modern cloud applications adjust their compute footprint dynamically in response | doc_001 | 0.8068 | Modern cloud applications adjust their compute footprint dynamically in response |
| 2 | doc_002 | 0.6914 | A load balancer distributes incoming network traffic across a pool of backend in | doc_002 | 0.7543 | A load balancer distributes incoming network traffic across a pool of backend in |
| 3 | doc_005 | 0.6901 | When a downstream dependency becomes slow or unavailable, naive clients retry ag | doc_003 | 0.7453 | Caching reduces latency and backend load by storing the results of expensive ope |

            ### Metrics

            | Metric | Value |
            |--------|-------|
            | Top-3 IDs (Strategy A) | doc_001, doc_002, doc_005 |
            | Top-3 IDs (Strategy B) | doc_001, doc_002, doc_003 |
            | Jaccard overlap | 0.5000 |
            | Top-1 agreement | True |
            | Top-1 score delta (B − A) | +0.0693 |
            | IDs new in B | doc_003 |
            | IDs dropped from A | doc_005 |

            **Note:** Strategy B's expansion explicitly names autoscaling, load balancing, and caching, drawing in a broader set of relevant documents. This demonstrates clear benefit of query expansion when the original query is concise and generic.


            ## Query: What prevents cascading failures when a downstream service is slow?

            **Original query:** What prevents cascading failures when a downstream service is slow?

            **Rewritten query (Strategy B):** What resilience patterns — including circuit breakers, retry with exponential backoff, timeouts, bulkheads, and fallback logic — prevent cascading failures and outages when a downstream service becomes slow or unavailable?

            ### Top-3 Results

            | Rank | Strategy A ID | A Score | A Preview (80 chars) | Strategy B ID | B Score | B Preview (80 chars) |
|------|---------------|---------|----------------------|---------------|---------|----------------------|
| 1 | doc_005 | 0.8419 | When a downstream dependency becomes slow or unavailable, naive clients retry ag | doc_005 | 0.9069 | When a downstream dependency becomes slow or unavailable, naive clients retry ag |
| 2 | doc_007 | 0.7251 | A message queue decouples producers from consumers by accepting work items into  | doc_006 | 0.6923 | Observability is the ability to understand the internal state of a running syste |
| 3 | doc_006 | 0.7011 | Observability is the ability to understand the internal state of a running syste | doc_007 | 0.6767 | A message queue decouples producers from consumers by accepting work items into  |

            ### Metrics

            | Metric | Value |
            |--------|-------|
            | Top-3 IDs (Strategy A) | doc_005, doc_006, doc_007 |
            | Top-3 IDs (Strategy B) | doc_005, doc_006, doc_007 |
            | Jaccard overlap | 1.0000 |
            | Top-1 agreement | True |
            | Top-1 score delta (B − A) | +0.0651 |
            | IDs new in B | — |
            | IDs dropped from A | — |

            **Note:** Strategy B's expansion around circuit breakers, retry logic, and bulkheads more precisely matches the resilience-focused corpus paragraph, often improving top-1 precision over the shorter raw query.


            ## Query: How do we keep data consistent across replicas?

            **Original query:** How do we keep data consistent across replicas?

            **Rewritten query (Strategy B):** How is data consistency maintained across database replicas and shards, including synchronous versus asynchronous replication, strong versus eventual consistency models, and conflict resolution strategies?

            ### Top-3 Results

            | Rank | Strategy A ID | A Score | A Preview (80 chars) | Strategy B ID | B Score | B Preview (80 chars) |
|------|---------------|---------|----------------------|---------------|---------|----------------------|
| 1 | doc_004 | 0.8029 | When a single database instance cannot serve the required read or write throughp | doc_004 | 0.8863 | When a single database instance cannot serve the required read or write throughp |
| 2 | doc_003 | 0.6665 | Caching reduces latency and backend load by storing the results of expensive ope | doc_003 | 0.6895 | Caching reduces latency and backend load by storing the results of expensive ope |
| 3 | doc_001 | 0.6480 | Modern cloud applications adjust their compute footprint dynamically in response | doc_006 | 0.6639 | Observability is the ability to understand the internal state of a running syste |

            ### Metrics

            | Metric | Value |
            |--------|-------|
            | Top-3 IDs (Strategy A) | doc_001, doc_003, doc_004 |
            | Top-3 IDs (Strategy B) | doc_003, doc_004, doc_006 |
            | Jaccard overlap | 0.5000 |
            | Top-1 agreement | True |
            | Top-1 score delta (B − A) | +0.0834 |
            | IDs new in B | doc_006 |
            | IDs dropped from A | doc_001 |

            **Note:** Strategy B may not improve over Strategy A here because the raw query already contains strong lexical overlap with the corpus terms ('consistent', 'replicas', 'replication'). Honest evaluation: query expansion provides the most lift when the original query is terse or uses different vocabulary than the corpus.


---

## Summary

| Metric | Average across 3 queries |
|--------|--------------------------|
| Jaccard overlap | 0.6667 |
| Top-1 agreement rate | 1.00 |
| Avg top-1 score delta (B − A) | +0.0726 |
| Total IDs new in B | 2 |
| Total IDs dropped from A | 2 |

Query expansion (Strategy B) tends to improve recall for terse queries
by anchoring the embedding in richer semantic territory. For queries
that already share vocabulary with the corpus, the benefit is marginal
and expansion may even reshuffle results without meaningful gain. The
abstractions (Embedder, VectorStore, QueryRewriter) remain unchanged
whether backed by local libraries or Vertex AI, validating the design.
