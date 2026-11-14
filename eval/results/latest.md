_Last run 2026-11-14, commit `8281b16`, embeddings `BAAI/bge-small-en-v1.5`, LLM `none`. 28 labeled queries, 169 relevant assessments._

| Pipeline stage | Mean Recall@10 | MAP@10 |
|---|---|---|
| Keyword search (BM25) | 0.707 | 0.605 |
| Embedding search | 0.704 | 0.583 |
| **Hybrid (RRF): default without an LLM** | **0.782** | 0.692 |
| Hybrid + one search per skill (rule parsing) | 0.756 | 0.647 |
| Hybrid + per-skill search + balanced selection | 0.756 | 0.663 |
