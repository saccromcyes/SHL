_Last run 2026-11-14, commit `1b9c519`, embeddings `BAAI/bge-small-en-v1.5`, LLM `none`. 28 labeled queries, 169 relevant assessments._

| Pipeline stage | Mean Recall@10 | MAP@10 |
|---|---|---|
| Keyword search (BM25) | 0.707 | 0.605 |
| Embedding search | 0.704 | 0.583 |
| Hybrid (RRF) | 0.782 | 0.692 |
| + one search per skill | 0.756 | 0.647 |
| **+ balanced selection** | **0.756** | 0.661 |
