"""Hybrid search over the catalog: BM25 + dense embeddings, fused with reciprocal rank fusion.

377 items fit in a 377 x 384 float matrix (~0.6 MB), so dense search is one matrix-vector
product. A vector database would add operational cost for no gain at this size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from recommender.catalog import Assessment
from recommender.text import tokenize


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    scores: dict[int, float] = {}
    for n, ranking in enumerate(rankings):
        w = weights[n] if weights else 1.0
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return scores


GENERIC_TERMS = frozenset(
    "new level advanced entry development developer engineering engineer report solution v1 v2 us uk business team "
    "data working work time digital process experience project manager management software web customer phone service "
    "basic general concept skill professional individual contributor front end interactive essential focus simulation "
    "application system systems program programming technical communication screen split".split()
)


@dataclass
class Filters:
    max_minutes: int | None = None
    remote_only: bool = False
    include_reports: bool = False
    include_untimed: bool = True

    def allows(self, a: Assessment) -> bool:
        if a.is_report and not self.include_reports:
            return False
        if self.remote_only and not a.remote:
            return False
        if self.max_minutes is not None:
            if a.minutes is None:
                return self.include_untimed
            return a.minutes <= self.max_minutes
        return True


@dataclass
class SearchResult:
    ranked: list[int]  # assessment ids, best first
    scores: dict[int, float]
    per_query: dict[str, list[int]] = field(default_factory=dict)  # sub-query -> its top hits


class CatalogIndex:
    def __init__(self, items: list[Assessment], embedder, rrf_k: int = 60):
        self.items = items
        self.by_id = {a.id: a for a in items}
        self.embedder = embedder
        self.rrf_k = rrf_k
        texts = [a.search_text() for a in items]
        # Names carry most of the signal ("Python (New)"), so they are repeated for BM25.
        self.bm25 = BM25Okapi([tokenize(f"{a.name} {a.name} {t}") or ["_"] for a, t in zip(items, texts)])
        self.matrix = embedder.documents(texts)
        self.skill_vocab = self._skill_vocabulary()

    def _skill_vocabulary(self) -> set[str]:
        """Words that name a specific assessment (java, kafka, nursing, typing...). Used to split a
        multi-skill request into one search per skill when no LLM is available."""
        df: dict[str, int] = {}
        for a in self.items:
            for t in set(tokenize(a.name)):
                df[t] = df.get(t, 0) + 1
        return {t for t, n in df.items() if n <= 8 and len(t) > 1 and t not in GENERIC_TERMS and not any(c.isdigit() for c in t)}

    def bm25_rank(self, query: str, allowed: set[int], limit: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query) or ["_"])
        order = np.argsort(scores)[::-1]
        return [int(i) for i in order if scores[i] > 0 and int(i) in allowed][:limit]

    def dense_rank(self, query_vec: np.ndarray, allowed: set[int], limit: int) -> list[int]:
        sims = self.matrix @ query_vec
        order = np.argsort(sims)[::-1]
        return [int(i) for i in order if int(i) in allowed][:limit]

    def search(
        self, queries: list[str], filters: Filters, mode: str = "hybrid", limit: int = 30, weights: list[float] | None = None
    ) -> SearchResult:
        allowed = {a.id for a in self.items if filters.allows(a)}
        vectors = self.embedder.queries(queries) if mode in ("dense", "hybrid") else None
        rankings, per_query = [], {}
        for i, q in enumerate(queries):
            lists = []
            single_term = len(tokenize(q)) == 1  # a bare skill name ("kafka") is a keyword lookup
            if mode in ("bm25", "hybrid") or single_term:
                lists.append(self.bm25_rank(q, allowed, limit))
            if mode in ("dense", "hybrid") and not single_term:
                lists.append(self.dense_rank(vectors[i], allowed, limit))
            fused = rrf(lists, self.rrf_k)
            ranking = sorted(fused, key=fused.get, reverse=True)[:limit]
            per_query[q] = ranking
            rankings.append(ranking)
        # Fusing the per-skill rankings rewards items relevant to several skills but still lets
        # every skill place its best matches near the top.
        scores = rrf(rankings, self.rrf_k, weights)
        return SearchResult(sorted(scores, key=scores.get, reverse=True)[:limit], scores, per_query)
