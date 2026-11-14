"""The recommendation pipeline: parse -> multi-query hybrid search -> filter -> rerank -> battery."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from recommender.catalog import Assessment, load_catalog
from recommender.config import Settings
from recommender.llm import LLM, LLMError
from recommender.query import Requirements, parse_with_llm, parse_with_rules
from recommender.rerank import balanced_rerank, build_battery, llm_rerank
from recommender.search import CatalogIndex, Filters
from recommender.text import build_embedder

log = logging.getLogger(__name__)

TYPE_QUERIES = {
    "A": "cognitive ability reasoning aptitude test",
    "P": "personality questionnaire behaviour at work",
    "B": "situational judgement scenarios",
    "S": "job simulation work sample",
    "C": "competencies questionnaire",
}


@dataclass
class Options:
    """Pipeline switches, used by the evaluation to measure each stage's contribution."""

    retrieval: str = "hybrid"  # bm25 | dense | hybrid
    # One search per skill. "auto" uses it only when an LLM parsed the request: with rule-based
    # parsing it measured slightly worse than a single hybrid search (see eval/results).
    decompose: bool | str = "auto"
    apply_filters: bool = True
    rerank: str = "auto"  # auto (LLM if available) | llm | balanced | none


class Recommender:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.items = load_catalog(settings.catalog_path)
        self.embedder = build_embedder(settings.embedder, settings.embed_model, settings.model_cache_dir)
        self.index = CatalogIndex(self.items, self.embedder, settings.rrf_k)
        self.llm = LLM(settings) if settings.llm_enabled else None

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "rules"

    def parse(self, text: str) -> Requirements:
        if self.llm:
            try:
                return parse_with_llm(self.llm, self.settings.llm_fast_model, text, self.index.skill_vocab)
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("LLM parsing failed (%s); using rules", exc)
        return parse_with_rules(text, self.index.skill_vocab)

    def recommend(self, text: str, options: Options | None = None, overrides: dict | None = None) -> dict:
        opts = options or Options()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        req = self.parse(text)
        for key, value in (overrides or {}).items():  # user edits from the UI (e.g. time limit)
            if hasattr(req, key):
                setattr(req, key, value)
        timings["parse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        decompose = (req.parsed_by == "llm") if opts.decompose == "auto" else bool(opts.decompose)
        queries = req.search_queries if decompose else [text[:500]]
        if decompose:
            queries = queries + [TYPE_QUERIES[t] for t in req.test_types if t in TYPE_QUERIES]
        queries = list(dict.fromkeys(q for q in queries if q.strip())) or [text[:500]]
        filters = (
            Filters(max_minutes=req.max_minutes, remote_only=req.remote_testing)
            if opts.apply_filters
            else Filters(include_reports=True)
        )

        t0 = time.perf_counter()
        # The full request keeps as much weight as all the per-skill searches together.
        full = text[:500]
        sub = max(1, sum(q != full for q in queries))
        weights = [float(sub) if q == full else 1.0 for q in queries]
        search = self.index.search(queries, filters, opts.retrieval, self.settings.candidates, weights)
        timings["search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        limit = self.settings.max_results
        reranked_by = opts.rerank
        if opts.rerank in ("auto", "llm") and self.llm:
            try:
                picks = llm_rerank(self.llm, self.settings.llm_model, req, [self.index.by_id[i] for i in search.ranked], limit)
                reranked_by = "llm"
                if len(picks) < min(limit, len(search.ranked)):  # top up if the model chose fewer
                    chosen = {i for i, _ in picks}
                    picks += [(i, "") for i in search.ranked if i not in chosen][: limit - len(picks)]
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("LLM rerank failed (%s); using balanced rerank", exc)
                picks, reranked_by = balanced_rerank(search, limit), "balanced"
        elif opts.rerank in ("auto", "balanced", "llm"):
            picks, reranked_by = balanced_rerank(search, limit), "balanced"
        else:
            picks = [(i, "") for i in search.ranked[:limit]]
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        results: list[Assessment] = [self.index.by_id[i] for i, _ in picks]
        reasons = dict(picks)
        matched = {i: [q for q, hits in search.per_query.items() if i in hits[:10] and q != text[:500]] for i, _ in picks}
        battery_budget = req.max_minutes or 60
        battery = build_battery(results, battery_budget)
        timings["total_ms"] = round(sum(timings.values()), 1)

        return {
            "requirements": req.to_dict(),
            "search_queries": queries,
            "results": [
                {**a.to_dict(), "reason": reasons.get(a.id, ""), "matched_queries": matched.get(a.id, []), "rank": n}
                for n, a in enumerate(results, start=1)
            ],
            "battery": {
                "budget_minutes": battery_budget,
                "ids": [a.id for a in battery],
                "total_minutes": sum(a.minutes or 0 for a in battery),
            },
            "pipeline": {
                "parsed_by": req.parsed_by,
                "decomposed": decompose,
                "retrieval": opts.retrieval,
                "reranked_by": reranked_by,
                "candidates": len(search.ranked),
                "filtered_out": len(self.items) - sum(1 for a in self.items if filters.allows(a)),
            },
            "timings": timings,
            "mode": self.mode,
        }
