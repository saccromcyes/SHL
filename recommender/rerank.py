"""Turn search candidates into a balanced shortlist, with a reason for each pick."""

from __future__ import annotations

from recommender.catalog import Assessment
from recommender.query import Requirements
from recommender.search import SearchResult

RERANK_SYSTEM = """You are an assessment consultant choosing tests for a hiring request.
From the numbered candidates, choose up to 10 that best assess what the request needs, best first.

Rules:
- Cover every skill in the requirements. A role that needs soft skills (collaboration, communication,
  leadership, customer handling) should include a personality, behaviour or situational test.
- Prefer tests whose job levels match the requested level.
- Do not choose two versions of the same test unless both add something.
- Only choose from the candidates. Never invent tests.
- Each reason is at most 15 words and says which requirement the test covers.

Return JSON only: {"ranking": [{"id": <candidate id>, "reason": "..."}]}"""


def describe(a: Assessment) -> str:
    duration = f"{a.minutes} min" if a.minutes else a.duration_label
    levels = ", ".join(a.job_levels[:4]) or "any level"
    return f"[{a.id}] {a.name} | {', '.join(a.test_type_names)} | {duration} | {levels} | {a.description[:180]}"


def llm_rerank(llm, model: str, req: Requirements, candidates: list[Assessment], limit: int) -> list[tuple[int, str]]:
    request = (
        f"Role: {req.summary}\nSkills to assess: {', '.join(req.skills) or 'see role'}\n"
        f"Job level: {req.job_level or 'not stated'}\n"
        f"Time limit per test: {str(req.max_minutes) + ' minutes' if req.max_minutes else 'none'}\n"
        f"Requested test types: {', '.join(req.test_types) or 'any'}"
    )
    user = f"{request}\n\nCandidates:\n" + "\n".join(describe(a) for a in candidates)
    data = llm.json(RERANK_SYSTEM, user, model, max_tokens=900)
    valid = {a.id for a in candidates}
    picked: list[tuple[int, str]] = []
    for row in data.get("ranking", []):
        try:
            item_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if item_id in valid and item_id not in {i for i, _ in picked}:
            picked.append((item_id, str(row.get("reason", "")).strip()[:160]))
    return picked[:limit]


def balanced_rerank(search: SearchResult, limit: int) -> list[tuple[int, str]]:
    """No-LLM fallback: take the best hit of each sub-query in turn, so a request for Python, SQL and
    JavaScript returns all three instead of five Python tests."""
    lists = list(search.per_query.values())
    picked: list[int] = []
    depth = 0
    while len(picked) < limit and any(depth < len(lst) for lst in lists):
        for lst in lists:
            if depth < len(lst) and lst[depth] not in picked:
                picked.append(lst[depth])
                if len(picked) == limit:
                    break
        depth += 1
    for item in search.ranked:
        if len(picked) >= limit:
            break
        if item not in picked:
            picked.append(item)
    return [(i, "") for i in picked[:limit]]


def build_battery(results: list[Assessment], budget: int) -> list[Assessment]:
    """Greedy selection of a test battery that fits the time budget, favouring variety of test types."""
    battery, used_types, total = [], set(), 0
    for prefer_new_types in (True, False):
        for a in results:
            if a in battery or a.minutes is None or total + a.minutes > budget:
                continue
            if prefer_new_types and set(a.test_types) <= used_types:
                continue
            battery.append(a)
            used_types |= set(a.test_types)
            total += a.minutes
    return sorted(battery, key=results.index)
