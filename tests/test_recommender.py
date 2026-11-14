import pytest

from recommender.catalog import parse_minutes
from recommender.query import parse_time_limit, parse_with_llm, parse_with_rules
from recommender.rerank import build_battery, llm_rerank
from recommender.search import Filters, rrf
from recommender.service import Options
from tests.conftest import FakeLLM


@pytest.mark.parametrize(
    "raw, minutes",
    [
        ("30 mins", 30),
        ("max 45 mins", 45),
        ("15 to 35 mins", 35),
        ("Untimed, approx. 25 mins", 25),
        ("Variable mins", None),
        ("", None),
        ("0 mins", None),
        ("TBC mins", None),
    ],
)
def test_duration_parsing(raw, minutes):
    assert parse_minutes(raw)[0] == minutes


def test_catalog_flags_report_products(rec):
    names = {a.name: a for a in rec.items}
    assert len(rec.items) == 377
    assert names["OPQ Leadership Report"].is_report
    assert not names["Occupational Personality Questionnaire OPQ32r"].is_report
    assert names["Python (New)"].test_types == ["K"]


@pytest.mark.parametrize(
    "text, limit",
    [
        ("Each assessment must fit in 40 minutes.", 40),
        ("Need a package with max duration of 60 minutes", 60),
        ("time limit is less than 30 minutes", 30),
        ("can be completed within 1 hour", 60),
        ("no time constraint mentioned", None),
    ],
)
def test_time_limit_parsing(text, limit):
    assert parse_time_limit(text) == limit


def test_rule_parser_extracts_types_level_and_skills(rec):
    req = parse_with_rules(
        "Hiring a graduate analyst; screen with cognitive and personality tests and SQL.", rec.index.skill_vocab
    )
    assert set(req.test_types) == {"A", "P"}
    assert req.job_level == "Graduate"
    assert "sql" in req.skills


def test_weighted_rrf():
    scores = rrf([["a", "b"], ["b", "a"]], k=60, weights=[3.0, 1.0])
    assert scores["a"] > scores["b"]


def test_filters():
    from recommender.catalog import Assessment

    short = Assessment(1, "Short", "", "", 10, "10 min", [], True, False, ["K"])
    long = Assessment(2, "Long", "", "", 50, "50 min", [], False, False, ["A"])
    report = Assessment(3, "OPQ Report", "", "", None, "Not stated", [], True, False, ["P"], is_report=True)
    f = Filters(max_minutes=30)
    assert f.allows(short) and not f.allows(long) and not f.allows(report)
    assert not Filters(remote_only=True).allows(long)


def test_java_request_returns_java_tests_within_time_limit(rec):
    out = rec.recommend("Java developers who collaborate with business teams. Each test under 40 minutes.")
    names = [r["name"] for r in out["results"]]
    assert sum("Java" in n for n in names) >= 3
    assert all(r["minutes"] is None or r["minutes"] <= 40 for r in out["results"])
    assert out["requirements"]["max_minutes"] == 40
    assert not any(r["is_report"] for r in out["results"])


def test_balanced_selection_covers_every_skill(rec):
    names = [
        r["name"]
        for r in rec.recommend("Python, SQL and JavaScript developer", Options(decompose=True, rerank="balanced"))["results"]
    ]
    for skill in ("Python", "SQL", "JavaScript"):
        assert any(skill in n for n in names), skill


def test_battery_fits_budget(rec):
    out = rec.recommend("Front-end developer: React, HTML/CSS and JavaScript. Under 30 minutes per test.")
    assert out["battery"]["total_minutes"] <= out["battery"]["budget_minutes"] == 30
    assert out["battery"]["ids"]


def test_build_battery_prefers_variety():
    from recommender.catalog import Assessment

    k1 = Assessment(1, "K1", "", "", 10, "", [], True, False, ["K"])
    k2 = Assessment(2, "K2", "", "", 10, "", [], True, False, ["K"])
    p1 = Assessment(3, "P1", "", "", 25, "", [], True, False, ["P"])
    assert [a.name for a in build_battery([k1, k2, p1], 40)] == ["K1", "P1"]


def test_llm_parser_trusts_explicit_time_limit_over_model(rec):
    llm = FakeLLM(
        [
            {
                "summary": "Java dev",
                "skills": ["Java"],
                "search_queries": ["Java programming"],
                "max_minutes": 90,
                "test_types": ["K", "X"],
            }
        ]
    )
    req = parse_with_llm(llm, "m", "Java developer, tests must fit in 40 minutes", rec.index.skill_vocab)
    assert req.max_minutes == 40
    assert req.test_types == ["K"]
    assert req.parsed_by == "llm"


def test_llm_rerank_ignores_invented_or_duplicate_ids(rec):
    candidates = rec.items[:5]
    llm = FakeLLM(
        [
            {
                "ranking": [
                    {"id": 3, "reason": "good"},
                    {"id": 999, "reason": "invented"},
                    {"id": 3, "reason": "dup"},
                    {"id": "x"},
                    {"id": 1},
                ]
            }
        ]
    )
    picks = llm_rerank(llm, "m", parse_with_rules("x", set()), candidates, limit=10)
    assert [i for i, _ in picks] == [3, 1]


def test_recommender_falls_back_when_llm_errors(rec, monkeypatch):
    from recommender.llm import LLMError

    class Broken:
        def json(self, *a, **k):
            raise LLMError("429")

    monkeypatch.setattr(rec, "llm", Broken())
    out = rec.recommend("Python developer")
    assert out["requirements"]["parsed_by"] == "rules"
    assert out["pipeline"]["reranked_by"] == "balanced"
    monkeypatch.setattr(rec, "llm", None)


def test_decomposition_is_automatic_only_with_llm_parsing(rec):
    out = rec.recommend("Python, SQL and JavaScript developer")
    assert out["pipeline"]["parsed_by"] == "rules"
    assert out["pipeline"]["decomposed"] is False
    assert out["search_queries"] == ["Python, SQL and JavaScript developer"]


def test_catalog_embeddings_are_cached(tmp_path, rec):
    from recommender.search import CatalogIndex

    class CountingEmbedder:
        name = "counting"

        def __init__(self):
            self.calls = 0

        def documents(self, texts):
            import numpy as np

            self.calls += 1
            return np.ones((len(texts), 4), dtype=np.float32)

    cache = tmp_path / "emb.npz"
    first = CountingEmbedder()
    CatalogIndex(rec.items[:20], first, cache_path=str(cache))
    second = CountingEmbedder()
    index = CatalogIndex(rec.items[:20], second, cache_path=str(cache))
    assert (first.calls, second.calls) == (1, 0)
    assert index.matrix.shape == (20, 4)
    CatalogIndex(rec.items[:19], second, cache_path=str(cache))  # catalog changed: recompute
    assert second.calls == 1
