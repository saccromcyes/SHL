"""Turn a free-text request or job description into structured hiring requirements."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from recommender.text import tokenize

JOB_LEVELS = [
    "Entry-Level",
    "Graduate",
    "Mid-Professional",
    "Professional Individual Contributor",
    "Manager",
    "Director",
    "Executive",
]

PARSE_SYSTEM = """You turn a hiring request or job description into search requirements for an assessment catalog.
The catalog has knowledge tests (one technology or subject each), ability tests (numerical, verbal, inductive,
deductive reasoning), personality and behaviour questionnaires, situational judgement scenarios, and job simulations.

Return JSON only:
{
  "summary": "one short line describing the role",
  "skills": ["each distinct skill or trait to assess, 1-4 words each"],
  "search_queries": ["2 to 6 short catalog search phrases, one per skill area; include a behaviour/personality query when soft skills matter"],
  "max_minutes": null or an integer time limit per assessment if the request states one,
  "job_level": null or one of ["Entry-Level", "Graduate", "Mid-Professional", "Manager", "Director", "Executive"],
  "remote_testing": true only if the request requires remote/online testing, else false,
  "test_types": subset of ["A","B","C","K","P","S"] only if the request explicitly asks for that kind of test
    (A=ability/cognitive, B=situational judgement, C=competencies, K=knowledge, P=personality, S=simulation)
}"""

_TIME_PATTERNS = [
    r"(?:under|within|less than|below|max(?:imum)?(?: duration)?(?: of)?|duration(?: of| is)?|no more than|up to|at most|fit (?:in|into)|limit(?: of)?|in)\s+(\d{1,3})\s*(?:-?\s*)(minutes?|mins?|hours?|hrs?)\b",
    r"(\d{1,3})\s*(minutes?|mins?|hours?|hrs?)\s*(?:max(?:imum)?|or less|limit|time limit)\b",
]
_TYPE_HINTS = {
    "A": r"\b(cognitive|aptitude|reasoning|numerical|verbal|logical|general ability|mental ability|iq)\b",
    "P": r"\b(personality|behaviou?r(al)?|culture fit|motivation|traits?)\b",
    "B": r"\b(situational judge?ment|sjt|scenarios?)\b",
    "S": r"\b(simulations?|work sample|hands-on)\b",
}
_LEVEL_HINTS = [
    ("Graduate", r"\b(graduate|fresher|campus|trainee|intern)\b"),
    ("Entry-Level", r"\b(entry[- ]level|junior|beginner)\b"),
    ("Executive", r"\b(executive|c-suite|ceo|cfo|cto|vp)\b"),
    ("Director", r"\bdirector\b"),
    ("Manager", r"\b(manager|team lead|supervisor)\b"),
    ("Mid-Professional", r"\b(mid[- ]level|mid[- ]senior|experienced|senior)\b"),
]


@dataclass
class Requirements:
    summary: str
    skills: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    max_minutes: int | None = None
    job_level: str | None = None
    remote_testing: bool = False
    test_types: list[str] = field(default_factory=list)
    parsed_by: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_time_limit(text: str) -> int | None:
    lowered = text.lower()
    for pattern in _TIME_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            value, unit = int(match.group(1)), match.group(2)
            return value * 60 if unit.startswith("h") else value
    return None


def parse_with_rules(text: str, skill_vocab: set[str]) -> Requirements:
    lowered = text.lower()
    types = [code for code, pattern in _TYPE_HINTS.items() if re.search(pattern, lowered)]
    level = next((name for name, pattern in _LEVEL_HINTS if re.search(pattern, lowered)), None)
    skills = list(dict.fromkeys(t for t in tokenize(text) if t in skill_vocab))[:8]
    queries = [text[:500]] + skills
    if re.search(r"\b(collaborat\w*|teamwork|communicat\w*|stakeholder|interpersonal|leadership)\b", lowered):
        queries.append("interpersonal communication collaboration personality")
    return Requirements(
        summary=text.strip().split("\n")[0][:120],
        skills=skills,
        search_queries=queries,
        max_minutes=parse_time_limit(text),
        job_level=level,
        remote_testing=bool(re.search(r"\bremote(ly)? (testing|proctor\w*|assess\w*)\b", lowered)),
        test_types=types,
        parsed_by="rules",
    )


def parse_with_llm(llm, model: str, text: str, skill_vocab: set[str]) -> Requirements:
    data = llm.json(PARSE_SYSTEM, f"Request:\n{text[:6000]}", model, max_tokens=500)
    queries = [str(q).strip() for q in data.get("search_queries", []) if str(q).strip()][:6]
    max_minutes = data.get("max_minutes")
    rules_limit = parse_time_limit(text)  # the regex is more reliable than the model for explicit limits
    return Requirements(
        summary=str(data.get("summary", "")).strip()[:160] or text[:120],
        skills=[str(s).strip() for s in data.get("skills", []) if str(s).strip()][:10],
        search_queries=[text[:500]] + queries if len(text) < 500 else queries or [text[:500]],
        max_minutes=rules_limit
        if rules_limit is not None
        else (int(max_minutes) if isinstance(max_minutes, (int, float)) else None),
        job_level=data.get("job_level") if data.get("job_level") in JOB_LEVELS else None,
        remote_testing=bool(data.get("remote_testing", False)),
        test_types=[t for t in data.get("test_types", []) if t in {"A", "B", "C", "K", "P", "S"}],
        parsed_by="llm",
    )
