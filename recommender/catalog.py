"""The SHL product catalog: parsing durations, test types and job levels into clean records."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

TEST_TYPES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


@dataclass
class Assessment:
    id: int
    name: str
    url: str
    description: str
    minutes: int | None  # None when untimed or unknown
    duration_label: str
    job_levels: list[str]
    remote: bool
    adaptive: bool
    test_types: list[str] = field(default_factory=list)
    is_report: bool = False

    @property
    def test_type_names(self) -> list[str]:
        return [TEST_TYPES.get(t, t) for t in self.test_types]

    def search_text(self) -> str:
        """What retrieval sees: name first, then what it measures, types and levels."""
        return (
            f"{self.name}. {self.description} "
            f"Test type: {', '.join(self.test_type_names)}. "
            f"Job levels: {', '.join(self.job_levels) or 'any'}."
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["test_type_names"] = self.test_type_names
        return data


def parse_minutes(raw: str) -> tuple[int | None, str]:
    """'max 45 mins' -> 45, '15 to 35 mins' -> 35 (worst case), 'Untimed, approx. 25 mins' -> 25,
    'Variable mins' / '' / 'TBC mins' -> None."""
    text = (raw or "").strip().lower()
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if not numbers or text.startswith("0 "):
        word = re.sub(r"\s*mins?$", "", text).strip(" -").capitalize()
        return None, word if word and not word.startswith("0") and word not in ("N/a", "-") else "Not stated"
    minutes = max(numbers) * (60 if "hour" in text else 1)
    if "untimed" in text or "approx" in text:
        return minutes, f"About {minutes} min (untimed)"
    if " to " in text:
        return minutes, f"{min(numbers)}–{minutes} min"
    return minutes, f"{minutes} min"


def load_catalog(path: str | Path) -> list[Assessment]:
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            name = re.sub(r"\s+", " ", row["name"]).strip()
            minutes, label = parse_minutes(row.get("duration", ""))
            codes = [c for c in re.split(r"[,\s]+", row.get("test_type", "")) if c]
            types = list(dict.fromkeys(codes))  # de-duplicate, keep order
            levels = [lvl.strip() for lvl in (row.get("roles") or "").split(",") if lvl.strip()]
            # Report products are outputs of other assessments, not tests a candidate takes.
            is_report = bool(re.search(r"\b(report|guide|cards|planner|profile)\b", name, re.I)) and minutes is None
            items.append(
                Assessment(
                    id=i,
                    name=name,
                    url=row["url"].strip(),
                    description=re.sub(r"\s+", " ", row.get("description", "")).strip(),
                    minutes=minutes,
                    duration_label=label,
                    job_levels=levels,
                    remote=row.get("remote_support", "").strip().lower() == "yes",
                    adaptive=row.get("adaptive_support", "").strip().lower() == "yes",
                    test_types=types,
                    is_report=is_report,
                )
            )
    return items
