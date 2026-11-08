from __future__ import annotations

import pytest

from recommender.config import Settings
from recommender.service import Recommender


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, embedder="hash", llm_api_key="", rate_limit_per_10min=1000)


@pytest.fixture(scope="session")
def rec(settings) -> Recommender:
    return Recommender(settings)


class FakeLLM:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls = 0

    def json(self, system, user, model, max_tokens=1200):
        self.calls += 1
        return self.responses.pop(0)
