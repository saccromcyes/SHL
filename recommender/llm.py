"""JSON-mode calls to any OpenAI-compatible chat API."""

from __future__ import annotations

import json
import re

from recommender.config import Settings


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class LLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=settings.llm_timeout_s, max_retries=2
        )

    def json(self, system: str, user: str, model: str, max_tokens: int = 1200) -> dict:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return parse_json(response.choices[0].message.content or "{}")
        except Exception as exc:
            raise LLMError(str(exc)) from exc
