"""Tokenisation and embedding models (FastEmbed ONNX, with a dependency-free fallback)."""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.#+/][a-z0-9+#]+)*\+*#?")
STOPWORDS = frozenset(
    (
        "a about all also an and any are as at be been being but by can could do does each for from has have "
        "having he her his how i if in into is it its looking me more most must my need needs of on or our "
        "should so some such that the their them they this those to under up us was we were what when where "
        "which who will with within would you your want hire hiring candidate candidates role job test tests "
        "assessment assessments minute minutes min mins"
    ).split()
)


def _fold(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    # Keeps tech tokens intact: c#, c++, .net -> "net", node.js, html/css, pl/sql.
    tokens = _TOKEN.findall(text.lower().replace(".net", " dotnet "))
    return [_fold(t) for t in tokens if t not in STOPWORDS]


class HashEmbedder:
    """Deterministic feature-hashing embedder for tests and offline use."""

    name = "hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._normalise(np.stack(list(self._model.embed(texts, batch_size=64))).astype(np.float32))

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._normalise(np.stack(list(self._model.query_embed(texts))).astype(np.float32))


def build_embedder(kind: str, model_name: str, cache_dir: str):
    if kind == "fastembed":
        try:
            return FastEmbedEmbedder(model_name, cache_dir)
        except Exception as exc:
            log.warning("FastEmbed unavailable (%s); using hash embeddings", exc)
    return HashEmbedder()
