"""Measure each pipeline stage on the labeled queries with Recall@10 and MAP@10.

python -m eval.run_eval                 # every stage (LLM stages run only with LLM_API_KEY)
python -m eval.run_eval --no-llm        # retrieval stages only
python -m eval.run_eval --fail-under-recall 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import UTC, datetime

from recommender.config import ROOT, Settings
from recommender.service import Options, Recommender

LABELS = ROOT / "eval" / "labeled_queries.jsonl"
RESULTS = ROOT / "eval" / "results"
README = ROOT / "README.md"
K = 10

STAGES = [
    ("bm25", "Keyword search (BM25)", Options(retrieval="bm25", decompose=False, rerank="none", parse="rules")),
    ("dense", "Embedding search", Options(retrieval="dense", decompose=False, rerank="none", parse="rules")),
    (
        "hybrid",
        "Hybrid (RRF): default without an LLM",
        Options(retrieval="hybrid", decompose=False, rerank="none", parse="rules"),
    ),
    (
        "hybrid_multi",
        "Hybrid + one search per skill (rule parsing)",
        Options(retrieval="hybrid", decompose=True, rerank="none", parse="rules"),
    ),
    (
        "hybrid_multi_balanced",
        "Hybrid + per-skill search + balanced selection",
        Options(retrieval="hybrid", decompose=True, rerank="balanced", parse="rules"),
    ),
    (
        "full_llm",
        "LLM parsing + per-skill search + LLM rerank (full pipeline)",
        Options(retrieval="hybrid", decompose=True, rerank="llm"),
    ),
]


def recall_at_k(predicted: list[str], relevant: set[str], k: int = K) -> float:
    return len(set(predicted[:k]) & relevant) / len(relevant)


def average_precision_at_k(predicted: list[str], relevant: set[str], k: int = K) -> float:
    hits, total = 0, 0.0
    for i, name in enumerate(predicted[:k], start=1):
        if name in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def render_markdown(report: dict) -> str:
    lines = [
        f"_Last run {report['run_at'][:10]}, commit `{report['commit']}`, embeddings `{report['embedder']}`, "
        f"LLM `{report['llm'] or 'none'}`. {report['n_queries']} labeled queries, {report['n_labels']} relevant assessments._",
        "",
        "| Pipeline stage | Mean Recall@10 | MAP@10 |",
        "|---|---|---|",
    ]
    best = max(report["stages"], key=lambda r: (r["recall@10"], r["map@10"]))
    for row in report["stages"]:
        bold = "**" if row is best else ""
        lines.append(f"| {bold}{row['label']}{bold} | {bold}{row['recall@10']:.3f}{bold} | {row['map@10']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--fail-under-recall", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = Settings()
    if args.no_llm:
        settings.llm_api_key = ""
    rec = Recommender(settings)
    labeled = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]

    stages = []
    for key, label, opts in STAGES:
        if opts.rerank == "llm" and not rec.llm:
            continue
        recalls, aps, per_query, fallbacks = [], [], [], 0
        for item in labeled:
            result = rec.recommend(item["query"], opts)
            names = [r["name"] for r in result["results"]]
            relevant = set(item["relevant"])
            r, ap = recall_at_k(names, relevant), average_precision_at_k(names, relevant)
            recalls.append(r)
            aps.append(ap)
            per_query.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "recall@10": round(r, 3),
                    "returned": names,
                    "missed": sorted(relevant - set(names[:K])),
                }
            )
            if opts.rerank == "llm":
                if result["requirements"]["parsed_by"] != "llm" or result["pipeline"]["reranked_by"] != "llm":
                    fallbacks += 1
                time.sleep(args.pause)
        if fallbacks > 0.1 * len(labeled):
            print(
                f"FAIL: the LLM was unavailable for {fallbacks} of {len(labeled)} queries; results not published.",
                file=sys.stderr,
            )
            return 1
        row = {
            "stage": key,
            "label": label,
            "recall@10": round(sum(recalls) / len(recalls), 3),
            "map@10": round(sum(aps) / len(aps), 3),
        }
        stages.append({**row, "queries": per_query})
        print(f"{label:<32} Recall@10={row['recall@10']:.3f}  MAP@10={row['map@10']:.3f}", flush=True)

    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "embedder": getattr(rec.embedder, "name", "hashing"),
        "llm": settings.llm_model if rec.llm else None,
        "n_queries": len(labeled),
        "n_labels": sum(len(x["relevant"]) for x in labeled),
        "stages": stages,
    }
    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "latest.json").write_text(json.dumps(report, indent=2))
        md = render_markdown(report)
        (RESULTS / "latest.md").write_text(md)
        if README.exists():
            text = README.read_text()
            pattern = re.compile(r"(<!-- EVAL:START -->\n).*?(<!-- EVAL:END -->)", flags=re.DOTALL)
            README.write_text(pattern.sub(lambda m: m.group(1) + md + m.group(2), text))
    best = stages[-1]["recall@10"]
    if args.fail_under_recall is not None and best < args.fail_under_recall:
        print(f"FAIL: Recall@10 {best:.3f} < {args.fail_under_recall}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
